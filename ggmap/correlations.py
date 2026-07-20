import sys
import time
from datetime import datetime
import itertools
import re
import os

import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, spearmanr, pearsonr
import matplotlib.pyplot as plt
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa
from skbio.tree import TreeNode
from skbio import OrdinationResults
from scipy.cluster.hierarchy import ward
from tqdm import tqdm
import inspect
import textwrap

from ggmap.execute import _executor
from ggmap import settings


# From this answer on stack overflow, I'm calculating Cramer's V:
# https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram
# %C3%A9rs-coefficient-matrix
def _cramers_corrected_stat(chi2, confusion_matrix):
    """calculate Cramers V statistic for categorial-categorial association.
       uses correction from Bergsma and Wicher,
       Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denominator = min((kcorr-1), (rcorr-1))
    if denominator == 0:
        return np.nan
    else:
        return np.sqrt(phi2corr / denominator)


def _get_pivot(field_a, field_b, metadata):
    return metadata[[field_a, field_b]].dropna(how='any').groupby(
        [field_a, field_b]).size().unstack().fillna(0)


def _clear_metadata(metadata,
                    categorials=[], ordinals={}, intervals=[], dates={}, omit=[],
                    err=sys.stderr,
                    for_metadata_correlation=False):
    """Clean up metadata and perform necessary conversions.
    """
    if type(ordinals) != dict:
        raise ValueError(
            '"ordinals" need to be a dictionary! Use None as value for columns'
            ' where you don\'t want to specify a mapping')

    if type(dates) != dict:
        raise ValueError('"dates" need to be a dictionary!')

    # check that columns are distinct
    all_columns = set(categorials) | set(ordinals.keys()) |\
        set(intervals) | set(dates.keys())
    # remove columns of user provided list "omit"
    if len(omit) > 0:
        all_columns -= set(omit)
        if err:
            print("You decided that following %i column%s will be omitted:\n  - %s" % (len(omit), 's' if len(omit) > 1 else '', '\n  - '.join(sorted(omit))), file=err)

    if len(all_columns) < (len(set(categorials)) + len(set(ordinals.keys())) +
                           len(set(intervals)) + len(set(dates.keys())) - len(set(omit))):
        col_usage = pd.Series(categorials + list(ordinals.keys()) +\
                              intervals + list(dates.keys())).value_counts()
        display(col_usage[col_usage > 1])
        raise ValueError(
            'You have repeatedly used metadata column names for ordinals, '
            'categorials or intervals. Make sure a column name only occures in'
            ' one of those three categories!')

    for obj, name in zip([categorials, ordinals, intervals, dates],
                         ['categorials', 'ordinals', 'intervals', 'dates']):
        keys = []
        if type(obj) == list:
            keys = obj
        elif type(obj) == dict:
            keys = obj.keys()
        value_counts = {k: 0 for k in keys}
        for k in keys:
            value_counts[k] += 1
        duplicate_keys = [k for k, v in value_counts.items() if v > 1]
        if len(duplicate_keys) > 0:
            raise ValueError('You specified columns "%s" as %s more than once!'
                             % ('", "'.join(duplicate_keys), name))

    min_col_nr = 2 if for_metadata_correlation else 1
    if len(all_columns) < min_col_nr:
        raise ValueError('You need to specify at least %s columns!' %
                         min_col_nr)

    # check that all columns are actually in metadata
    not_present = []
    for column in categorials + list(ordinals.keys()) + intervals +\
            list(dates.keys()):
        if column not in metadata:
            not_present.append(column)
    if len(not_present) > 0:
        raise ValueError('The column(s) "%s" is/are not in the metadata!' %
                         '", "'.join(sorted(not_present)))

    meta = metadata.copy()

    # check that interval columns can be interpreted as floats and do the
    # conversion
    for column in intervals:
        try:
            meta[column] = meta[column].astype(float)
        except ValueError:
            raise ValueError(('Not all values in column "%s" can be '
                              'interpreted as floats!') % column)

    # convert all date fiels into seconds from epoch to make them interval
    # data. Format string reference at:
    # https://docs.python.org/3/library/datetime.html
    def _convdate(date, frmt):
        if pd.isnull(date):
            return 0.0
        if type(date) == float:
            return date
        elif type(date) == pd.Timestamp:
            return time.mktime(date.timetuple())
        else:
            try:
                return time.mktime(datetime.strptime(date, frmt).timetuple())
            except ValueError:
                try:
                    return datetime.fromisoformat(date).timestamp()
                except ValueError:
                    return np.nan

    for column in dates.keys():
        states_pre_convert = meta[column].dropna().unique()
        meta[column] = meta[column].apply(lambda x: _convdate(x, dates[column]))
        if (meta[column].dropna().unique().shape[0] <= 1) and (len(states_pre_convert) > 1):
            raise ValueError("Applying date format %s to your metadata column %s fails, as now all values are %s" % (dates[column], column, meta[column].dropna().unique()))
            #lambda x: time.mktime(datetime.strptime(
            #    x, dates[column]).timetuple()) if type(x) != float else x)

    # convert ordinal labels into floats
    for column in ordinals.keys():
        if ordinals[column] is None:
            map_ = sorted(meta[column].dropna().unique())
            meta[column] = meta[column].dropna().apply(lambda x: map_.index(x))
        else:
            if type(ordinals[column]) != list:
                raise ValueError(
                    'Mapping for ordinal "%s" must be either None or a list '
                    'but not a "%s"!' % (column, type(ordinals[column])))
            if len(meta[column].dropna().unique()) > len(ordinals[column]):
                err.write('Ordinal "%s" does not specify label(s) "%s".\n' % (
                    column, '", "'.join(sorted(
                        set(meta[column].dropna().unique()) - set(
                            ordinals[column])))))
            meta[column] = meta[column].dropna().apply(
                lambda x: ordinals[column].index(x)
                if x in ordinals[column] else np.nan)

    # omit columns which are either constant or have a different value for every single sample
    omit_constant = []
    omit_alldiff = []
    for column in all_columns:
        if meta[column].dropna().unique().shape[0] <= 1:
            omit_constant.append(column)
        elif meta[column].dropna().unique().shape[0] == meta[column].dropna().shape[0]:
            omit_alldiff.append(column)
    if len(omit_constant) > 0:
        if err:
            print("The following %i column%s will be omitted, as they are CONSTANT for every sample:\n  - %s" % (len(omit_constant), 's' if len(omit_constant) > 1 else '', '\n  - '.join(sorted(omit_constant))), file=err)
    if len(omit_alldiff) > 0:
        if err:
            print("The following %i column%s will be omitted, as they are DIFFERENT for every sample:\n  - %s" % (len(omit_alldiff), 's' if len(omit_alldiff) > 1 else '', '\n  - '.join(sorted(omit_alldiff))), file=err)
    all_columns = [column for column in all_columns
                   if column not in omit_constant
                   if column not in omit_alldiff]

    omit_duplicates = set([])
    for a in tqdm(range(len(all_columns)), 'determin redundant columns', disable=err is None):
        col_a = all_columns[a]
        if col_a in omit_duplicates:
            continue
        for b in range(a + 1, len(all_columns)):
            col_b = all_columns[b]
            if col_b in omit_duplicates:
                continue
            # columns are exact same values
            drop_col = col_a
            if col_b > col_a:
                drop_col = col_b
            if (meta[col_a] == meta[col_b]).all():
            #if meta[[col_a, col_b]].T.drop_duplicates().shape[0] < 2:
                omit_duplicates |= set([drop_col])
            # columns have diff values, but there is a 1:1 mapping
            if meta.fillna('?').groupby([col_a, col_b]).size().shape[0] == meta[col_a].fillna('?').unique().shape[0] == meta[col_b].fillna('?').unique().shape[0]:
                omit_duplicates |= set([drop_col])
    if len(omit_duplicates) > 0:
        if err:
            print("The following %i column%s will be omitted, as they are REDUNDANT to other columns:\n  - %s" % (len(omit_duplicates), 's' if len(omit_duplicates) > 1 else '', '\n  - '.join(sorted(list(omit_duplicates)))), file=err)
    all_columns = [column for column in all_columns
                   if column not in omit_duplicates]

    if len(all_columns) < min_col_nr:
        raise ValueError('After filtering columns, only %i column remains, which is too little for this type of analysis! Need to be %i' %
                         (len(all_columns), min_col_nr))

    return meta, all_columns


def correlate_metadata(metadata,
                       categorials=[], ordinals={}, intervals=[], dates={},
                       err=sys.stderr):
    """Generate correlation Heatmaps for metadata.

    Parameters
    ----------
    metadata : Pandas.DataFrame
        DataFrame holding all metadata about an experiment.
    categorials : [str]
        List of column names of provided metadata DataFrame.
    ordinals : dict{str, None | [str]}
        Two level dictionary, where first key must be metadata column names.
        Value is either None, i.e. no mapping provided, or it must be an
        ordered list of labels. Labels not covered by this list will be
        ignored!
    intervals : [str]
        List of column names of provided metadata DataFrame.
    dates : dict{str: str}
        Dictionary of column names and their format string to convert the date
        into a machine readable datetime object.
    err : StringIO
        Default: sys.stderr
        Stream to print warnings to.

    Returns
    -------
    fig, correlations : pd.DataFrame , tree : skbio.tree.TreeNode
    """
    meta, all_columns = _clear_metadata(
        metadata, categorials=categorials, ordinals=ordinals, intervals=intervals, dates=dates, err=err,
        for_metadata_correlation=True)
    categorials = [c for c in categorials if c in all_columns]
    intervals = [c for c in intervals if c in all_columns]
    ordinals = {c: v for c, v in ordinals.items() if c in all_columns}
    dates = {c: v for c, v in dates.items() if c in all_columns}

    summary = dict()
    # start computing correlations
    # if err is not None:
    #     err.write('correlations intra categorial\n')
    for (column_a, column_b) in itertools.combinations(categorials, 2):
        pivot = _get_pivot(column_a, column_b, meta[all_columns]).values

        if (pivot.shape[0] > 1) & (pivot.shape[1] > 1):
            chi2, p, _, _ = chi2_contingency(pivot)
            v = _cramers_corrected_stat(chi2, pivot)

            res = {'chi2': chi2,
                   'p-value': p,
                   'r_': v}
            summary[(column_a, column_b)] = res
            summary[(column_b, column_a)] = res

    # if err is not None:
    #     err.write('correlations intra ordinal & interval\n')
    set_of_fields = list(ordinals.keys()) + intervals + list(dates.keys())
    for (column_a, column_b) in itertools.combinations(set_of_fields, 2):
        sub = meta[[column_a, column_b]].dropna()
        if (len(sub[column_a].unique()) > 1) and \
           (len(sub[column_b].unique()) > 1):
            spearman_r, spearman_p = spearmanr(sub[column_a], sub[column_b])
            pearson_r, pearson_p = pearsonr(sub[column_a], sub[column_b])
        else:
            spearman_r = np.nan
            pearson_r, pearson_p = np.nan, 1.0

        res = {'stat_': np.absolute(spearman_r),
               'r_': np.absolute(pearson_r),
               'p-value': pearson_p}
        summary[(column_a, column_b)] = res
        summary[(column_b, column_a)] = res

    # if err is not None:
    #     err.write('correlations between categorial and ordinal/interval\n')
    for column_a in categorials:
        for column_b in list(ordinals.keys()) + intervals + list(dates.keys()):
            if (len(meta[column_a].dropna().unique()) <= 1) or \
               (len(meta[column_b].dropna().unique()) <= 1):
                # too few categories
                continue

            groups = [g.dropna().values
                      for n, g
                      in meta.dropna(subset=column_b).groupby(column_a)[column_b]]
            if len(groups) < 2:
                # too little data
                # print(f"Info: skipping correlation due to insufficient data: {column_a} vs {column_b}")
                continue
            if max(map(lambda x: len(set(x)), groups)) == 1:
                # all arrays have only one or a constant value
                continue
            f_, p_ = f_oneway(*groups)

            df_n = len(np.hstack(groups)) - len(groups)
            df_d = len(groups) - 1
            r_ = (f_ / (f_*df_n + df_d))
            res = {'stat': f_,
                   'p-value': p_,
                   'r_': r_}
            summary[(column_a, column_b)] = res
            summary[(column_b, column_a)] = res

    for field in all_columns:
        summary[(field, field)] = {'r_': 1.0}
    correlations = pd.DataFrame.from_dict(
        summary, orient='index')['r_'].unstack().fillna(0.0)

    # create heatmap visualization
    heatmap = sns.clustermap(correlations, cmap='viridis',
                             yticklabels=1, xticklabels=1)
    plt.setp(heatmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    fig = plt.gcf()
    # fig.set_size_inches(15, 15, forward=True)

    # create cluster tree
    dm = DistanceMatrix(1-correlations, ids=correlations.columns)
    tree = TreeNode.from_linkage_matrix(ward(dm.condensed_form()), dm.ids)

    return fig, correlations, tree


CODE_CLAUDE = r"""
# -- Helper: get complete rows for a given set of variables -----------------
get_complete_rows <- function(predictors, data) {
  cols_needed <- c(vars_diversity, predictors)
  complete.cases(data[, cols_needed, drop = FALSE])
}

# -- Helper: count degrees of freedom on the filtered subset ---------------
count_df <- function(predictors, data) {
  rows <- get_complete_rows(predictors, data)
  sub  <- data[rows, , drop = FALSE]
  total <- 0
  for (v in predictors) {
    col <- sub[[v]]
    if (is.factor(col)) {
      col <- droplevels(col)
      total <- total + nlevels(col) - 1
    } else {
      total <- total + 1
    }
  }
  total
}

# -- Helper: safely compute adjusted R² ------------------------------------
get_r2 <- function(predictors, data) {
  rows <- get_complete_rows(predictors, data)
  sub  <- data[rows, , drop = FALSE]
  sub  <- droplevels(sub)

  n_obs   <- sum(rows)
  df_used <- count_df(predictors, data)

  if (n_obs < 10) return(NA)
  if (df_used >= n_obs - 1) return(NA)

  formula_str <- paste("sub[, vars_diversity] ~", paste(predictors, collapse = " + "))

  result <- tryCatch({
    mod <- rda(as.formula(formula_str), data = sub)
    if (is.null(mod$CA) || mod$CA$tot.chi == 0) return(NA)
    RsquareAdj(mod)$adj.r.squared
  }, warning = function(w) {
    if (grepl("overfitted", conditionMessage(w))) return(NA)
    tryCatch({
      mod <- rda(as.formula(formula_str), data = sub)
      RsquareAdj(mod)$adj.r.squared
    }, error = function(e) NA)
  }, error = function(e) NA)

  if (is.null(result) || length(result) != 1) return(NA_real_)
  result
}

# -- Step 1: test each variable individually --------------------------------
cat("=== Step 1: Individual variance explained per variable ===\n")

single_r2 <- sapply(vars_meta, function(v) {
  rows <- get_complete_rows(v, meta_diversity)
  r2   <- get_r2(v, meta_diversity)
  df   <- count_df(v, meta_diversity)
  cat(sprintf("  %-45s | n=%d | df=%d | R²adj=%s\n",
              v, sum(rows), df,
              ifelse(is.na(r2), "NA (overfitted/error)", sprintf("%.4f", r2))))
  r2
})

single_r2_df <- data.frame(
  variable = names(single_r2),
  R2adj    = single_r2,
  n        = sapply(vars_meta, function(v) sum(get_complete_rows(v, meta_diversity))),
  df_used  = sapply(vars_meta, count_df, data = meta_diversity)
)
single_r2_df <- single_r2_df[order(-single_r2_df$R2adj, na.last = TRUE), ]

cat("\nRanking of individual variables:\n")
print(single_r2_df)

# -- Forward selection ------------------------------------------------------
usable_vars <- names(single_r2)[!is.na(single_r2)]
cat(sprintf("\n%d of %d variables are usable\n", length(usable_vars), length(vars_meta)))

selected   <- c()
remaining  <- usable_vars
results    <- list()
current_r2 <- 0

cat("\n=== Forward selection ===\n")

repeat {
  if (length(remaining) == 0) break

  candidate_r2 <- sapply(remaining, function(v) {
    get_r2(c(selected, v), meta_diversity)
  })

  valid <- !is.na(candidate_r2)
  if (!any(valid)) {
    cat("  -> No further valid variables. Selection stopped.\n")
    break
  }

  best_idx <- which.max(candidate_r2)
  best_var <- remaining[best_idx]
  best_r2  <- candidate_r2[best_idx]
  gain     <- best_r2 - current_r2
  n_used   <- sum(get_complete_rows(c(selected, best_var), meta_diversity))

  cat(sprintf("Step %d | %-45s | n=%d | R²adj=%.4f | gain=%.4f\n",
              length(selected) + 1, best_var, n_used, best_r2, gain))

  if (gain <= 1e-6) {
    cat("  -> No further increase. Selection stopped.\n")
    break
  }

  selected   <- c(selected, best_var)
  remaining  <- setdiff(remaining, best_var)
  current_r2 <- best_r2

  results[[length(results) + 1]] <- data.frame(
    step     = length(selected),
    variable = best_var,
    n        = n_used,
    R2adj    = best_r2,
    R2_gain  = gain
  )

  if (length(remaining) == 0) break
}

# -- Output -----------------------------------------------------------------
if (length(results) > 0) {
  result_df <- do.call(rbind, results)
  cat("\n=== Result of forward selection ===\n")
  print(result_df)
  write.table(result_df,
              file = paste0(workdir, '/forward_selection_result_', metric_name, '.tsv'),
              quote = FALSE, sep = '\t', col.names = NA)
  vars_meta <- rownames(result_df)
} else {
  cat("\nNo variable could be selected.\n")
  vars_meta <- NULL
}

write.table(single_r2_df,
            file = paste0(workdir, '/single_variable_r2_', metric_name, '.tsv'),
            quote = FALSE, sep = '\t', col.names = NA)

cols_needed   <- c(vars_diversity, vars_meta)
complete_rows <- complete.cases(meta_diversity[, cols_needed])
meta_diversity      <- droplevels(meta_diversity[complete_rows, ])

cat(sprintf("\nn = %d samples for ordiR2step\n", nrow(meta_diversity)))

"""
def _read_single_variable_r2(workdir, metric_name):
    fp_singlevar = '%s/single_variable_r2_%s.tsv' % (workdir, metric_name)
    if os.path.exists(fp_singlevar):
        df = pd.read_csv(fp_singlevar, sep="\t", index_col=0)
        del df['variable']
        df = df.rename(columns={'n': 'number_samples',
                                'df_used': 'number_states'})
        df.index.name = 'covariate'
        return df
    else:
        return None

def _read_forward_selection(workdir, metric_name):
    fp_fwd_sel = '%s/forward_selection_result_%s.tsv' % (workdir, metric_name)
    if os.path.exists(fp_fwd_sel):
        df = pd.read_csv(fp_fwd_sel, sep="\t", index_col=0)
        del df['variable']
        df = df.rename(columns={'n': 'number_samples'})
        df.index.name = 'covariate'
        return df
    else:
        return None

def redundancy_analysis_alpha(metadata, alpha, auto=False,
                              categorials=[], ordinals={}, intervals=[],
                              dates={}, title=None, colors=None, ax=None,
                              **executor_args):
    """Perform a forward step redundancy analysis rearding alpha diversity.

    Parameters
    ----------
    metadata : Pandas.DataFrame
        DataFrame holding all metadata about an experiment.
    alpha : Pandas.Series
        Series holding alpha diversity values for every sample.
    auto : Boolean
        Default: False
        If True, we take all metadata columns and iteratively try to compose
        a model that explains most of the variability.
    categorials : [str]
        List of column names of provided metadata DataFrame.
    ordinals : dict{str, None | [str]}
        Two level dictionary, where first key must be metadata column names.
        Value is either None, i.e. no mapping provided, or it must be an
        ordered list of labels. Labels not covered by this list will be
        ignored!
    dates : dict{str: str}
        Dictionary of column names and their format string to convert the date
        into a machine readable datetime object.
    intervals : [str]
        List of column names of provided metadata DataFrame.
    title : str
        Additional string that will be printed into figure's title
    ax : plt.axes
        The axis to plot on.
    colors : dict(str -> str)
        A dictionary, defining the color for every metadata column.
    err : StringIO
        Default: sys.stderr
        Stream to print warnings to.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Notes
    -----
    Following Serenes approach.
    """
    def pre_execute(workdir, args):
        COL_NAME_ALPHA = 'forRDA_alpha'

        # if user choose "auto", we make all metadata columns to "categorials" if not otherise
        # classified as ordinals, intervals or dates
        if (auto is True) and (len(args['categorials']) <= 0):
            args['categorials'] = list(set(args['metadata'].columns) - set(args['ordinals'].keys()) - set(args['intervals']) - set(args['dates'].keys()))

        # samples = set(args['metadata'].index) & set(args['alpha'].index)
        meta, all_columns = _clear_metadata(
            args['metadata'], args['categorials'], args['ordinals'],
            args['intervals'], args['dates'], for_metadata_correlation=False)
        _alpha = args['alpha'].copy()
        _alpha.name = COL_NAME_ALPHA
        meta_alpha = meta.loc[:, list(all_columns)].merge(
            _alpha.to_frame(), left_index=True, right_index=True)
        if args['metadata'].shape[0] != args['alpha'].shape[0]:
            sys.stderr.write(
                'You provided %s and %s samples in metadata and alpha '
                'respectively. Merging to %s samples for further analysis.\n'
                % (args['metadata'].shape[0], args['alpha'].shape[0],
                   meta_alpha.shape[0]))

        meta_alpha.to_csv('%s/metadata_alpha_%s.tsv' % (workdir, args['alpha'].name), sep="\t",
                          index=False)
        with open('%s/rscript_%s.R' % (workdir, args['alpha'].name), 'w') as f:
            f.write('library(vegan)\n')
            f.write("workdir <- '%s'\n" % workdir)
            f.write("metric_name <- '%s'\n" % args['alpha'].name)
            f.write('meta_diversity = read.csv(paste0(workdir, \'/metadata_alpha_\', metric_name, \'.tsv\'), '
                    'stringsAsFactors=FALSE, sep=\'\\t\')\n')
            f.write('vars_meta = c(\'%s\')\n' %
                    "', '".join([c for c in sorted(args['categorials'] + list(args['ordinals'].keys())) if c in all_columns]))
            f.write('meta_diversity[vars_meta] = lapply(meta_diversity[vars_meta],'
                    ' factor)\n')
            f.write('vars_diversity = c(\'%s\')\n' % COL_NAME_ALPHA)
            if args['auto'] is True:
                f.write(CODE_CLAUDE)
                f.write('meta_diversity <- na.omit(meta_diversity[, cols_needed])\n')
            else:
                f.write('meta_diversity = meta_diversity[complete.cases(meta_diversity), ]\n')

            f.write('mod0 <- rda(meta_diversity$%s ~ 1., meta_diversity)  # Model with intercept only\n' % COL_NAME_ALPHA)
            f.write('mod1 <- rda(meta_diversity$%s ~ ., meta_diversity)  # Model with all explanatory variables\n' % COL_NAME_ALPHA)
            f.write('step.res <- ordiR2step(mod0, mod1, perm.max = 1000)\n')
            f.write('write.table(step.res$anova, file=paste0(workdir, \'/result_\', metric_name, \'.tsv\'), '
                    'quote=FALSE, sep=\'\\t\', col.names = NA)\n')

    def commands(workdir, ppn, args):
        commands = []

        if (settings.GRIDNAME != 'JLU') and ('use_grid' in executor_args) and (executor_args['use_grid'] is True):
            commands.append('module load %s' % settings.R_MODULE)
        commands.append('R --vanilla < %s/rscript_%s.R > %s/rscript_%s.out 2> %s/rscript_%s.err' % (workdir, args['alpha'].name, workdir, args['alpha'].name, workdir, args['alpha'].name))

        return commands

    def post_execute(workdir, args):
        try:
            rda = pd.read_csv('%s/result_%s.tsv' % (workdir, args['alpha'].name), sep='\t', index_col=0)
        except EmptyDataError:
            return {'table': pd.DataFrame()}

        # drop fields not starting with a +, i.e. are <All variables> or <none>
        rda = rda.loc[[idx for idx in rda.index if idx.startswith('+')], :]

        # compute adjusted effect size
        rda['effect size'] = rda['R2.adj'] - ([0] + list(
            rda['R2.adj'].values)[:-1])

        rda.index = map(lambda x: x.replace('+ ', ''), rda.index)

        rda = rda.reset_index().rename(columns={'index': 'covariate'})

        return {'table': rda,
                'single_variable_r2': _read_single_variable_r2(workdir, args['alpha'].name),
                'forward_selection': _read_forward_selection(workdir, args['alpha'].name)}

    def post_cache(cache_results):
        if cache_results['results']['table'].shape[0] > 0:
            if ax is None:
                fig, axes = plt.subplots(1, 1)
            else:
                axes = ax
            rda = cache_results['results']['table']
            rda['label'] = rda['covariate'] + '\n' + rda['Pr(>F)'].apply(
                lambda x: '(p: %.3f)' % x)
            if colors is not None:
                palette = {row['label']: colors[row['covariate']]
                           for idx, row
                           in rda.iterrows()}
            else:
                palette = None
            sns.barplot(data=rda.reset_index(),
                        x='effect size',
                        y='label',
                        order=rda.sort_values(
                            'effect size', ascending=False)['label'],
                        ax=axes,
                        palette=palette)
            ttl = 'Redundancy analysis "%s"' % alpha.name
            if title is not None:
                ttl = '%s\n%s' % (ttl, title)
            axes.set_title(ttl)
            axes.set_ylabel('covariate')
            cache_results['results']['figure'] = axes
        else:
            sys.stderr.write('No significant findings.\n')
        return cache_results

    return _executor('fRDAalpha',
                     {'metadata': metadata,
                      'alpha': alpha,
                      'categorials': categorials,
                      'ordinals': ordinals,
                      'intervals': intervals,
                      'dates': dates,
                      'auto': auto},
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache,
                     ppn=1,
                     environment=settings.QIIME2_ENV,
                     **executor_args)


def redundancy_analysis_beta(metadata, beta, metric_name, auto=False,
                             categorials=[], ordinals={}, intervals=[],
                             dates={}, num_dimensions=10, title=None, seed=None,
                             colors=None, ax=None,
                             **executor_args):
    """Perform a forward step redundancy analysis rearding alpha diversity.

    Parameters
    ----------
    metadata : Pandas.DataFrame
        DataFrame holding all metadata about an experiment.
    beta : skbio.DistanceMatrix | skbio.OrdinationResults
        Series holding alpha diversity values for every sample.
    metric_name : str
        Beta diversity metric name; only for printing a speaking label.
    auto : Boolean
        Default: False
        If True, we take all metadata columns and iteratively try to compose
        a model that explains most of the variability.
    categorials : [str]
        List of column names of provided metadata DataFrame.
    ordinals : dict{str, None | [str]}
        Two level dictionary, where first key must be metadata column names.
        Value is either None, i.e. no mapping provided, or it must be an
        ordered list of labels. Labels not covered by this list will be
        ignored!
    intervals : [str]
        List of column names of provided metadata DataFrame.
    dates : dict{str: str}
        Dictionary of column names and their format string to convert the date
        into a machine readable datetime object.
    num_dimensions : int
        Default 10.
        Number of PCoA dimensions to consider.
    title : str
        Additional string that will be printed into figure's title
    seed : int or None
        As the R function ordiR2step relies on permutations, results won't be
        stable unless to define a distinct seed.
    colors : dict(str -> str)
        A dictionary, defining the color for every metadata column.
    ax : plt.axes
        The axis to plot on.
    err : StringIO
        Default: sys.stderr
        Stream to print warnings to.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Notes
    -----
    Following Serenes approach.
    """
    def pre_execute(workdir, args):
        COL_NAME_BETA = 'forRDA_beta'

        # if user choose "auto", we make all metadata columns to "categorials" if not otherise
        # classified as ordinals, intervals or dates
        if (auto is True) and (len(args['categorials']) <= 0):
            args['categorials'] = list(set(args['metadata'].columns) - set(args['ordinals'].keys()) - set(args['intervals']) - set(args['dates'].keys()))

        # samples = set(args['metadata'].index) & set(args['alpha'].index)
        meta, all_columns = _clear_metadata(
            args['metadata'], args['categorials'], args['ordinals'],
            args['intervals'], args['dates'], for_metadata_correlation=False)

        if type(args['beta']) == OrdinationResults:
            dimred = args['beta'].samples
        else:
            idx_shared = set(args['beta'].ids) & set(meta.index)
            if len(idx_shared) <= 0:
                raise ValueError("There is no overlap between samples in your metadata and those in your beta diversity DistanceMatrix!")
            dimred = pcoa(args['beta'].filter(idx_shared)).samples
        dimred = dimred.iloc[:, :num_dimensions]
        dimred.columns = ['%s_%s' % (COL_NAME_BETA, c) for c in dimred.columns]

        meta_diversity = meta.loc[:, list(all_columns)].merge(
            dimred, left_index=True, right_index=True)
        if args['metadata'].shape[0] != dimred.shape[0]:
            sys.stderr.write(
                'You provided %s and %s samples in metadata and beta '
                'respectively. Merging to %s samples for further analysis.\n'
                % (args['metadata'].shape[0], dimred.shape[0],
                   meta_diversity.shape[0]))

        meta_diversity.to_csv('%s/metadata_beta_%s.tsv' % (workdir, args['metric_name']), sep="\t",
                               index=False)

        with open('%s/rscript_%s.R' % (workdir, args['metric_name']), 'w') as f:
            f.write('library(vegan)\n')
            f.write("workdir <- '%s'\n" % workdir)
            f.write("metric_name <- '%s'\n" % args['metric_name'])
            f.write('meta_diversity = read.csv(paste0(workdir, \'/metadata_beta_\', metric_name, \'.tsv\'), '
                    'stringsAsFactors=FALSE, sep=\'\\t\')\n')
            f.write('vars_cat = c(\'%s\')\n' %
                    "', '".join([c for c in sorted(args['categorials'] + list(args['ordinals'].keys())) if c in all_columns]))
            f.write('vars_diversity = c(\'%s\')\n' %
                    "', '".join([c for c in dimred.columns]))
            f.write('vars_meta = c(\'%s\')\n' %
                    "', '".join(sorted(all_columns)))
            f.write('meta_diversity[vars_cat] = lapply(meta_diversity[vars_cat],'
                 ' factor)\n')
            if args['auto'] is True:
                f.write(CODE_CLAUDE)
            else:
                f.write('meta_diversity = meta_diversity[complete.cases(meta_diversity), ]\n')

            f.write('mod0 <- rda(meta_diversity[, vars_diversity] ~ 1., meta_diversity[, '
                    'vars_meta])  # Model with intercept only\n')
            f.write('mod1 <- rda(meta_diversity[, vars_diversity] ~ ., meta_diversity[, '
                    'vars_meta])  # Model with all explanatory variables\n')
            if args['seed'] is not None:
                f.write('set.seed(%i)\n' % args['seed'])
            f.write('step.res <- ordiR2step(mod0, mod1, perm.max = 1000)\n')
            f.write('write.table(step.res$anova, file=paste0(workdir, \'/result_\', metric_name, \'.tsv\'), '
                    'quote=FALSE, sep=\'\\t\', col.names = NA)\n')

    def commands(workdir, ppn, args):
        commands = []

        if (settings.GRIDNAME != 'JLU_SLURM') and ('use_grid' in executor_args) and (executor_args['use_grid'] is True):
            commands.append('module load %s' % settings.R_MODULE)
        commands.append('R --vanilla < %s/rscript_%s.R > %s/rscript_%s.out 2> %s/rscript_%s.err' % (workdir, args['metric_name'], workdir, args['metric_name'], workdir, args['metric_name']))

        return commands

    def post_execute(workdir, args):
        try:
            rda = pd.read_csv('%s/result_%s.tsv' % (workdir, args['metric_name']), sep='\t', index_col=0)
        except EmptyDataError:
            sys.stderr.write('No significant covariates found!\n')
            return {'table': pd.DataFrame()}

        # drop fields not starting with a +, i.e. are <All variables> or <none>
        rda = rda.loc[[idx for idx in rda.index if idx.startswith('+')], :]

        # compute adjusted effect size
        rda['effect size'] = rda['R2.adj'] - ([0] + list(
            rda['R2.adj'].values)[:-1])

        rda.index = map(lambda x: x.replace('+ ', ''), rda.index)

        rda = rda.reset_index().rename(columns={'index': 'covariate'})

        return {'table': rda, 'seed': args['seed'],
                'single_variable_r2': _read_single_variable_r2(workdir, args['metric_name']),
                'forward_selection': _read_forward_selection(workdir, args['metric_name'])
               }

    def post_cache(cache_results):
        if cache_results['results']['table'].shape[0] > 0:
            if ax is None:
                fig, axes = plt.subplots(1, 1)
            else:
                axes = ax
            rda = cache_results['results']['table']
            rda['label'] = rda['covariate'] + '\n' + rda['Pr(>F)'].apply(
                lambda x: '(p: %.3f)' % x)
            if colors is not None:
                palette = {row['label']: colors[row['covariate']]
                           for idx, row
                           in rda.iterrows()}
            else:
                palette = None
            sns.barplot(data=rda.reset_index(),
                        x='effect size',
                        y='label',
                        order=rda.sort_values(
                            'effect size', ascending=False)['label'],
                        ax=axes,
                        palette=palette)
            axes.set_ylabel('covariate')
            ttl = 'Redundancy analysis "%s"' % metric_name
            if title is not None:
                ttl = '%s\n%s' % (ttl, title)
            axes.set_title(ttl)
            if ax is None:
                cache_results['results']['figure'] = fig
            else:
                cache_results['results']['figure'] = axes
        else:
            sys.stderr.write('No significant findings.\n')
        return cache_results

    return _executor('fRDAbeta',
                     {'metadata': metadata,
                      'beta': beta,
                      'metric_name': metric_name,
                      'categorials': categorials,
                      'ordinals': ordinals,
                      'intervals': intervals,
                      'dates': dates,
                      'seed': seed,
                      'auto': auto},
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache,
                     ppn=1,
                     environment=settings.QIIME2_ENV,
                     **executor_args)

def _generateRcode_redundancy(div_type, diversity_columns:[str], metadata_columns:[str], metric_name, workdir, auto, seed=None):
    assert div_type in ['alpha', 'beta']

    code = []
    code.append('library(vegan)')
    code.append("workdir <- '%s'" % workdir)
    code.append("metric_name <- '%s'" % metric_name)
    code.append('meta = read.csv(paste0(workdir, \'/metadata.tsv\'), stringsAsFactors=FALSE, sep=\'\\t\', row.names = 1)')
    code.append('vars_meta = colnames(meta)')
    code.append('diversity = read.csv(paste0(workdir, \'/diversity_\', metric_name, \'.tsv\'), stringsAsFactors=FALSE, sep=\'\\t\', row.names = 1)')
    code.append('vars_diversity = colnames(diversity)')
    code.append('meta_diversity <- merge(meta, diversity, by = "row.names")')
    code.append('meta_diversity[vars_meta] = lapply(meta_diversity[vars_meta], factor)')

    if auto is True:
        code.append(CODE_CLAUDE)
        if div_type == 'alpha':
            code.append('meta_diversity <- na.omit(meta_diversity[, cols_needed])')
    else:
        code.append('meta_diversity = meta_diversity[complete.cases(meta_diversity), ]')

    code.append('if (length(vars_meta) > 1) {')
    code.append('    mod0 <- rda(meta_diversity[, vars_diversity] ~ 1., meta_diversity[, vars_meta])  # Model with intercept only')
    code.append('    mod1 <- rda(meta_diversity[, vars_diversity] ~  ., meta_diversity[, vars_meta])  # Model with all explanatory variables')
    if seed is not None:
        code.append('    set.seed(%i)' % seed)
    code.append('    step.res <- ordiR2step(mod0, mod1, perm.max = 1000)')
    code.append('    write.table(step.res$anova, file=paste0(workdir, \'/result_\', metric_name, \'.tsv\'), quote=FALSE, sep=\'\\t\', col.names = NA)')
    code.append('}')

    return '\n'.join(code)

def redundancy(metadata: pd.DataFrame, alpha: pd.DataFrame, beta: dict[str: DistanceMatrix],
               auto:bool=True,
               categorials:[str]=[],
               ordinals:dict[str: [str]]={}, intervals:str=[], dates:dict[str: [str]]={}, omit:[str]=[],
               beta_axis:int=10, seed:int=None, palette:dict[str: str]=None,
               **executor_args):
    """Forward step redundancy analysis for Alpha- and Beta-Diversity.

    Parameters
    ----------
    beta : dict(str: skbio.DistanceMatrix | skbio.OrdinationResults)
        Dictionary of "metric_name": object, where object is
        Beta diversity skbio.DistanceMatrix OR (if already pre-computed) OrdinationResults
    auto : Bool
        Default: True.
        Uses a greedy heuristic to first determine column with most variance explained
        and then iteratively addes columsn until total variance converges.
    categorials : [str]
        If auto is False, use these metadata columns as categorial variables.
    ordinals : dict[str: [str]]
        Metadata columns that shall be explicitely treated as ordinales, i.e.
        you need to provide a total ordering.
    intervals : [str]
        Metadata columns that shall explicitely be treated as interval data, like age, bmi, ...
    dates : dict[str: [str]]
        Metadata columns that shall explicitely be converted as dates. You need
        to provide the date format string like "%Y-%m-%d"
    omit : [str]
        List of metadata columns to be omitted in analysis.
    beta_axis : int
        Number of PCoA axis that shall be used.
    seed : int
        Random seed.
    palette : dict[str: str]
        A palette for coloring the resulting graphs.
    """
    metrics = []
    if beta is not None:
        for bmetric in beta.keys():
            metrics.append([bmetric, 'beta', isinstance(beta[bmetric], DistanceMatrix)])
    if alpha is not None:
        for ametric in alpha.columns:
            metrics.append([ametric, 'alpha', False])
    verbose = executor_args.get('verbose', sys.stderr)

    def generate_code_python_pcoa():
        import sys
        from skbio.stats.distance import DistanceMatrix
        from skbio.stats.ordination import pcoa
        workdir, bmetric, beta_axis = sys.argv[1:]
        res = pcoa(DistanceMatrix.read('%s/distancematrix_%s.tsv' % (workdir, bmetric)))
        res.samples.iloc[:, :int(beta_axis)].to_csv('%s/diversity_%s.tsv' % (workdir, bmetric), sep='\t', index_label='sample_name')

    def pre_execute(workdir, args):
        # if user choose "auto", we make all metadata columns to "categorials" if not otherise
        # classified as ordinals, intervals or dates
        if (auto is True) and (len(args['categorials']) <= 0):
            args['categorials'] = list(set(args['metadata'].columns) - set(args['ordinals'].keys()) - set(args['intervals']) - set(args['dates'].keys()))

        if args['alpha'] is not None:
            metricsnames_in_metadata = (set(args['metadata'].columns) - set(args['omit'])) & set(args['alpha'].columns)
            if len(metricsnames_in_metadata) > 0:
                raise ValueError("Your metadata table contains the following %i columns, whose names collidate with your alpha diversity data!\n  - %s\n" % (len(metricsnames_in_metadata), '\n  - '.join(metricsnames_in_metadata)))

        meta, all_columns = _clear_metadata(
            args['metadata'], args['categorials'], args['ordinals'],
            args['intervals'], args['dates'], args['omit'], for_metadata_correlation=False, err=verbose)

        # sync samples in metadata, alpha- and beta-diversity
        idx = {'meta': set(meta.index)}
        if args['alpha'] is not None:
            for ametric in args['alpha'].columns:
                idx[ametric] = set(args['alpha'][ametric].dropna().index)
        for bmetric in args['beta'].keys():
            if isinstance(args['beta'][bmetric], OrdinationResults):
                idx[bmetric] = set(args['beta'][bmetric].samples.index)
                if args['beta'][bmetric].samples.shape[1] < args['beta_axis']:
                    raise ValueError("Your provided ordination for metric '%s' has fewer dimensions than you requested with parameter beta_axis=%s" % (bmetric, args['beta_axis']))
            elif isinstance(args['beta'][bmetric], DistanceMatrix):
                idx[bmetric] = set(args['beta'][bmetric].ids)
            else:
                raise ValueError("Unknown data type for beta diversity metric '%s'" % bmetric)

        idx_shared = idx['meta']
        for metric in idx.keys():
            if metric == 'meta':
                continue
            idx_shared &= idx[metric]
        idx_shared = sorted(list(idx_shared))
        if len(idx_shared) <= 0:
            raise ValueError("There is no overlap between samples in your metadata and those in your alpha- and beta- diversity inputs!")
        num_idx_diversities = min([len(v) for k, v in idx.items() if k != 'meta'])
        if len(idx['meta']) != num_idx_diversities:
            if verbose:
                verbose.write(
                    'You provided %s and %s samples in metadata and diversity objects, '
                    'respectively. Merging to %s samples for further analysis.\n'
                    % (args['metadata'].shape[0], num_idx_diversities, len(idx_shared)))

        # write metadata to tmp file
        args['metadata'].loc[idx_shared, all_columns].to_csv('%s/metadata.tsv' % workdir, sep="\t", index_label='sample_name')

        # prepare R input sheets and R scripts
        if beta is not None:
            for bmetric in args['beta'].keys():
                if isinstance(args['beta'][bmetric], OrdinationResults):
                    args['beta'][bmetric].samples.rename(columns={c: '%s_%s' % (bmetric, c) for c in args['beta'][bmetric].samples.columns}).loc[idx_shared, :].iloc[:, :args['beta_axis']].to_csv('%s/diversity_%s.tsv' % (workdir, bmetric), sep="\t", index_label='sample_name')
                elif isinstance(args['beta'][bmetric], DistanceMatrix):
                    # pcoa will be computed as a step in the cluster script
                    args['beta'][bmetric].filter(idx_shared).write('%s/distancematrix_%s.tsv' % (workdir, bmetric))
                    with open('%s/pcoa_script_%s.py' % (workdir, bmetric), "w", encoding="utf-8") as f:
                        code = textwrap.dedent(inspect.getsource(generate_code_python_pcoa))
                        code += '\nif __name__ == "__main__":\n    generate_code_python_pcoa()\n'
                        f.write(code + '\n')

                with open('%s/rscript_%s.R' % (workdir, bmetric), 'w') as f:
                    f.write(_generateRcode_redundancy(
                        'beta',
                        [], #['%s_%s' % (bmetric, c) for c in dimred.columns],
                        [c for c in sorted(args['categorials'] + list(args['ordinals'].keys())) if c in all_columns],
                        bmetric,
                        workdir, args['auto'], seed=args['seed']))

        if alpha is not None:
            for ametric in args['alpha'].columns:
                meta_alpha = meta.loc[:, list(all_columns)].merge(
                    args['alpha'][ametric], left_index=True, right_index=True)

                args['alpha'].loc[idx_shared, ametric].to_frame().to_csv('%s/diversity_%s.tsv' % (workdir, ametric), sep="\t", index_label='sample_name')

                with open('%s/rscript_%s.R' % (workdir, ametric), 'w') as f:
                    f.write(_generateRcode_redundancy(
                        'alpha',
                        [], #[ametric],
                        [c for c in sorted(args['categorials'] + list(args['ordinals'].keys())) if c in all_columns],
                        ametric,
                        workdir, args['auto'], seed=args['seed']))

        with open('%s/metrics.txt' % workdir, 'w') as f:
            f.write('\n'.join(map(lambda x: '\t'.join(map(str, x)), metrics)))

    def commands(workdir, ppn, args):
        commands = {'pre': [], 'main': [], 'post': []}

        commands['main'].append('var_info=`head -n ${%s} %s/metrics.txt | tail -n 1`' % (
            settings.VARNAME_PBSARRAY, workdir))
        commands['main'].append('var_metric=`echo "${var_info}" | cut -f 1`')
        commands['main'].append('var_pcoa=`echo "${var_info}" | cut -f 3`')
        if (settings.GRIDNAME != 'JLU_SLURM') and ('use_grid' in executor_args) and (executor_args['use_grid'] is True):
            commands['main'].append('module load %s' % settings.R_MODULE)
        commands['main'].append('if [ "${var_pcoa}" == "True" ]; then python %s/pcoa_script_${var_metric}.py "%s" "${var_metric}" "%s"; fi' % (workdir, workdir, str(args['beta_axis'])))
        commands['main'].append('R --vanilla < %s/rscript_${var_metric}.R > %s/rscript_${var_metric}.out 2> %s/rscript_${var_metric}.err' % (
            workdir, workdir, workdir))

        return commands

    def post_execute(workdir, args):
        results = {'table': dict(), 'single_variable_r2': dict(), 'forward_selection': dict(),
                   'seed': args['seed']}

        for (metric, _, _) in metrics:
            try:
                rda = pd.read_csv('%s/result_%s.tsv' % (workdir, metric), sep='\t', index_col=0)

                # drop fields not starting with a +, i.e. are <All variables> or <none>
                rda = rda.loc[[idx for idx in rda.index if idx.startswith('+')], :]

                # compute adjusted effect size
                rda['effect size'] = rda['R2.adj'] - ([0] + list(rda['R2.adj'].values)[:-1])
                rda.index = map(lambda x: x.replace('+ ', ''), rda.index)
                rda = rda.reset_index().rename(columns={'index': 'covariate'})

                results['table'][metric] = rda
            except (EmptyDataError, FileNotFoundError):
                #sys.stderr.write('No significant covariates found for %s!\n' % metric)
                results['table'][metric] = pd.DataFrame()

            results['single_variable_r2'][metric] = _read_single_variable_r2(workdir, metric)
            results['forward_selection'][metric] = _read_forward_selection(workdir, metric)

        return results

    def post_cache(cache_results, palette=dict(), title=None):
        cols = (1 if alpha is not None else 0) + (1 if beta is not None else 0)
        rows = max(alpha.shape[1] if alpha is not None else 0,
                   len(beta.keys()) if beta is not None else 0)

        max_num_covariates = max(
            [cache_results['results']['table'][metric].shape[0]
             for metric in cache_results['results']['table'].keys()])

        fig, axes = plt.subplots(rows, cols,
                                 gridspec_kw={"wspace": 0.8, "hspace": 0.5},
                                 figsize=(5 * cols, 1. * max_num_covariates * rows))

        axmetrics = []
        if cols == 1:
            if rows == 1:
                axes = np.array([axes])
            if alpha is not None:
                axmetrics.extend(list(zip(axes, alpha.columns)))
            if beta is not None:
                axmetrics.extend(list(zip(axes, beta.keys())))
        elif cols == 2:
            axmetrics.extend(list(zip(axes[:, 0], alpha.columns)))
            axmetrics.extend(list(zip(axes[:, 1], beta.keys())))

        # collect axes which are used to later clear those without data,
        # happens if number of alpha and beta metrics differ
        no_clear = []
        for (ax, metric) in axmetrics:
            if cache_results['results']['table'][metric].shape[0] > 0:
                rda = cache_results['results']['table'][metric]
                rda['label'] = rda['covariate'] + '\n' + rda['Pr(>F)'].apply(lambda x: '(p: %.3f)' % x)
                rda['color'] = rda['covariate'].apply(lambda x: palette.get(x, sns.color_palette()[0]))
                pltte = rda.set_index('label')['color'].to_dict()

                sns.barplot(data=rda.reset_index(),
                           x='effect size',
                           hue='label',
                           y='label',
                           order=rda.sort_values('effect size', ascending=False)['label'],
                           ax=ax,
                           palette=pltte, legend=False,
                           )
                #ax.set_ylabel('covariate')
                ax.set_ylabel("")
                ax.set_title(metric)
                no_clear.append(ax)
                ax.set_ylim((ax.get_ylim()[-1] + max_num_covariates, ax.get_ylim()[-1]))
            else:
                ax.text(0.5, 0.5, 'No significant findings\nfor %s' % metric, ha='center')
                #sys.stderr.write('No significant findings for %s.\n' % metric)

        # remove unused axes
        for ax in axes.flatten():
            if ax not in no_clear:
                ax.set_axis_off()

        if title is not None:
            fig.suptitle(title)

        cache_results['results']['figure'] = fig

        return cache_results

    return _executor('redundancy',
                     {'metadata': metadata,
                      'alpha': alpha,
                      'beta': beta,
                      'auto': auto,
                      'categorials': categorials,
                      'ordinals': ordinals,
                      'intervals': intervals,
                      'dates': dates,
                      'omit': omit,
                      'beta_axis': beta_axis,
                      'seed': seed,
                     },
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache,
                     ppn=1,
                     environment=settings.QIIME2_ENV,
                     array=len(metrics),
                     **executor_args)

def adonis(metadata: pd.DataFrame, dm: DistanceMatrix,
           formula: str, strat: str=None, permutations: int=999, ppn=1,
           **executor_args):
    """Performs multiway adonis on beta diversity.
       See: http://cc.oulu.fi/~jarioksa/softhelp/vegan/html/adonis.html

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata.
    dm : DistanceMatrix
        A beta diversity distance matrix, e.g. unweighted unifrac.
    formula : str
        A typical model formula such as Y ~ A + B*C, but Y will be the provided
        distance matrix and should NOT be added.
        A, B, and C may be factors or continuous variables, must be columns
        in metadata table.
    strat : str
        Default: None.
        groups (strata) within which to constrain permutations.
        Must be a column in provided metadata table.
    permutations : int
        Default: 999
        number of replicate permutations used for the hypothesis tests (F tests).
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    ?
    """
    def pre_execute(workdir, args):
        if ('strat' in args) and (args['strat'] is not None) and (args['strat'] not in args['metadata'].columns):
            raise ValueError("Column '%s' for strat cannot be found in metadata." % args['strat'])
        if '~' in args['formula']:
            raise ValueError('Please omit the "Y~" part in the formula as it will be automatically added.')
        for factor in re.split(r'\W+', args['formula']):
            if factor not in args['metadata'].columns:
                raise ValueError("Column '%s' of formula '%s' cannot be found in metadata." % (args['formula'], factor))

        idx_samples = set(args['metadata'].index) & set(args['dm'].ids)
        if len(set(args['metadata'].index)) != len(set(args['dm'].ids)):
            sys.stderr.write(
                'You provided %s and %s samples in metadata and beta dm'
                'respectively. Merging to %s samples for further analysis.\n'
                % (args['metadata'].shape[0], len(set(args['dm'].ids)),
                   len(idx_samples)))
        if len(idx_samples) < 2:
            raise ValueError('You must provide at least 2 samples!')

        args['metadata'].loc[list(idx_samples), :].to_csv('%s/metadata.tsv' % workdir, sep="\t",
                                index=True)
        args['dm'].filter(idx_samples, strict=False).write('%s/beta.tsv' % workdir)

        with open('%s/rscript.R' % workdir, 'w') as f:
            f.write('library(vegan)\n')
            f.write('dm <- as.dist(as(read.table("%s/beta.tsv", sep="\\t", header=TRUE, row.names=1), "matrix"))\n' % workdir)
            f.write('meta = read.csv("%s/metadata.tsv", sep="\\t", row.names=1, header=TRUE)\n' % workdir)
            flag_strat = ''
            if args['strat'] is not None:
                flag_strat = ' strat=meta$%s,' % args['strat']
            flag_parallel = ''
            if 'ppn' in executor_args:
                if executor_args['ppn'] > 1:
                    flag_parallel = 'parallel=%i' % executor_args['ppn']
            f.write('res <- adonis2(dm ~ %s, data=meta, %s permutations=%i %s)\n' % (args['formula'], flag_strat, args['permutations'], flag_parallel))
            f.write('write.table(res, "%s/result.tsv", sep="\\t", append=F, quote=FALSE)\n' % workdir)
        if 'dry' in executor_args:
            if executor_args['dry']:
                print('R SCRIPT:\n----------------')
                with open('%s/rscript.R' % workdir, 'r') as f:
                    print(''.join(f.readlines()))
                print('----------------')

    def commands(workdir, ppn, args):
        commands = []

        commands.append('R --vanilla < %s/rscript.R' % workdir)

        return commands

    def post_execute(workdir, args):
        try:
            rda = pd.read_csv('%s/result.tsv' % workdir, sep='\t', index_col=0)
        except EmptyDataError:
            sys.stderr.write('No significant covariates found!\n')
            return {'table': pd.DataFrame()}

        # # drop fields not starting with a +, i.e. are <All variables> or <none>
        # rda = rda.loc[[idx for idx in rda.index if idx.startswith('+')], :]
        #
        # # compute adjusted effect size
        # rda['effect size'] = rda['R2.adj'] - ([0] + list(
        #     rda['R2.adj'].values)[:-1])
        #
        # rda.index = map(lambda x: x.replace('+ ', ''), rda.index)
        #
        # rda = rda.reset_index().rename(columns={'index': 'covariate'})

        return {'table': rda}

    return _executor('adonis',
                     {'metadata': metadata,
                      'dm': dm,
                      'formula': formula,
                      'strat': strat,
                      'permutations': permutations},
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=ppn,
                     environment=settings.QIIME2_ENV,
                     **executor_args)
