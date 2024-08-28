import sys
import time
from datetime import datetime
import itertools
import re

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

from ggmap.analyses import _executor
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
                    categorials=[], ordinals={}, intervals=[], dates={},
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
    if len(all_columns) < (len(set(categorials)) + len(set(ordinals.keys())) +
                           len(set(intervals)) + len(set(dates.keys()))):
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
            return 0
        if type(date) == float:
            return date
        elif type(date) == pd.Timestamp:
            return time.mktime(date.timetuple())
        else:
            try:
                return time.mktime(datetime.strptime(date, frmt).timetuple())
            except ValueError:
                return pd.NaT
    for column in dates.keys():
        meta[column] = meta[column].apply(lambda x: _convdate(x, dates[column]))
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
        metadata, categorials, ordinals, intervals, dates, err,
        for_metadata_correlation=True)
    summary = dict()
    # start computing correlations
    # if err is not None:
    #     err.write('correlations intra categorial\n')
    for (column_a, column_b) in itertools.combinations(categorials, 2):
        pivot = _get_pivot(column_a, column_b, meta).values

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
                      in meta.groupby(column_a)[column_b]]

            if any(len(group) <= 1 for group in groups):
                print(f"Info: skipping correlation due to insufficient data: {column_a} vs {column_b}")
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


def redundancy_analysis_alpha(metadata, alpha,
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

        # samples = set(args['metadata'].index) & set(args['alpha'].index)
        meta, all_columns = _clear_metadata(
            args['metadata'], args['categorials'], args['ordinals'],
            args['intervals'], args['dates'], for_metadata_correlation=False)
        _alpha = args['alpha'].copy()
        _alpha.name = COL_NAME_ALPHA
        meta_alpha = meta.loc[:, all_columns].merge(
            _alpha.to_frame(), left_index=True, right_index=True)
        if args['metadata'].shape[0] != args['alpha'].shape[0]:
            sys.stderr.write(
                'You provided %s and %s samples in metadata and alpha '
                'respectively. Merging to %s samples for further analysis.\n'
                % (args['metadata'].shape[0], args['alpha'].shape[0],
                   meta_alpha.shape[0]))

        meta_alpha.to_csv('%s/metadata_alpha.tsv' % workdir, sep="\t",
                          index=False)
        with open('%s/rscript.R' % workdir, 'w') as f:
            f.write('library(vegan)\n')
            f.write('meta_alpha = read.csv(\'%s/metadata_alpha.tsv\', '
                    'stringsAsFactors=FALSE, sep=\'\\t\')\n' % workdir)
            f.write('vars_cat = c(\'%s\')\n' %
                    "', '".join(categorials + list(ordinals.keys())))
            f.write('meta_alpha[vars_cat] = lapply(meta_alpha[vars_cat],'
                    ' factor)\n')
            f.write('meta_alpha = meta_alpha[complete.cases(meta_alpha), ]\n')
            f.write('mod0 <- rda(meta_alpha$%s ~ 1., meta_alpha)  # Model with'
                    ' intercept only\n' % COL_NAME_ALPHA)
            f.write('mod1 <- rda(meta_alpha$%s ~ ., meta_alpha)  # Model with'
                    ' all explanatory variables\n' % COL_NAME_ALPHA)
            f.write('step.res <- ordiR2step(mod0, mod1, perm.max = 1000)\n')
            f.write('write.table(step.res$anova, file=\'%s/result.tsv\', '
                    'quote=FALSE, sep=\'\\t\', col.names = NA)\n' % workdir)

    def commands(workdir, ppn, args):
        commands = []

        if settings.GRIDNAME != 'JLU':
            commands.append('module load %s' % settings.R_MODULE)
        commands.append('R --vanilla < %s/rscript.R' % workdir)

        return commands

    def post_execute(workdir, args):
        try:
            rda = pd.read_csv('%s/result.tsv' % workdir, sep='\t', index_col=0)
        except EmptyDataError:
            return {'table': pd.DataFrame()}

        # drop fields not starting with a +, i.e. are <All variables> or <none>
        rda = rda.loc[[idx for idx in rda.index if idx.startswith('+')], :]

        # compute adjusted effect size
        rda['effect size'] = rda['R2.adj'] - ([0] + list(
            rda['R2.adj'].values)[:-1])

        rda.index = map(lambda x: x.replace('+ ', ''), rda.index)

        rda = rda.reset_index().rename(columns={'index': 'covariate'})

        return {'table': rda}

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
                      'dates': dates},
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache,
                     ppn=1,
                     environment=settings.QIIME2_ENV,
                     **executor_args)


def redundancy_analysis_beta(metadata, beta, metric_name,
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

        meta_beta = meta.loc[:, list(all_columns)].merge(
            dimred, left_index=True, right_index=True)
        if args['metadata'].shape[0] != dimred.shape[0]:
            sys.stderr.write(
                'You provided %s and %s samples in metadata and beta '
                'respectively. Merging to %s samples for further analysis.\n'
                % (args['metadata'].shape[0], dimred.shape[0],
                   meta_beta.shape[0]))

        meta_beta.to_csv('%s/metadata_beta.tsv' % workdir, sep="\t",
                         index=False)
        with open('%s/rscript.R' % workdir, 'w') as f:
            f.write('library(vegan)\n')
            f.write('meta_beta = read.csv(\'%s/metadata_beta.tsv\', '
                    'stringsAsFactors=FALSE, sep=\'\\t\')\n' % workdir)
            f.write('vars_cat = c(\'%s\')\n' %
                    "', '".join(categorials + list(ordinals.keys())))
            f.write('vars_pcs = c(\'%s\')\n' %
                    "', '".join([c for c in dimred.columns]))
            f.write('vars_meta = c(\'%s\')\n' %
                    "', '".join(all_columns))
            f.write('meta_beta[vars_cat] = lapply(meta_beta[vars_cat],'
                    ' factor)\n')
            f.write('meta_beta = meta_beta[complete.cases(meta_beta), ]\n')
            f.write('mod0 <- rda(meta_beta[, vars_pcs] ~ 1., meta_beta[, '
                    'vars_meta])  # Model with intercept only\n')
            f.write('mod1 <- rda(meta_beta[, vars_pcs] ~ ., meta_beta[, '
                    'vars_meta])  # Model with all explanatory variables\n')
            if args['seed'] is not None:
                f.write('set.seed(%i)\n' % args['seed'])
            f.write('step.res <- ordiR2step(mod0, mod1, perm.max = 1000)\n')
            f.write('write.table(step.res$anova, file=\'%s/result.tsv\', '
                    'quote=FALSE, sep=\'\\t\', col.names = NA)\n' % workdir)

    def commands(workdir, ppn, args):
        commands = []

        if settings.GRIDNAME != 'JLU':
            commands.append('module load %s' % settings.R_MODULE)
        commands.append('R --vanilla < %s/rscript.R' % workdir)

        return commands

    def post_execute(workdir, args):
        try:
            rda = pd.read_csv('%s/result.tsv' % workdir, sep='\t', index_col=0)
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

        return {'table': rda, 'seed': args['seed']}

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
                      'seed': seed},
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache,
                     ppn=1,
                     environment=settings.QIIME2_ENV,
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
        for factor in re.split('\W+', args['formula']):
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

        args['metadata'].loc[idx_samples, :].to_csv('%s/metadata.tsv' % workdir, sep="\t",
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
            f.write('res <- adonis(dm ~ %s, data=meta, %s permutations=%i %s)\n' % (args['formula'], flag_strat, args['permutations'], flag_parallel))
            f.write('write.table(res$aov.tab, "%s/result.tsv", sep="\\t", append=F, quote=FALSE)\n' % workdir)
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
