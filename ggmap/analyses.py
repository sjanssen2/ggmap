import tempfile
import shutil
import subprocess
import sys
import hashlib
import os
import pickle
from io import StringIO
import collections
import datetime
import time
import numpy as np
import json
import re
import csv
from glob import glob
from IPython.display import Image
from tqdm import tqdm

import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skbio import OrdinationResults

from skbio.stats.distance import DistanceMatrix
from skbio.tree import TreeNode
from skbio.io import read
from skbio import DNA

from biom.table import Table
from biom.util import biom_open

from ggmap.snippets import (pandas2biom, cluster_run, biom2pandas, sync_counts_metadata, check_column_presents, adjust_saturation, collapseCounts_objects, get_conda_activate_cmd, plotTaxonomy)
from ggmap import settings
import seaborn as sns
import networkx as nx

plt.switch_backend('Agg')
plt.rc('font', family='DejaVu Sans')
settings.init()


def _get_ref_phylogeny(file_tree=None, env=settings.QIIME_ENV):
    """Use QIIME config to infer location of reference tree or pass given tree.

    Parameters
    ----------
    file_tree : str
        Default: None.
        If None is set, than we need to activate qiime environment, print
        config and search for the rigth path information.
        Otherwise, specified reference tree is returned without doing anything.
    env : str
        Default: global constant settings.QIIME_ENV value.
        Conda environment name for QIIME.

    Returns
    -------
    Filepath to reference tree.
    """
    if file_tree is not None:
        return file_tree
    if settings.FILE_REFERENCE_TREE is None:
        err = StringIO()
        with subprocess.Popen(("source activate %s && "
                               "print_qiime_config.py "
                               "| grep 'pick_otus_reference_seqs_fp:'" %
                               env),
                              shell=True,
                              stdout=subprocess.PIPE,
                              executable="bash") as call_x:
            out, err = call_x.communicate()
            if (call_x.wait() != 0):
                raise ValueError("_get_ref_phylogeny(): something went wrong")

            # convert from b'' to string
            out = out.decode()
            # split key:\tvalue
            out = out.split('\t')[1]
            # remove trailing \n
            out = out.rstrip()
            # chop '/rep_set/97_otus.fasta' from found path
            out = '/'.join(out.split('/')[:-2])
            settings.FILE_REFERENCE_TREE = out + '/trees/97_otus.tree'
    return settings.FILE_REFERENCE_TREE


def _getremaining(counts_sums:pd.Series, sample_grouping:pd.Series=None):
    """Compute number of samples that have at least X read counts.

    Parameters
    ----------
    counts_sum : Pandas.Series
        Reads per sample.
    sample_grouping : pd.Series
        Default: None
        Group samples according to this series.

    Returns
    -------
    Pandas.DataFrame:
        Index = sequencing depths,
        Values = number samples with at least this sequencing depth.
    """
    if (sample_grouping is not None) and (len(set(counts_sums.index) & set(sample_grouping.index)) <= 0):
        raise ValueError("Feature counts and metadata seem to share no sample indices. Have you set your metadata.index correctly?")
    sums = pd.concat([counts_sums, sample_grouping], axis=1, sort=False).dropna()
    if sample_grouping is not None:
        grps = sums.groupby(sample_grouping.name)[0]
    else:
        grps = [('remaining', sums[0])]

    remainings = []
    for grp, g in grps:
        rem = g.shape[0]+1 - g.value_counts().sort_index().cumsum()
        rem.name = grp
        remainings.append(rem)
    remainings = pd.concat(remainings, axis=1)
    if sample_grouping is not None:
        return remainings
    else:
        # 2023-03-06: fillna ... leads to additional horizontal lines when sample group switches, nut sure why I used fillna before?!
        return remainings.ffill().bfill().astype(int)


def _parse_alpha_div_collated(workdir, samplenames):
    """Parse QIIME's alpha_div_collated file for plotting with matplotlib.

    Parameters
    ----------
    workdir : str
        Directory path to workdir which contains all indidually computed alpha
        diversities per rarefaction depth and iteration, i.e. the product of
        rarefaction_curves() execution.
    samplenames : [str]
        The expected sample names. Not necessarily all samples have sufficient
        counts to be covered by all depth, therefore we might otherwise mis
        samples.

    Returns
    -------
    Pandas.DataFrame with the averaged (over all iterations) alpha diversities
    per rarefaction depth per sample.
    """
    res = []
    for dir_alpha in tqdm(next(os.walk(workdir))[1], 'collect rarefaction data'):
        parts = dir_alpha.split('_')
        depth, iteration, metric = parts[1], parts[2], '_'.join(parts[3:])
        # we get parse errors if sample names are only numeric
        # pandas tries to convert to float and migh than map two samples
        # to the same string due to rounding issues
        #alphas = pd.read_csv(
        #    '%s/%s/alpha-diversity.tsv' % (workdir, dir_alpha),
        #    sep="\t", index_col=0)
        alphas = pd.read_csv(
            '%s/%s/alpha-diversity.tsv' % (workdir, dir_alpha),
            sep="\t", dtype=str
        )
        alphas = alphas.set_index(alphas.columns[0])
        alphas = alphas.astype(float)
        alphas = alphas.reindex(index=samplenames).reset_index()
        alphas.columns = ['sample_name', 'value']
        alphas['iteration'] = iteration
        alphas['rarefaction depth'] = depth
        alphas['metric'] = metric
        res.append(alphas)
    pd_res = pd.concat(res)

    final = dict()
    for metric in pd_res['metric'].unique():
        final[metric] = pd_res[pd_res['metric'] == metric].groupby(
            ['sample_name', 'rarefaction depth'])\
            .mean(numeric_only=True)\
            .reset_index()\
            .rename(columns={'value': metric})\
            .loc[:, ['rarefaction depth', 'sample_name', metric]]
        final[metric]['rarefaction depth'] = \
            final[metric]['rarefaction depth'].astype(int)

    return final


def _plot_rarefaction_curves(data, _plot_rarefaction_curves=None,
                             control_sample_names=[],
                             sample_grouping:pd.Series=None,
                             onlyshow=None, plot_max_samples:int=1000,
                             max_depth=None):
    """Plot rarefaction curves along with loosing sample stats + read count
       histogram.

    Parameters
    ----------
    data : dict()
        The result of rarefaction_curves(), i.e. a dict with the three keys
        - metrics
        - remaining
        - readcounts
    control_sample_names : {str}
        Default: [].
        A set of samples that serve as controls, i.e. samples that we are
        willing to loose during rarefaction. Only used for plotting.
    plot_max_samples : int
        By default, each sample will be plotted individually. For huge feature
        tables, this can take very long time and exceed memory. Therefore,
        we prematurely stop iterating samples, once plot_max_samples have
        been plotted.

    Returns
    -------
    Matplotlib figure
    """
    grp_colors = None
    if sample_grouping is not None:
        # check if sample IDs correspond to grouping names
        if len(set(data['readcounts'].index) - set(sample_grouping.index)) > 0:
            raise ValueError("Not all samples in rarefaction grouping got a value!")
        grp_colors = {grp: col
                      for grp, col
                      in zip(sample_grouping.unique(), sns.color_palette() * (int(len(sample_grouping.unique()) / 10)+1))}

    fig, axes = plt.subplots(2+len(data['metrics']) if onlyshow is None else 1,
                             1,
                             figsize=(5, (2+len(data['metrics']))*5) if onlyshow is None else (5, 5),
                             sharex=False)

    # read count histogram
    if onlyshow is None:
        ax = axes[0]
        n, bins, patches = ax.hist(data['readcounts'].fillna(0.0),
                                   50,
                                   facecolor='black',
                                   alpha=0.75)
        ax.set_title('Read count distribution across samples')
        ax.set_xlabel("read counts")
        ax.set_ylabel("# samples")
        ax.get_xaxis().set_major_formatter(
            FuncFormatter(lambda x, p: format(int(x), ',')))

    # loosing samples
    lost = data['remaining'].max().sum() - data['remaining'].sum(axis=1)
    if onlyshow is None:
        ax = axes[1]
        for control in control_sample_names:
            # plot a vertical gray line to indicate one control sample.
            if control in data['readcounts']:
                ax.axvline(x=data['readcounts'].loc[control], color='lightgray')

        for grp, g in data['remaining'].T.iterrows():
            color = None
            if sample_grouping is not None:
                color = grp_colors[grp]
            ax.plot(g.index, g, label=grp, color=color)
            # add an horizontal line for the left region where no samples are lost, since no sample has that little reads
            ax.plot([0, g.index[0]], [g.iloc[0], g.iloc[0]], color=color, lw=2)
        if sample_grouping is not None:
            ax.legend(title=sample_grouping.name, bbox_to_anchor=(1.05, 1))
        else:
            ax.plot(lost.index, lost, label='lost')

        ax.set_xlabel("rarefaction depth")
        ax.set_ylabel("# samples")
        ax.set_title('How many of the %i samples do we loose?' %
                     data['readcounts'].shape[0])
        ax.get_xaxis().set_major_formatter(
            FuncFormatter(lambda x, p: format(int(x), ',')))

    lostHalf = abs(data['remaining'].sum(axis=1) - lost)
    lostHalf = lostHalf[lostHalf == lostHalf.min()].index[0]
    maxX = lostHalf * 1.1

    if maxX < data['metrics'][list(data['metrics'].keys())[0]]['rarefaction depth'].mean():
        maxX = data['metrics'][list(data['metrics'].keys())[0]]['rarefaction depth'].max() * 1.1
    if max_depth is not None:
        # if user gives maximal rarefaction depth (which is NOT the default),
        # also plot this region!
        maxX = max_depth * 1.1
    if onlyshow is None:
        ax.set_xlim(0, maxX)

    for i, metric in enumerate(sorted(data['metrics'].keys())):
        ax = None
        if onlyshow is not None:
            if metric != onlyshow:
                continue
            else:
                ax = axes
        else:
            ax = axes[i+2]

        # using different drawing methods for normal grouping by sample name and alternative grouping due to speed
        if sample_grouping is None:
            num_total_samples = data['metrics'][metric]['sample_name'].unique().shape[0]
            num_skipped_samples = num_total_samples - plot_max_samples
            for ns, (sample, g) in enumerate(data['metrics'][metric].groupby('sample_name')):
                gsorted = g.sort_values('rarefaction depth')
                ax.errorbar(
                    gsorted['rarefaction depth'],
                    gsorted[gsorted.columns[-1]])
                if ns >= plot_max_samples:
                    if i == 0:
                        print(("Abort plotting rarefaction curves, as data for %i have already been visualized.\n"
                               "Note that you miss additional %i (=%.1f%%) samples!\n"
                               "Either increase parameter 'plot_max_samples', but this will result in time and memory demanding computations.\n"
                               "Alternatively, group samples by metadata - "
                               "as each group only result in ONE line (with error bars).") % (
                                plot_max_samples,
                                num_skipped_samples,
                                num_skipped_samples / num_total_samples * 100))
                    ax.set_title("Skipped %i of %i (=%.1f%%) samples!" % (num_skipped_samples, num_total_samples, num_skipped_samples / num_total_samples * 100))
                    break
        else:
            grps = data['metrics'][metric].merge(sample_grouping, left_on='sample_name', right_index=True).groupby(sample_grouping.name)
            for grp, g in grps:
                d = g.groupby('rarefaction depth')[metric].describe()
                d = d[d['count'] > 0]
                if d.shape[0] > 0:
                    ax.errorbar(d.index, d['mean'], yerr=[e if pd.notnull(e) else 0 for e in d['std']], label=grp, color=grp_colors[grp], ecolor=adjust_saturation(grp_colors[grp], 0.9))
            if i == 0:
                ax.legend(title=sample_grouping.name, bbox_to_anchor=(1.05, 1))
        ax.set_ylabel(metric)
        ax.set_xlabel('rarefaction depth')
        ax.set_xlim(0, maxX)
        ax.get_xaxis().set_major_formatter(
            FuncFormatter(lambda x, p: format(int(x), ',')))

    return fig


def writeReferenceTree(fp_reftree, workdir, fix_zero_len_branches=False,
                       verbose=sys.stderr, name_analysis='beta_diversity'):
    """Pre-process skbio tree to ensure every branch has a length info.

    Notes
    -----
    This is very costly and should only be done when really necessary!

    Paramaters
    ----------
    fp_reftree : str
        Filepath to reference phylogenetic tree for alpha- or beta- diversity
        computation.
    workdir : str
        Filepath to working directory of executore.
    fix_zero_len_branches : bool
        Default: False
        If True, add 0 length to branches without length information,
        otherwise just use the tree as is.
    """
    if fix_zero_len_branches:
        tree_ref = TreeNode.read(
            _get_ref_phylogeny(fp_reftree), format='newick')
        for node in tree_ref.preorder():
            if node.length is None:
                node.length = 0
        tree_ref.write(workdir+'/reference.tree')
    else:
        if verbose is not None:
            verbose.write(
                ('%s: skipping check for branches without '
                 'length information. Reactivate via fix_zero_le'
                 'n_branches=True.\n') % name_analysis)
        shutil.copyfile(
            _get_ref_phylogeny(fp_reftree),
            workdir+'/reference.tree')


def rarefaction_curves(counts,
                       metrics=["PD_whole_tree", "shannon", "observed_features"],
                       num_steps=20, reference_tree=None, max_depth=None,
                       min_depth=1000, num_iterations=10,
                       control_sample_names=[], fix_zero_len_branches=False,
                       sample_grouping:pd.Series=None, onlyshow=None,
                       pmem='8GB', plot_max_samples:int=5000, **executor_args):
    """Produce rarefaction curves, i.e. reads/sample and alpha vs. depth plots.

    Parameters
    ----------
    counts : Pandas.DataFrame OR biom.table.Table
        The raw read counts. Columns are samples, rows are features.
    metrics : [str]
        List of alpha diversity metrics to use.
        Default is ["PD_whole_tree", "shannon", "observed_features"]
    num_steps : int
        Number of different rarefaction steps to test. The higher the slower.
        Default is 20.
    reference_tree : str
        Filepath to a newick tree file, which will be the phylogeny for unifrac
        alpha diversity distances. By default, qiime's GreenGenes tree is used.
    max_depth : int
        Maximal rarefaction depth. By default counts.sum().describe()['75%'] is
        used.
    min_depth : int
        Minimal rarefaction depth. By default: 1000.
    num_iterations : int
        Default: 10.
        Number of iterations to rarefy the input table.
    control_sample_names : {str}
        Default: [].
        A set of samples that serve as controls, i.e. samples that we are
        willing to loose during rarefaction. Only used for plotting.
    onlyshow : str
        Only return the single rarefaction graph with the given metric name.
        Default: None, i.e. return all graphs
    plot_max_samples : int
        Stop plotting rarefaction curves for individual samples, after
        plot_max_samples have been plotted to avoid overallocating memory / time.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    plt figure
    """
    assert isinstance(counts, pd.DataFrame) or isinstance(counts, Table)
    if isinstance(counts, pd.DataFrame):
        counts = counts.fillna(0.0)

    def pre_execute(workdir, args):
        # store counts as a biom file
        if isinstance(counts, pd.DataFrame):
            pandas2biom(workdir+'/input.biom', args['counts'])
        elif isinstance(counts, Table):
            with biom_open(workdir+'/input.biom', 'w') as f:
                counts.to_hdf5(f, "ggmap")
        # copy reference tree and correct missing branch lengths
        if len(set(args['metrics']) &
               set(['PD_whole_tree'])) > 0:
            if 'verbose' not in executor_args:
                verbose = sys.stderr
            else:
                verbose = executor_args['verbose']
            writeReferenceTree(args['reference_tree'], workdir,
                               fix_zero_len_branches, verbose=verbose,
                               name_analysis='rarefaction_curves')

        # prepare execution list
        sample_count_sums = None
        if isinstance(counts, pd.DataFrame):
            sample_count_sums = args['counts'].sum()
        elif isinstance(counts, Table):
            sample_count_sums = pd.Series(args['counts'].sum(axis='sample'))
        max_rare_depth = sample_count_sums.describe()['75%']
        if args['max_depth'] is not None:
            max_rare_depth = args['max_depth']
        if args['min_depth'] is None:
            # fall back to a default minimal rarefaction depth of 1000, if
            # min_depth argument is None
            args['min_depth'] = 1000
        f = open("%s/commands.txt" % workdir, "w")
        for depth in np.linspace(max(args['min_depth'], sample_count_sums.min()),
                                 max_rare_depth,
                                 args['num_steps'], endpoint=True):
            for iteration in range(args['num_iterations']):
                f.write("%i\t%s\n" % (
                    depth, iteration))
        f.close()

    def commands(workdir, ppn, args):
        commands = {'pre': [], 'main': [], 'post': []}

        commands['pre'].append(
            ('qiime tools import '
             '--input-path %s '
             '--type "FeatureTable[Frequency]" '
             '--output-path %s') %
            (workdir+'/input.biom', workdir+'/input'))
        if args['reference_tree'] is not None:
            commands['pre'].append(
                ('qiime tools import '
                 '--input-path %s '
                 '--output-path %s '
                 '--type "Phylogeny[Rooted]"') %
                (workdir+'/reference.tree',
                 workdir+'/reference_tree.qza'))

        commands['main'] = [
            ('var_depth=`head -n ${%s} %s/commands.txt | '
             'tail -n 1 | cut -f 1`') % (settings.VARNAME_PBSARRAY, workdir),
            ('var_iteration=`head -n ${%s} %s/commands.txt | '
             'tail -n 1 | cut -f 2`') % (settings.VARNAME_PBSARRAY, workdir)]
        commands['main'].append((
            'qiime feature-table rarefy '
            '--i-table %s/input.qza '
            '--p-sampling-depth ${var_depth} '
            '--o-rarefied-table %s/rare_${var_depth}_${var_iteration} ') % (
            workdir, workdir))
        for metric in args['metrics']:
            plugin = 'alpha'
            treeinput = ''
            if metric == 'PD_whole_tree':
                plugin = 'alpha-phylogenetic'
                treeinput = '--i-phylogeny %s' % (
                    workdir+'/reference_tree.qza')
            commands['main'].append(
                ('qiime diversity %s '
                 '--i-table %s/rare_${var_depth}_${var_iteration}.qza '
                 '--p-metric %s '
                 ' %s '
                 '--o-alpha-diversity '
                 '%s/alpha_${var_depth}_${var_iteration}_%s') %
                (plugin, workdir,
                 _update_metric_alpha(metric),
                 treeinput, workdir, metric))
            commands['main'].append(
                ('qiime tools export '
                 '--input-path %s/alpha_${var_depth}_${var_iteration}_%s.qza '
                 '--output-path %s/alpharaw_${var_depth}_${var_iteration}_%s')
                % (workdir, metric, workdir, metric))

        return commands

    def post_execute(workdir, args):
        sample_ids = None
        sums = None
        if isinstance(args['counts'], pd.DataFrame):
            sample_ids = args['counts'].columns
            sums = args['counts'].sum()
        elif isinstance(args['counts'], Table):
            sample_ids = args['counts'].ids(axis='sample')
            sums = pd.Series(args['counts'].sum(axis='sample'), index=args['counts'].ids('sample'))

        results = {'metrics':
                   _parse_alpha_div_collated(workdir, sample_ids),
                   'readcounts': sums}
        return results

    def post_cache(cache_results):
        cache_results['results']['remaining'] = _getremaining(cache_results['results']['readcounts'], sample_grouping)
        cache_results['results'] = \
            _plot_rarefaction_curves(cache_results['results'],
                                     control_sample_names=control_sample_names,
                                     sample_grouping=sample_grouping,
                                     onlyshow=onlyshow,
                                     plot_max_samples=plot_max_samples,
                                     max_depth=max_depth)
        return cache_results


    return _executor('rare',
                     {'counts': counts,
                      'metrics': metrics,
                      'num_steps': num_steps,
                      'max_depth': max_depth,
                      'min_depth': min_depth,
                      'num_iterations': num_iterations,
                      'reference_tree': reference_tree},
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache,
                     environment=settings.QIIME2_ENV,
                     ppn=1,
                     array=num_steps*num_iterations,
                     **executor_args)


def rarefy(counts, rarefaction_depth,
           ppn=1, **executor_args):
    """Rarefies a given OTU table to a given depth. This depth should be
       determined by looking at rarefaction curves.

    Paramaters
    ----------
    counts : Pandas.DataFrame OR biom.table.Table
        OTU counts
    rarefaction_depth : int
        Rarefaction depth that must be applied to counts.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    Pandas.DataFrame: Rarefied OTU table."""

    assert isinstance(counts, pd.DataFrame) or isinstance(counts, Table)
    if isinstance(counts, pd.DataFrame):
        counts = counts.fillna(0.0)

    def pre_execute(workdir, args):
        # store counts as a biom file
        if isinstance(counts, pd.DataFrame):
            pandas2biom(workdir+'/input.biom', args['counts'])
        elif isinstance(counts, Table):
            with biom_open(workdir+'/input.biom', 'w') as f:
                counts.to_hdf5(f, "ggmap")

    def commands(workdir, ppn, args):
        commands = []

        commands.append((
            'qiime tools import '
            '--input-path %s/input.biom '
            '--output-path %s/counts '
            '--type "FeatureTable[Frequency]"') % (workdir, workdir))
        commands.append((
            'qiime feature-table rarefy '
            '--i-table %s/counts.qza '
            '--p-sampling-depth %i '
            '--o-rarefied-table %s/rare ') % (
            workdir, args['rarefaction_depth'], workdir))
        commands.append(
            ('qiime tools export '
             '--input-path %s/rare.qza '
             '--output-path %s') %
            (workdir, workdir))

        return commands

    def post_execute(workdir, args):
        if isinstance(counts, pd.DataFrame):
            return biom2pandas(workdir+'/feature-table.biom')
        else:
            with biom_open(workdir+'/feature-table.biom', 'r') as f:
                return Table.from_hdf5(f)

    return _executor('rarefy',
                     {'counts': counts,
                      'rarefaction_depth': rarefaction_depth},
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=ppn,
                     environment=settings.QIIME2_ENV,
                     **executor_args)


def _update_metric_alpha(metric):
    if metric == 'PD_whole_tree':
        return 'faith_pd'
    return metric


def alpha_diversity(counts, rarefaction_depth,
                    metrics=["PD_whole_tree", "shannon", "observed_features"],
                    num_iterations=10, reference_tree=None,
                    fix_zero_len_branches=False, ppn=1,
                    **executor_args):
    """Computes alpha diversity values for given BIOM table.

    Paramaters
    ----------
    counts : Pandas.DataFrame
        OTU counts
    rarefaction_depth : int
        Rarefaction depth that must be applied to counts.
    metrics : [str]
        Alpha diversity metrics to be computed.
    num_iterations : int
        Number of iterations to rarefy the input table.
    reference_tree : str
        Reference tree file name for phylogenetic metics like unifrac.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    Pandas.DataFrame: alpha diversity values for each sample (rows) for every
    chosen metric (columns)."""

    if rarefaction_depth is None:
        num_iterations = 1

    def pre_execute(workdir, args):
        # store counts as a biom file
        pandas2biom(workdir+'/input.biom', args['counts'].fillna(0.0))
        os.mkdir(workdir+'/rarefaction/')
        os.mkdir(workdir+'/alpha/')
        os.mkdir(workdir+'/alpha_plain/')
        # copy reference tree and correct missing branch lengths
        if len(set(args['metrics']) &
               set(['PD_whole_tree'])) > 0:
            if 'verbose' not in executor_args:
                verbose = sys.stderr
            else:
                verbose = executor_args['verbose']
            writeReferenceTree(args['reference_tree'], workdir,
                               fix_zero_len_branches, verbose=verbose,
                               name_analysis='alpha_diversity')

    def commands(workdir, ppn, args):
        commands = {'pre': [], 'main': [], 'post': []}

        commands['pre'].append(
            ('qiime tools import '
             '--input-path %s '
             '--type "FeatureTable[Frequency]" '
             # '--source-format BIOMV210Format '
             '--output-path %s ') %
            (workdir+'/input.biom', workdir+'/input'))
        if 'PD_whole_tree' in args['metrics']:
            commands['pre'].append(
                ('qiime tools import '
                 '--input-path %s '
                 '--output-path %s '
                 '--type "Phylogeny[Rooted]"') %
                (workdir+'/reference.tree',
                 workdir+'/reference_tree.qza'))

        file_raretable = workdir+'/rarefaction/rare_%s_${%s}.qza' % (
            args['rarefaction_depth'], settings.VARNAME_PBSARRAY)
        if args['rarefaction_depth'] is not None:
            commands['main'].append(
                ('qiime feature-table rarefy '
                 '--i-table %s '
                 '--p-sampling-depth %i '
                 '--o-rarefied-table %s') %
                (workdir+'/input.qza', args['rarefaction_depth'],
                 file_raretable)
            )
        else:
            commands['main'].append('cp %s %s' % (
                workdir+'/input.qza',
                workdir+'/rarefaction/rare_%s_${%s}.qza' % (
                    rarefaction_depth, settings.VARNAME_PBSARRAY)))
        for metric in args['metrics']:
            file_alpha = workdir+'/alpha/alpha_%s_${%s}_%s.qza' % (
                args['rarefaction_depth'], settings.VARNAME_PBSARRAY, metric)
            plugin = 'alpha'
            treeinput = ''
            if metric == 'PD_whole_tree':
                plugin = 'alpha-phylogenetic'
                treeinput = '--i-phylogeny %s' % (
                    workdir+'/reference_tree.qza')
            commands['main'].append(
                ('qiime diversity %s '
                 '--i-table %s '
                 '--p-metric %s '
                 ' %s '
                 '--o-alpha-diversity %s') %
                (plugin, file_raretable,
                 _update_metric_alpha(metric),
                 treeinput,
                 file_alpha))
            commands['main'].append(
                ('qiime tools export '
                 '--input-path %s/alpha/alpha_%s_${%s}_%s.qza '
                 '--output-path %s/alpha_plain/%s/${%s}/%s') %
                (workdir, args['rarefaction_depth'], settings.VARNAME_PBSARRAY, metric,
                 workdir, args['rarefaction_depth'], settings.VARNAME_PBSARRAY, metric))

        return commands

    def post_execute(workdir, args):
        dir_plain = '%s/alpha_plain/%s/' % (workdir, args['rarefaction_depth'])
        results_alpha = dict()
        for iteration in next(os.walk(dir_plain))[1]:
            for metric in next(os.walk(dir_plain + '/' + iteration))[1]:
                if metric not in results_alpha:
                    results_alpha[metric] = []
                file_alpha = '%s/%s/%s/alpha-diversity.tsv' % (
                    dir_plain, iteration, metric)
                alpha_results = pd.read_csv(file_alpha, sep="\t", dtype=str)
                alpha_results = alpha_results.set_index(alpha_results.columns[0])
                alpha_results = alpha_results.astype(float)
                results_alpha[metric].append(alpha_results)

        for metric in results_alpha.keys():
            results_alpha[metric] = pd.concat(
                results_alpha[metric], axis=1).mean(axis=1)
            results_alpha[metric].name = metric
        result = pd.concat(results_alpha.values(), axis=1)
        result.index.name = 'iter%s_depth%s' % (
            args['num_iterations'], args['rarefaction_depth'])
        return result

    return _executor('adiv',
                     {'counts': counts.fillna(0.0),
                      'metrics': metrics,
                      'rarefaction_depth': rarefaction_depth,
                      'num_iterations': num_iterations,
                      'reference_tree': reference_tree},
                     pre_execute,
                     commands,
                     post_execute,
                     environment=settings.QIIME2_ENV,
                     ppn=ppn,
                     array=num_iterations,
                     **executor_args)


def _update_metric_beta(metric):
    if metric == 'bray_curtis':
        return 'braycurtis'
    elif metric == 'weighted_unifrac':
        return 'weighted_normalized_unifrac'
    return metric


def beta_diversity(counts,
                   metrics=["unweighted_unifrac",
                            "weighted_unifrac",
                            "bray_curtis"],
                   reference_tree=None,
                   fix_zero_len_branches=False,
                   remove_zero_entropy_samples: bool=False,
                   use_sklearn_parallel: bool=False,
                   **executor_args):
    """Computes beta diversity values for given BIOM table.

    Parameters
    ----------
    counts : Pandas.DataFrame OR biom.table.Table
        OTU counts
    metrics : [str]
        Beta diversity metrics to be computed.
    reference_tree : str
        Reference tree file name for phylogenetic metics like unifrac.
    use_sklearn_parallel : bool
        For huge feature-tables, you might want to replace the Qiime2 distance
        computation with a more hacky one based on scikit-learn, see
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
        To also use parallel computation of non-phylogenetic metrics.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    Dict of Pandas.DataFrame, one per metric."""
    assert isinstance(counts, pd.DataFrame) or isinstance(counts, Table)
    if isinstance(counts, pd.DataFrame):
        counts = counts.fillna(0.0)

    zeroEntropySamples = []
    lst_samplenames = []
    if isinstance(counts, pd.DataFrame):
        lst_samplenames = counts.columns
    elif isinstance(counts, Table):
        lst_samplenames = counts.ids('sample')
    for sample in lst_samplenames:
        if isinstance(counts, pd.DataFrame):
            if counts[sample].value_counts().shape[0] <= 1:
                zeroEntropySamples.append(sample)
        elif isinstance(counts, Table):
            if len(np.unique(counts.data(sample, axis='sample'))) <= 1:
                zeroEntropySamples.append(sample)
    if remove_zero_entropy_samples is False:
        if len(zeroEntropySamples) > 0:
            raise ValueError("%i samples have zero entropy, i.e. all features have the same value, which can lead to downstream errors and is quite unlikely to happen for real data. Please check and remove those samples:\n%s\nOr set 'remove_zero_entropy_samples=True'." % (len(zeroEntropySamples), ', '.join(map(lambda x: '"%s"' % x, zeroEntropySamples))))
    else:
        if isinstance(counts, pd.DataFrame):
            counts = counts.loc[:, [s for s in counts.columns if s not in zeroEntropySamples]]
        else:
            counts = counts.filter([s for s in counts.ids('sample') if s not in zeroEntropySamples], axis='sample')
        if 'verbose' not in executor_args:
            verbose = sys.stderr
        else:
            verbose = executor_args['verbose']
        if len(zeroEntropySamples) > 0:
            verbose.write('Silently excluded %i samples, due to their zero entropy.\n' % len(zeroEntropySamples))
    if counts.shape[1] <= 1:
        raise ValueError("Your feature table has less than two samples!")

    for sample in lst_samplenames:
        if ',' in sample:
            raise ValueError("You sample names contain ',' character. This will hurt skbio. You need to remove/replace ','!")

    def pre_execute(workdir, args):
        # store counts as a biom file
        if isinstance(args['counts'], pd.DataFrame):
            pandas2biom(workdir+'/input.biom', args['counts'])
        elif isinstance(args['counts'], Table):
            with biom_open(workdir+'/input.biom', 'w') as f:
                args['counts'].to_hdf5(f, "ggmap")

        os.mkdir(workdir+'/beta_qza')
        # copy reference tree and correct missing branch lengths
        if len(set(args['metrics']) &
               set(['unweighted_unifrac', 'weighted_unifrac'])) > 0:
            if 'verbose' not in executor_args:
                verbose = sys.stderr
            else:
                verbose = executor_args['verbose']
            writeReferenceTree(args['reference_tree'], workdir,
                               fix_zero_len_branches, verbose=verbose,
                               name_analysis='beta_diversity')

        if use_sklearn_parallel:
            # copy the necessary helper python script into the working directory
            # to make sure it can be called with the correct file path
            shutil.copy(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/beta_parallel.py')),
                        '%s' % workdir)

    def commands(workdir, ppn, args):
        metrics_phylo = []
        metrics_nonphylo = []
        for metric in map(_update_metric_beta, args['metrics']):
            if metric.endswith('_unifrac'):
                metrics_phylo.append(metric)
            else:
                metrics_nonphylo.append(metric)

        commands = {'pre': [], 'main': [], 'post': []}
        # import biom table into q2 fragment
        # commands.append('mkdir -p %s' % (workdir+'/beta_qza'))
        if (not use_sklearn_parallel) or (len(metrics_phylo) > 0):
            commands['pre'].append(
                ('qiime tools import '
                 '--input-path %s '
                 '--type "FeatureTable[Frequency]" '
                 '--output-path %s ') %
                (workdir+'/input.biom', workdir+'/input'))
        if len(metrics_phylo) > 0:
            commands['pre'].append(
                ('qiime tools import '
                 '--input-path %s '
                 '--output-path %s '
                 '--type "Phylogeny[Rooted]"') %
                (workdir+'/reference.tree',
                 workdir+'/reference_tree.qza'))
        for i, metric in enumerate(metrics_nonphylo + metrics_phylo):
            if metric in metrics_nonphylo:
                if use_sklearn_parallel:
                    commands['main'].append(
                        ('if [ ${%s} -eq %i ]; then '
                         'python %s/beta_parallel.py '
                         '%s/input.biom '
                         '%s/beta_qza/%s.qza '
                         '%s %s; '
                         'fi') % (
                         settings.VARNAME_PBSARRAY, i+1,
                         workdir,
                         workdir,
                         workdir, metric,
                         metric,
                         ppn))
                else:
                    commands['main'].append(
                        ('if [ ${%s} -eq %i ]; then qiime diversity beta '
                         '--i-table %s '
                         '--p-metric %s '
                         '--p-n-jobs %i '
                         '--o-distance-matrix %s%s; fi') %
                        (settings.VARNAME_PBSARRAY, i+1,
                         workdir+'/input.qza', metric, ppn,
                         workdir+'/beta_qza/', metric))
            elif metric in metrics_phylo:
                commands['main'].append(
                    ('if [ ${%s} -eq %i ]; then qiime diversity beta-phylogenetic '
                     '--i-table %s '
                     '--i-phylogeny %s '
                     '--p-metric %s '
                     '--p-threads %i '
                     '--o-distance-matrix %s%s; fi') %
                    (settings.VARNAME_PBSARRAY, i+1,
                     workdir+'/input.qza', workdir+'/reference_tree.qza',
                     metric,
                     # bug in q2 plugin: crashs 'if the number of threads requested
                     # exceeds the approximately n / 2 samples, then an exception
                     # is raised'
                     min(ppn, int(args['counts'].shape[1] / 2.2)),
                     workdir+'/beta_qza/', metric))
            commands['main'].append(
                ('if [ ${%s} -eq %i ]; then qiime tools export '
                 '--input-path %s/beta_qza/%s.qza '
                 '--output-path %s/beta/%s/; fi') %
                (settings.VARNAME_PBSARRAY, i+1, workdir, metric, workdir, metric))
        return commands

    def post_execute(workdir, args):
        results = dict()
        for metric in args['metrics']:
            results[metric] = DistanceMatrix.read(
                '%s/beta/%s/distance-matrix.tsv' % (
                    workdir,
                    _update_metric_beta(metric)))
        return results

    return _executor('bdiv',
                     {'counts': counts,
                      'metrics': metrics,
                      'reference_tree': reference_tree},
                     pre_execute,
                     commands,
                     post_execute,
                     array=len(metrics),
                     environment=settings.QIIME2_ENV,
                     **executor_args)


def sepp(counts, chunksize=10000, reference_database=settings.FILE_REFERENCE_SEPP,
         #reference_phylogeny=None, reference_alignment=None,
         #reference_taxonomy=None, #reference_info=None,
         alignment_subset_size=None, placement_subset_size=None,
         ppn=20, pmem='8GB', walltime='12:00:00', debug=False,
         environment=settings.QIIME2_ENV, **executor_args):
    """Tip insertion of deblur sequences into GreenGenes backbone tree.

    Parameters
    ----------
    counts : Pandas.DataFrame | Pandas.Series | biom.table.Table
        a) OTU counts in form of a Pandas.DataFrame.
        b) If providing a Pandas.Series, we expect the index to be a fasta
           headers and the colum the fasta sequences.
        c) a biom table
    reference_database : str
        Package holding reference phylogeny+alignment+taxonomy+info for SEPP.
        Should point by default to Greengenes 13.8 99% tree.
    alignment_subset_size : int
        Default: None, i.e. 1000
        Each placement subset is further broken into
        subsets of at most these many sequences and
        a separate HMM is trained on each subset.
        The default alignment subset size is set to
        balance the exhaustiveness of the alignment
        step with the running time.
    placement-subset-size : int
        Default: None, i.e. 5000
        The tree is divided into subsets such that
        each subset includes at most these many
        subsets. The placement step places the
        fragment on only one subset, determined
        based on alignment scores. The default
        placement subset is set to make sure the
        memory requirement of the pplacer step does
        not become prohibitively large.
    chunksize: int
        Default: 10000
        SEPP jobs seem to fail if too many sequences are submitted per job.
        Therefore, we can split your sequences in chunks of chunksize.
    debug : boolean
        Request --verbose and --p-debug from qiime2 plugin.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    ???"""
    def pre_execute(workdir, args):
        if (reference_database is None) or (reference_database == ""):
            raise ValueError("Reference database is not given!")
        if (not os.path.exists(reference_database)):
            raise ValueError("Reference database cannot be found at '%s'" % reference_database)
        chunks = range(0, len(seqs), args['chunksize'])
        for chunk, i in enumerate(chunks):
            # write all deblur sequences into one file per chunk
            chunkname = chunk + 1
            if (('use_grid' not in executor_args) or (executor_args['use_grid'] is True)) and \
               (settings.GRIDNAME == 'JLU') and \
               (len(chunks) == 1):
               chunkname = 'undefined'
            file_fragments = workdir + '/sequences%s.mfa' % chunkname
            f = open(file_fragments, 'w')
            chunk_seqs = seqs[i:i + args['chunksize']]
            for sequence in chunk_seqs:
                f.write('>%s\n%s\n' % (sequence, sequence))
            f.close()

    def commands(workdir, ppn, args):
        commands = []

        # import fasta sequences into qza
        commands.append(
            ('qiime tools import '
             '--input-path %s '
             '--output-path %s/rep-seqs${%s} '
             '--type "FeatureData[Sequence]"') %
            (workdir + ('/sequences${%s}.mfa' % settings.VARNAME_PBSARRAY), workdir, settings.VARNAME_PBSARRAY))

        ss_alignment = ""
        if args['alignment_subset_size'] is not None:
            ss_alignment = ' --p-alignment-subset-size %s ' % (
                args['alignment_subset_size'])
        ss_placement = ""
        if args['placement_subset_size'] is not None:
            ss_placement = ' --p-placement-subset-size %s ' % (
                args['placement_subset_size'])

        commands.append(
            ('qiime fragment-insertion sepp '
             '--i-representative-sequences %s/rep-seqs${%s}.qza '
             '--i-reference-database %s '
             '%s'
             '--p-threads %i '
             '%s%s'
             '--output-dir %s/res_${%s}') %
            (workdir, settings.VARNAME_PBSARRAY, reference_database,
             ' --p-debug --verbose ' if debug else '',
             ppn, ss_alignment, ss_placement, workdir,
             settings.VARNAME_PBSARRAY))

        # export the placements
        commands.append(
            ('qiime tools export '
             '--input-path %s/res_${%s}/placements.qza '
             '--output-path %s/res_${%s}/') %
            (workdir, settings.VARNAME_PBSARRAY, workdir, settings.VARNAME_PBSARRAY))

        # export the tree
        commands.append(
            ('qiime tools export '
             '--input-path %s/res_${%s}/tree.qza '
             '--output-path %s/res_${%s}/') %
            (workdir, settings.VARNAME_PBSARRAY, workdir, settings.VARNAME_PBSARRAY))

        if False:  # disabled with 2020.2 since we would need to have taxonomy archive available + RDP returns better results!
            # compute taxonomy from resulting tree and placements
            ref_taxonomy = ""
            if args['reference_taxonomy'] is not None:
                ref_taxonomy = \
                    " --i-reference-taxonomy %s " % args['reference_taxonomy']
                commands.append(
                    ('qiime fragment-insertion classify-otus-experimental '
                     '--i-representative-sequences %s/rep-seqs${%s}.qza '
                     '--i-tree %s/res_${%s}/tree.qza '
                     '--i-reference-taxonomy %s/taxonomy.qza '
                     '--o-classification %s/res_taxonomy_${%s}') %
                    (workdir, settings.VARNAME_PBSARRAY, workdir, settings.VARNAME_PBSARRAY, workdir, workdir, settings.VARNAME_PBSARRAY))

            # export taxonomy to tsv file
            commands.append(
                ('qiime tools export '
                 '--input-path %s/res_taxonomy_${%s}.qza '
                 '--output-path %s/res_taxonomy_${%s}/') %
                (workdir, settings.VARNAME_PBSARRAY, workdir, settings.VARNAME_PBSARRAY))

            # move taxonomy tsv to basedir
            commands.append(
                ('mv '
                 '%s/res_taxonomy_${%s}/taxonomy.tsv '
                 '%s/taxonomy_${%s}.tsv') %
                (workdir, settings.VARNAME_PBSARRAY, workdir, settings.VARNAME_PBSARRAY))

        return commands

    def post_execute(workdir, args):
        use_grid = executor_args['use_grid'] \
            if 'use_grid' in executor_args else True
        dry = executor_args['dry'] if 'dry' in executor_args else True

        files_placement = []
        for d in next(os.walk(workdir))[1]:
            if d.startswith('res_'):
                for f in next(os.walk(workdir+'/'+d))[2]:
                    if f == 'placements.json':
                        files_placement.append(workdir+'/'+d+'/'+f)
        # if we used several chunks, we need to merge placements to produce one
        # unified insertion tree in the end
        if len(files_placement) > 1:
            sys.stderr.write("step 1) merging placement files: ")
            static = None
            placements = []
            for file_placement in files_placement:
                f = open(file_placement, 'r')
                plcmnts = json.loads(f.read())
                f.close()
                placements.extend(plcmnts['placements'])
                if static is None:
                    del plcmnts['placements']
                    static = plcmnts
            with open('%s/all_placements.json' % (workdir), 'w') as outfile:
                static['placements'] = placements
                json.dump(static, outfile)
            sys.stderr.write(' done.\n')

            sys.stderr.write("step 2) placing fragments into tree: ...")
            # guppy ran for: and consumed 45 GB of memory for 2M, chunked 10k
            # sepp benchmark:
            # real	37m39.772s
            # user	31m3.906s
            # sys	3m49.602s
            cluster_run([
                ('$HOME/miniconda3/envs/%s/bin/guppy tog -o '
                 '%s/all_tree.nwk '
                 '%s/all_placements.json') % (environment,  workdir, workdir)],
                environment=environment,
                jobname='guppy_rename',
                result="%s/all_tree.nwk" % workdir,
                ppn=1, pmem='100GB', walltime='1:00:00',
                dry=dry,
                wait=True, use_grid=use_grid)
            sys.stderr.write(' done.\n')
        else:
            sys.stderr.write("step 1+2) extracting newick tree and placements: ")
            shutil.copy('%s/res_1/tree.nwk' % workdir,
                        '%s/all_tree.nwk' % workdir)
            shutil.copy('%s/res_1/placements.json' % workdir,
                        '%s/all_placements.json' % workdir)
            sys.stderr.write(' done.\n')

        if False:
            sys.stderr.write("step 3) merge taxonomy: ")
            taxonomies = []
            for file_taxonomy in next(os.walk(workdir))[2]:
                if file_taxonomy.startswith('taxonomy_') and \
                   file_taxonomy.endswith('.tsv'):
                    taxonomies.append(pd.read_csv(workdir + '/' + file_taxonomy,
                                      sep="\t", index_col=0))
            if len(taxonomies) > 0:
                taxonomy = pd.concat(taxonomies)
            else:
                taxonomy = pd.DataFrame()
            sys.stderr.write(' done.\n')

        f = open("%s/all_tree.nwk" % workdir, 'r')
        tree = f.readlines()[0].strip()
        f.close()

        with open('%s/all_placements.json' % workdir) as json_file:
            plt_content = json.load(json_file)

        return {#'taxonomy': taxonomy,
                'tree': tree,
                'placements': plt_content['placements']}

    seqs = []
    if isinstance(counts, pd.DataFrame):
        seqs = sorted(counts.index)
    elif isinstance(counts, pd.Series):
        # typically, the input is an OTU table with index holding sequences.
        # However, if provided a Pandas.Series, we expect index are sequence
        # headers and single column holds sequences.
        asvs = counts.values()
        seqs = pd.Series(index=asvs, data=asvs).sort_index()
    elif isinstance(counts, Table):
        seqs = pd.Series(index=counts.ids('observation'), data=counts.ids('observation')).sort_index()
    else:
        raise ValueError("unexpected data type for counts!")
    args = {'seqs': seqs,
            'reference_database': reference_database,
            #'reference_phylogeny': reference_phylogeny,
            #'reference_taxonomy': reference_taxonomy,
            #'reference_info': reference_info,
            'alignment_subset_size': alignment_subset_size,
            'placement_subset_size': placement_subset_size,
            'chunksize': chunksize}
    return _executor('sepp',
                     args,
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=ppn, pmem=pmem, walltime=walltime,
                     array=len(range(0, len(seqs), chunksize)),
                     environment=environment,
                     **executor_args)


def sepp_old(counts, chunksize=10000, reference=None, stopdecomposition=None,
             ppn=20, pmem='50GB', walltime='12:00:00',
             **executor_args):
    """Tip insertion of deblur sequences into GreenGenes backbone tree.

    Parameters
    ----------
    counts : Pandas.DataFrame | Pandas.Series
        a) OTU counts in form of a Pandas.DataFrame.
        b) If providing a Pandas.Series, we expect the index to be a fasta
           headers and the colum the fasta sequences.
    reference : str
        Default: None.
        Valid values are ['pynast']. Use a different alignment file for SEPP.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose
    chunksize: int
        Default: 30000
        SEPP jobs seem to fail if too many sequences are submitted per job.
        Therefore, we can split your sequences in chunks of chunksize.

    Returns
    -------
    ???"""
    def pre_execute(workdir, args):
        chunks = range(0, seqs.shape[0], args['chunksize'])
        for chunk, i in enumerate(chunks):
            # write all deblur sequences into one file per chunk
            file_fragments = workdir + '/sequences%s.mfa' % (chunk + 1)
            f = open(file_fragments, 'w')
            chunk_seqs = seqs.iloc[i:i + args['chunksize']]
            for header, sequence in chunk_seqs.items():
                f.write('>%s\n%s\n' % (header, sequence))
            f.close()

    def commands(workdir, ppn, args):
        commands = []
        commands.append('cd %s' % workdir)
        ref = ''
        if args['reference'] is not None:
            ref = ' -r %s' % args['reference']
        sdcomp = ''
        if 'stopdecomposition' in args:
            sdcomp = ' -M %f ' % args['stopdecomposition']
        fp_sepp = '$CONDA_PREFIX/'
        commands.append('%sbin/run-sepp.sh "%s/sequences%s.mfa" res${%s} -x %i %s %s -r %sshare/sepp/ref/RAxML_info-reference-gg-raxml-bl.info -b 1' % (
            fp_sepp,
            workdir,
            '${%s}' % settings.VARNAME_PBSARRAY if len(range(0, seqs.shape[0], chunksize)) > 1 else '1',
            settings.VARNAME_PBSARRAY,
            ppn,
            ref,
            sdcomp,
            fp_sepp))
        return commands

    def post_execute(workdir, args):
        files_placement = sorted(
            [workdir + '/' + file_placement
             for file_placement in next(os.walk(workdir))[2]
             if file_placement.endswith('_placement.json')])
        if len(files_placement) > 1:
            file_mergedplacements = workdir + '/merged_placements.json'
            if not os.path.exists(file_mergedplacements):
                sys.stderr.write("step 1) merging placement files: ")
                fout = open(file_mergedplacements, 'w')
                for i, file_placement in enumerate(files_placement):
                    sys.stderr.write('.')
                    fin = open(file_placement, 'r')
                    write = i == 0
                    for line in fin.readlines():
                        if '"placements": [{' in line:
                            write = True
                            if i != 0:
                                continue
                        if '}],' in line:
                            write = i+1 == len(files_placement)
                        if write is True:
                            fout.write(line)
                    fin.close()
                    if i+1 != len(files_placement):
                        fout.write('    },\n')
                        fout.write('    {\n')
                fout.close()
                sys.stderr.write(' done.\n')

            sys.stderr.write("step 2) placing fragments into tree: ...")
            # guppy ran for: and consumed 45 GB of memory for 2M, chunked 10k
            # sepp benchmark:
            # real	37m39.772s
            # user	31m3.906s
            # sys	3m49.602s
            file_merged_tree = file_mergedplacements[:-5] +\
                '.tog.relabelled.tre'
            cluster_run([
                'cd %s' % workdir,
                '$CONDA_PREFIX/bin/guppy tog %s' %
                file_mergedplacements,
                'cat %s | python %s > %s' % (
                    file_mergedplacements.replace('.json', '.tog.tre'),
                    files_placement[0].replace('placement.json',
                                               'rename-json.py'),
                    file_merged_tree)],
                environment='sepp',
                jobname='guppy_rename',
                result=file_merged_tree,
                ppn=1, pmem='100GB', walltime='1:00:00', dry=False,
                wait=True)
            sys.stderr.write(' done.\n')
        else:
            file_merged_tree = files_placement[0].replace(
                '.json', '.tog.relabelled.tre')

        sys.stderr.write("step 3) reading skbio tree: ...")
        tree = TreeNode.read(file_merged_tree, format='newick')
        sys.stderr.write(' done.\n')

        sys.stderr.write("step 4) use the phylogeny to det"
                         "ermine tips lineage: ")
        lineages = []
        features = []
        divisor = int(tree.count(tips=True) / min(10, tree.count(tips=True)))
        for i, tip in enumerate(tree.tips()):
            if i % divisor == 0:
                sys.stderr.write('.')
            if tip.name.isdigit():
                continue

            lineage = []
            for ancestor in tip.ancestors():
                try:
                    float(ancestor.name)
                except TypeError:
                    pass
                except ValueError:
                    lineage.append(ancestor.name)

            lineages.append("; ".join(reversed(lineage)))
            features.append(tip.name)
        sys.stderr.write(' done.\n')

        # storing tree as newick string is necessary since large trees would
        # result in too many recursions for the python heap :-/
        newick = StringIO()
        tree.write(newick)
        return {'taxonomy': pd.DataFrame(data=lineages,
                                         index=features,
                                         columns=['taxonomy']),
                'tree': newick.getvalue(),
                'reference': args['reference']}

    inp = sorted(counts.index)
    if type(counts) == pd.Series:
        # typically, the input is an OTU table with index holding sequences.
        # However, if provided a Pandas.Series, we expect index are sequence
        # headers and single column holds sequences.
        inp = counts.sort_index()

    def post_cache(cache_results):
        newicks = []
        for tree in cache_results['trees']:
            newicks.append(TreeNode.read(StringIO(tree), format='newick'))
        cache_results['trees'] = newicks
        return cache_results

    seqs = inp
    if type(inp) != pd.Series:
        seqs = pd.Series(inp, index=inp).sort_index()
    args = {'seqs': seqs,
            'reference': reference,
            'chunksize': chunksize}
    if stopdecomposition is not None:
        args['stopdecomposition'] = stopdecomposition
    return _executor('sepp',
                     args,
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=ppn, pmem=pmem, walltime=walltime,
                     array=len(range(0, seqs.shape[0], chunksize)),
                     **executor_args)


def sepp_stepbystep(counts, reference=None,
                    stopdecomposition=None,
                    ppn=20, pmem='8GB', walltime='12:00:00',
                    **executor_args):
    """Step by Step version of SEPP to track memory consumption more closely.
       Tip insertion of deblur sequences into GreenGenes backbone tree.

    Parameters
    ----------
    counts : Pandas.DataFrame | Pandas.Series
        a) OTU counts in form of a Pandas.DataFrame.
        b) If providing a Pandas.Series, we expect the index to be a fasta
           headers and the colum the fasta sequences.
    reference : str
        Default: None.
        Valid values are ['pynast']. Use a different alignment file for SEPP.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose
    chunksize: int
        Default: 30000
        SEPP jobs seem to fail if too many sequences are submitted per job.
        Therefore, we can split your sequences in chunks of chunksize.

    Returns
    -------
    ???"""
    def pre_execute(workdir, args):
        file_fragments = workdir + '/sequences.mfa'
        f = open(file_fragments, 'w')
        for header, sequence in seqs.items():
            f.write('>%s\n%s\n' % (header, sequence))
        f.close()
        os.makedirs(workdir + '/sepp-tempssd/', exist_ok=True)

    def commands(workdir, ppn, args):
        commands = []
        name = 'seppstepbysteprun'
        dir_base = ('/home/sjanssen/miniconda3/envs/seppGG_py3/'
                    'src/sepp-package/')
        dir_tmp = workdir + '/sepp-tempssd/'

        commands.append('cd %s' % workdir)

        commands.append(
            ('python %s -P %i -A %s -t %s -a %s -r %s -f %s -cp '
             '%s/chpoint-%s -o %s -d %s -p %s '
             '1>>%s/sepp-%s-out.log 2>%s/sepp-%s-err.log') % (
                ('%ssepp/run_sepp.py' % dir_base),  # python script of SEPP
                5000,  # problem size for tree
                1000,  # problem size for alignment
                # reference tree file
                ('%sref/reference-gg-raxml-bl-rooted-relabelled.tre' %
                    dir_base),
                # reference alignment file
                ('%sref/gg_13_5_ssu_align_99_pfiltered.fasta' % dir_base),
                # reference info file
                ('%sref/RAxML_info-reference-gg-raxml-bl.info' % dir_base),
                workdir + '/sequences.mfa',  # sequence input file
                dir_tmp,  # tmpdir
                name,
                name,
                workdir,
                dir_tmp,
                workdir,
                name,
                workdir,
                name))

        commands.append(('%s/sepp/tools/bundled/Linux/guppy-64 tog %s/%s_plac'
                         'ement.json') % (dir_base, workdir, name))
        commands.append(('python %s/%s_rename-json.py < %s/%s_placement.tog.t'
                         're > %s/%s_placement.tog.relabelled.tre') %
                        (workdir, name, workdir, name, workdir, name))
        commands.append(('%s/sepp/tools/bundled/Linux/guppy-64 tog --xml %s/%'
                         's_placement.json') % (dir_base, workdir, name))
        commands.append(('python %s/%s_rename-json.py < %s/%s_placement.tog.x'
                         'ml > %s/%s_placement.tog.relabelled.xml') %
                        (workdir, name, workdir, name, workdir, name))

        return commands

    def post_execute(workdir, args):
        file_merged_tree = workdir +\
            '/seppstepbysteprun_placement.tog.relabelled.tre'
        sys.stderr.write("step 1/2) reading skbio tree: ...")
        tree = TreeNode.read(file_merged_tree, format='newick')
        sys.stderr.write(' done.\n')

        sys.stderr.write("step 2/2) use the phylogeny to det"
                         "ermine tips lineage: ")
        lineages = []
        features = []
        divisor = int(tree.count(tips=True) / min(10, tree.count(tips=True)))
        for i, tip in enumerate(tree.tips()):
            if i % divisor == 0:
                sys.stderr.write('.')
            if tip.name.isdigit():
                continue

            lineage = []
            for ancestor in tip.ancestors():
                try:
                    float(ancestor.name)
                except TypeError:
                    pass
                except ValueError:
                    lineage.append(ancestor.name)

            lineages.append("; ".join(reversed(lineage)))
            features.append(tip.name)
        sys.stderr.write(' done.\n')

        # storing tree as newick string is necessary since large trees would
        # result in too many recursions for the python heap :-/
        newick = StringIO()
        tree.write(newick)
        return {'taxonomy': pd.DataFrame(data=lineages,
                                         index=features,
                                         columns=['taxonomy']),
                'tree': newick.getvalue(),
                'reference': args['reference']}

    inp = sorted(counts.index)
    if type(counts) == pd.Series:
        # typically, the input is an OTU table with index holding sequences.
        # However, if provided a Pandas.Series, we expect index are sequence
        # headers and single column holds sequences.
        inp = counts.sort_index()

    seqs = inp
    if type(inp) != pd.Series:
        seqs = pd.Series(inp, index=inp).sort_index()
    return _executor('seppstep',
                     {'seqs': seqs,
                      'reference': reference},
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=ppn, pmem=pmem, walltime=walltime,
                     **executor_args)


def sepp_git(counts,
             ppn=20, pmem='8GB', walltime='12:00:00',
             **executor_args):
    """Latest git version of SEPP.
       Tip insertion of deblur sequences into GreenGenes backbone tree.

    Parameters
    ----------
    counts : Pandas.DataFrame | Pandas.Series
        a) OTU counts in form of a Pandas.DataFrame.
        b) If providing a Pandas.Series, we expect the index to be a fasta
           headers and the colum the fasta sequences.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    ???"""
    def pre_execute(workdir, args):
        file_fragments = workdir + '/sequences.mfa'
        f = open(file_fragments, 'w')
        for header, sequence in seqs.iter():
            f.write('>%s\n%s\n' % (header, sequence))
        f.close()
        os.makedirs(workdir + '/sepp-tempssd/', exist_ok=True)

    def commands(workdir, ppn, args):
        commands = []
        commands.append('cd %s' % workdir)
        commands.append('%srun-sepp.sh "%s" res -x %i' % (
            ('/home/sjanssen/Benchmark_insertiontree/'
             'Software/sepp/sepp-package/'),
            workdir+'/sequences.mfa',
            ppn))
        return commands

    def post_execute(workdir, args):
        file_merged_tree = workdir +\
            '/res_placement.tog.relabelled.tre'
        sys.stderr.write("step 1/2) reading skbio tree: ...")
        tree = TreeNode.read(file_merged_tree, format='newick')
        sys.stderr.write(' done.\n')

        sys.stderr.write("step 2/2) use the phylogeny to det"
                         "ermine tips lineage: ")
        lineages = []
        features = []
        divisor = int(tree.count(tips=True) / min(10, tree.count(tips=True)))
        for i, tip in enumerate(tree.tips()):
            if i % divisor == 0:
                sys.stderr.write('.')
            if tip.name.isdigit():
                continue

            lineage = []
            for ancestor in tip.ancestors():
                try:
                    float(ancestor.name)
                except TypeError:
                    pass
                except ValueError:
                    lineage.append(ancestor.name)

            lineages.append("; ".join(reversed(lineage)))
            features.append(tip.name)
        sys.stderr.write(' done.\n')

        # storing tree as newick string is necessary since large trees would
        # result in too many recursions for the python heap :-/
        newick = StringIO()
        tree.write(newick)
        return {'taxonomy': pd.DataFrame(data=lineages,
                                         index=features,
                                         columns=['taxonomy']),
                'tree': newick.getvalue()}

    inp = sorted(counts.index)
    if type(counts) == pd.Series:
        # typically, the input is an OTU table with index holding sequences.
        # However, if provided a Pandas.Series, we expect index are sequence
        # headers and single column holds sequences.
        inp = counts.sort_index()

    seqs = inp
    if type(inp) != pd.Series:
        seqs = pd.Series(inp, index=inp).sort_index()
    return _executor('seppgit',
                     {'seqs': seqs},
                     pre_execute,
                     commands,
                     post_execute,
                     environment='sepp_git',
                     ppn=ppn, pmem=pmem, walltime=walltime,
                     **executor_args)


def sortmerna(sequences,
              reference='/projects/emp/03-otus/reference/97_otus.fasta',
              sortmerna_db='/projects/emp/03-otus/reference/97_otus.idx',
              ppn=5, pmem='20GB', walltime='2:00:00', **executor_args):
    """Assigns closed ref GreenGenes OTUids to sequences.

    Parameters
    ----------
    sequences : Pd.Series
        Set of sequences with header as index and nucleotide sequences as
        values.
    reference : filename
        Default: /projects/emp/03-otus/reference/97_otus.fasta
        Multiple fasta collection that serves as reference for sortmerna
        homology searches.
    sortmerna_db : filename
        Default: /projects/emp/03-otus/reference/97_otus.idx
        Can point to a precompiled reference DB. Make sure it matches your
        reference collection! Saves ~25min compute.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    """
    def pre_execute(workdir, args):
        # store all unique sequences to a fasta file
        file_fragments = workdir + '/sequences.mfa'
        file_mapping = workdir + '/headermap.tsv'
        f = open(file_fragments, 'w')
        m = open(file_mapping, 'w')
        for i, (header, sequence) in enumerate(args['seqs'].iteritems()):
            f.write('>%s\n%s\n' % ('seq_%i' % i, sequence))
            m.write('seq_%i\t%s\n' % (i, header))
        f.close()
        m.close()

    def commands(workdir, ppn, args):
        commands = []
        precompileddb = ''
        if args['sortmerna_db'] is not None:
            precompileddb = ' --sortmerna_db %s ' % args['sortmerna_db']
        commands.append(('pick_otus.py '
                         '-m sortmerna '
                         '-i %s '
                         '-r %s '
                         '%s'
                         '-o %s '
                         '--sortmerna_e_value 1 '
                         '-s 0.97 '
                         '--threads %i '
                         '--suppress_new_clusters ') % (
            workdir + '/sequences.mfa',
            args['reference'],
            precompileddb,
            workdir + '/sortmerna/',
            ppn))
        return commands

    def post_execute(workdir, args):
        assignments = []

        # parse header mapping file
        hmap = pd.read_csv(workdir + '/headermap.tsv', sep='\t', header=None,
                           index_col=0)
        # parse sucessful sequence to OTU assignments
        f = open(workdir+'/sortmerna/sequences_otus.txt', 'r')
        for line in f.readlines():
            parts = line.rstrip().split('\t')
            for header in parts[1:]:
                assignments.append({'otuid': parts[0],
                                    'header': hmap.loc[header].iloc[0]})
        f.close()

        # parse failed sequences
        f = open(workdir+'/sortmerna/sequences_failures.txt', 'r')
        for line in f.readlines():
            assignments.append({'header': hmap.loc[line.rstrip()].iloc[0]})
        f.close()

        return pd.DataFrame(assignments).set_index('header')

    if not os.path.exists(reference):
        raise ValueError('Reference multiple fasta file "%s" does not exist!' %
                         reference)

    if sortmerna_db is not None:
        if not os.path.exists(sortmerna_db+'.stats'):
            sys.stderr.write('Could not find SortMeRNA precompiled DB. '
                             'I continue by creating a new DB.')
    # core dump with 8GB with 10 nodes, 4h
    # trying 20GB with 10 nodes ..., 4h (long wait for scheduler)
    # trying 20GB with 5 nodes, 2h ...
    return _executor('sortmerna',
                     {'seqs': sequences.drop_duplicates().sort_index(),
                      'reference': reference,
                      'sortmerna_db': sortmerna_db},
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=ppn,
                     pmem=pmem,
                     walltime=walltime,
                     **executor_args)


def denovo_tree(counts, ppn=1, **executor_args):
    """Builds a de novo tree for given sequences using PyNAST + fasttree.

    Parameters
    ----------
    counts : Pandas.DataFrame | Pandas.Series
        a) OTU counts in form of a Pandas.DataFrame.
        b) If providing a Pandas.Series, we expect the index to be a fasta
           headers and the colum the fasta sequences.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    A newick string of the created phylogenetic tree."""
    def pre_execute(workdir, args):
        # store all unique sequences to a fasta file
        # as in sortmerna we need a header map, because fasttree will otherwise
        # throw strange errors
        file_fragments = workdir + '/sequences.mfa'
        file_mapping = workdir + '/headermap.tsv'
        f = open(file_fragments, 'w')
        m = open(file_mapping, 'w')
        for i, (header, sequence) in enumerate(args['seqs'].iteritems()):
            f.write('>%s\n%s\n' % ('seq%i' % i, sequence))
            m.write('seq%i\t%s\n' % (i, header))
        f.close()
        m.close()

    def commands(workdir, ppn, args):
        commands = []

        commands.append('parallel_align_seqs_pynast.py -O %i -i %s -o %s' % (
            ppn,
            workdir+'/sequences.mfa',
            workdir))
        commands.append('fasttree -nt %s > %s' % (
            workdir+'/sequences_aligned.fasta',
            workdir+'/tree.newick'))

        return commands

    def post_execute(workdir, args):
        # load resulting tree
        f = open(workdir+'/tree.newick', 'r')
        tree = "".join(f.readlines())
        f.close()

        # parse header mapping file and rename sequence identifier
        hmap = pd.read_csv(workdir + '/headermap.tsv', sep='\t', header=None,
                           index_col=0)[1]
        return {'tree': tree,
                'hmap': hmap}

    def post_cache(cache_results):
        hmap = cache_results['results']['hmap']
        tree = TreeNode.read(StringIO(cache_results['results']['tree']), format='newick')
        for node in tree.tips():
            node.name = hmap.loc[node.name]

        cache_results['results']['tree'] = tree
        del cache_results['results']['hmap']
        return cache_results

    inp = sorted(counts.index)
    if type(counts) == pd.Series:
        # typically, the input is an OTU table with index holding sequences.
        # However, if provided a Pandas.Series, we expect index are sequence
        # headers and single column holds sequences.
        inp = counts.sort_index()

    seqs = inp
    if type(inp) != pd.Series:
        seqs = pd.Series(inp, index=inp).sort_index()

    return _executor('pynastfasttree',
                     {'seqs': seqs},
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache=post_cache,
                     ppn=ppn,
                     **executor_args)


def denovo_tree_qiime2(counts, **executor_args):
    """Builds a de novo tree for given sequences using mafft + fasttree
       following the Qiime2 tutorial https://docs.qiime2.org/2017.9/tutorials
       /moving-pictures/#generate-a-tree-for-phylogenetic-diversity-analyses

    Parameters
    ----------
    counts : Pandas.DataFrame | Pandas.Series
        a) OTU counts in form of a Pandas.DataFrame.
        b) If providing a Pandas.Series, we expect the index to be a fasta
           headers and the colum the fasta sequences.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    {'tree': skbio.tree.TreeNode
        skbio TreeNode object.
    }
    """
    def pre_execute(workdir, args):
        # store all unique sequences to a fasta file
        # as in sortmerna we need a header map, because fasttree will otherwise
        # throw strange errors
        file_fragments = workdir + '/sequences.mfa'
        file_mapping = workdir + '/headermap.tsv'
        f = open(file_fragments, 'w')
        m = open(file_mapping, 'w')
        for i, (header, sequence) in enumerate(args['seqs'].iteritems()):
            f.write('>%s\n%s\n' % ('seq%i' % i, sequence.upper()))
            m.write('seq%i\t%s\n' % (i, header))
        f.close()
        m.close()

    def commands(workdir, ppn, args):
        commands = []

        # import fasta sequences into qza
        commands.append(
            ('qiime tools import '
             '--input-path %s '
             '--output-path %s/rep-seqs '
             '--type "FeatureData[Sequence]"') %
            (workdir + '/sequences.mfa', workdir))
        # First, we perform a multiple sequence alignment of the sequences in
        # our FeatureData[Sequence] to create a FeatureData[AlignedSequence]
        # QIIME 2 artifact. Here we do this with the mafft program.
        commands.append(
            ('qiime alignment mafft '
             '--i-sequences %s/rep-seqs.qza '
             '--o-alignment %s/aligned-rep-seqs.qza '
             '--p-n-threads %i') %
            (workdir, workdir, ppn))
        # Next, we mask (or filter) the alignment to remove positions that are
        # highly variable. These positions are generally considered to add
        # noise to a resulting phylogenetic tree.
        commands.append(
            ('qiime alignment mask '
             '--i-alignment %s/aligned-rep-seqs.qza '
             '--o-masked-alignment %s/masked-aligned-rep-seqs.qza') %
            (workdir, workdir))
        # Next, we’ll apply FastTree to generate a phylogenetic tree from the
        # masked alignment.
        commands.append(
            ('qiime phylogeny fasttree '
             '--i-alignment %s/masked-aligned-rep-seqs.qza '
             '--o-tree %s/unrooted-tree.qza '
             '--p-n-threads %i') %
            (workdir, workdir, ppn))
        # The FastTree program creates an unrooted tree, so in the final step
        # in this section we apply midpoint rooting to place the root of the
        # tree at the midpoint of the longest tip-to-tip distance in the
        # unrooted tree.
        commands.append(
            ('qiime phylogeny midpoint-root '
             '--i-tree %s/unrooted-tree.qza '
             '--o-rooted-tree %s/rooted-tree.qza ') %
            (workdir, workdir))

        # export the phylogeny
        commands.append(
            ('qiime tools export '
             '--input-path %s/rooted-tree.qza '
             '--output-path %s') %
            (workdir, workdir))

        return commands

    def post_execute(workdir, args):
        # load resulting tree
        f = open(workdir+'/tree.nwk', 'r')
        tree = "".join(f.readlines())
        f.close()

        # parse header mapping file and rename sequence identifier
        hmap = pd.read_csv(workdir + '/headermap.tsv', sep='\t', header=None,
                           index_col=0)[1]
        return {'tree': tree,
                'hmap': hmap}

    def post_cache(cache_results):
        hmap = cache_results['results']['hmap']
        tree = TreeNode.read(StringIO(cache_results['results']['tree']), format='newick')
        for node in tree.tips():
            node.name = hmap.loc[node.name]

        cache_results['results']['tree'] = tree
        del cache_results['results']['hmap']
        return cache_results

    inp = sorted(counts.index)
    if type(counts) == pd.Series:
        # typically, the input is an OTU table with index holding sequences.
        # However, if provided a Pandas.Series, we expect index are sequence
        # headers and single column holds sequences.
        inp = counts.sort_index()

    seqs = inp
    if type(inp) != pd.Series:
        seqs = pd.Series(inp, index=inp).sort_index()

    return _executor('qiime2denovo',
                     {'seqs': seqs},
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache=post_cache,
                     environment=settings.QIIME2_ENV,
                     **executor_args)


def _parse_cmpcat_table(lines, fieldname):
    header = lines[0].split()
    columns = lines[1].replace(' < ', '  ').split()
    columns[0] = fieldname
    residuals = lines[2].split()
    residuals[0] = fieldname  # + '_residuals'
    res = []
    for (_type, line) in zip(['field', 'residuals'], [columns, residuals]):
        r = dict()
        r['type'] = _type
        i = 0
        for name in ['field'] + header:
            if _type == 'residuals':
                if name in ['F.Model', 'Pr(>F)', 'F_value', 'F', 'N.Perm']:
                    r[name] = np.nan
                    continue
            r[name] = line[i]
            i += 1
        res.append(r)
    return pd.DataFrame(res).loc[:, ['field', 'type'] + header]


def _parse_adonis(filename, fieldname='unnamed'):
    """Parse the R result of an adonis test.

    Parameters
    ----------
    filename : str
        Filepath to R adonis output.
    fieldname: str
        Name for the field that has been tested.

    Returns
    -------
        Pandas.DataFrame holding adonis results.
        Two rows: first is the tested field, second the residuals."""
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    res = _parse_cmpcat_table(lines[9:12], fieldname)
    res['method'] = 'adonis'

    return res


def _parse_permdisp(filename, fieldname='unnamed'):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    # fix header names, i.e. remove white spaces for later splitting
    for i in [3, 12]:
        lines[i] = lines[i]\
            .replace('Sum Sq', 'Sum_Sq')\
            .replace('Mean Sq', 'Mean_Sq')\
            .replace('F value', 'F_value')
    upper = _parse_cmpcat_table(lines[3:6], fieldname)
    upper['method'] = 'permdisp'
    upper['kind'] = 'observed'
    lower = _parse_cmpcat_table(lines[12:15], fieldname)
    lower['method'] = 'permdisp'
    lower['kind'] = 'permuted'
    lower = lower.rename(columns={'F': 'F_value'})

    return pd.concat([upper, lower])


def _parse_permanova(filename, fieldname='unnamed'):
    res = pd.read_csv(filename, sep='\t', header=None).T
    res.columns = res.iloc[0, :]
    res = res.iloc[1:, :]
    del res['method name']
    res['method'] = 'permanova'
    res['field'] = fieldname
    return res


def compare_categories(beta_dm, metadata,
                       methods=['adonis', 'permanova', 'permdisp'],
                       num_permutations=999, **executor_args):
    """Tests for significance of a metadata field regarding beta diversity.

    Parameters
    ----------
    beta_dm : skbio.stats.distance._base.DistanceMatrix
        The beta diversity distance matrix for the samples
    metadata : pandas.DataFrame
        Metadata columns to be checked for variation.
    methods : [str]
        Default: ['adonis', 'permanova', 'permdisp'].
        Choose from ['adonis', 'permanova', 'permdisp'].
        The statistical test that should be applied.
    num_permutations : int
        Number of permutations to use for permanova test.

    Returns
    -------
    """
    def pre_execute(workdir, args):
        dm = args['beta_dm']
        meta = args['metadata']
        # only use samples present in both:
        # the distance metrix and the metadata
        idx = set(dm.ids) & set(meta.index)
        # make sure both objects have the same sorting of samples
        dm = dm.filter(idx, strict=False)
        meta = meta.loc[idx, :]

        dm.write(workdir + '/beta_distances.txt')
        meta.to_csv(workdir + '/meta.tsv',
                    sep="\t", index_label="#SampleID", header=True)
        f = open(workdir + '/fields.txt', 'w')
        f.write("\n".join(meta.columns)+"\n")
        f.close()

    def commands(workdir, ppn, args):
        commands = []

        commands.append('module load %s' % settings.R_MODULE)
        commands.append('cd %s' % workdir)
        for method in args['methods']:
            commands.append(
                ('compare_categories.py --method %s '
                 '-i %s/beta_distances.txt '
                 '-m %s/meta.tsv '
                 '-c `cat fields.txt | head -n ${%s} '
                 '| tail -n 1` '
                 '-o %s/res%s_`cat fields.txt | head -n ${%s} '
                 '| tail -n 1`/ '
                 '-n %i') %
                (method, workdir, workdir, settings.VARNAME_PBSARRAY, workdir, method, settings.VARNAME_PBSARRAY, num_permutations))

        return commands

    def post_execute(workdir, args):
        merged = dict()

        ms = zip(['adonis', 'permdisp', 'permanova'],
                 [_parse_adonis, _parse_permdisp, _parse_permanova])
        for (name, method) in list(ms):
            merged[name] = []
            for field in args['metadata'].columns:
                filename_result = '%s/res%s_%s/%s_results.txt' % (
                    workdir, name, field, name)
                if os.path.exists(filename_result):
                    merged[name].append(method(filename_result, field))
            if len(merged[name]) > 0:
                merged[name] = pd.concat(merged[name])
        return merged

    if type(metadata) == pd.core.series.Series:
        metadata = metadata.to_frame()

    return _executor('cmpcat',
                     {'beta_dm': beta_dm,
                      'metadata': metadata,
                      'num_permutations': num_permutations,
                      'methods': sorted(methods)},
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=1,
                     array=len(range(0, metadata.shape[1])),
                     **executor_args)


def picrust(counts, **executor_args):
    """Translate closed ref OTU tables into predicted meta-transcriptomics.

    Parameters
    ----------
    counts : Pandas.DataFrame
        OTU counts
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    One hash with following 8 tables:
    {'rfam': [c],
     'KEGG_Pathways': [c, c, c, c],
     'COG_Category': [c, c, c]}

    Install
    -------
    outdated, better use ggmap_picrust1 env from anaconda/sjanssen2 !

    conda create --name picrust python=2.7
    conda activate picrust
    mkdir $CONDA_PREFIX/src/
    cd $CONDA_PREFIX/src/
    wget https://github.com/picrust/picrust/releases/down \
        load/v1.1.3/picrust-1.1.3.tar.gz
    tar xzvf picrust-1.1.3.tar.gz
    pip install numpy
    pip install h5py
    pip install .
    download_picrust_files.py -t ko
    download_picrust_files.py -t cog
    download_picrust_files.py -t rfam
    """
    TYPES = ['rfam', 'ko', 'cog']

    def get_catname_levels(type_):
        levels = None
        category = None
        if type_ == 'ko':
            category = 'KEGG_Pathways'
            levels = list(range(1, 4))
        elif type_ == 'cog':
            category = 'COG_Category'
            levels = list(range(1, 3))
        elif type_ == 'rfam':
            category = 'Rfam'
            levels = []
        return category, levels

    def pre_execute(workdir, args):
        if all(map(lambda x: x.isdigit(), args['counts'].index)):
            pandas2biom(workdir+'/input.biom', args['counts'].fillna(0.0))
        else:
            raise ValueError(
                ('Not all features are numerical, that might point to the fact'
                 ' that your count table does not stem from closed reference '
                 'picking vs. Greengenes.'))

    def commands(workdir, ppn, args):
        commands = []

        # Normalize to copy number
        commands.append(('normalize_by_copy_number.py '
                         '-i "%s/input.biom" '
                         '-o "%s/normalized.biom"') %
                        (workdir, workdir))

        # PICRUSt prediction
        for type_ in TYPES:
            commands.append(('predict_metagenomes.py '
                             '-t %s '
                             '-i "%s/normalized.biom" '
                             '-o "%s/%s"') % (
                    type_, workdir, workdir, 'picrust_%s_coll-0.biom' % type_))

        # Collapse at higher hierarchy levels
        for type_ in TYPES:
            category, levels = get_catname_levels(type_)
            for level in levels:
                commands.append((
                    'categorize_by_function.py '
                    '-i "%s/picrust_%s_coll-0.biom" '
                    '-o "%s/picrust_%s_coll-%s.biom" -l %i -c %s') %
                    (workdir, type_, workdir, type_, level, level, category))

        return commands

    def post_execute(workdir, args):
        results = dict()
        for type_ in TYPES:
            category, levels = get_catname_levels(type_)
            results[category] = dict()
            for level in [0] + levels:
                results[category][level] = biom2pandas(
                    '%s/picrust_%s_coll-%s.biom' % (workdir, type_, level))
        return results

    return _executor('picrust',
                     {'counts': counts.fillna(0.0)},
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=1,
                     environment=settings.PICRUST_ENV,
                     **executor_args)


def picrust2(counts, **executor_args):
    """Translate ASV tables into predicted meta-transcriptomics.

    Parameters
    ----------
    counts : Pandas.DataFrame
        feature-table
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    One hash with following keys:
    {'counts': {'COG': pd.DataFrame,
                'EC': pd.DataFrame,
                'KO': pd.DataFrame,
                'PFAM': pd.DataFrame,
                'TIGRFAM': pd.DataFrame,
                'PHENO': pd.DataFrame,
                'MetaCyc_pathways': pd.DataFrame},
     'taxonomy': {'COG': pd.Series,
                  'EC': pd.Series,
                  'KO': pd.Series,
                  'PFAM': pd.Series,
                  'TIGRFAM': pd.Series,
                  'PHENO': None,
                  'MetaCyc_pathways: pd.Series'},
     'NSTI': {'features': pd.Series,
              'samples': pd.Series}
    }
    where 'counts' are the translated metagenomic counts per sample
          'taxonomy' are vectors describing features more elaborately and features may collapse to sample "taxonomy" description
          'NSTI' = "Nearest Sequenced Taxon Index": This index reflects the average phylogenetic distance between each 16S rRNA gene sequence in their sample
    """
    # determine if features are ASVs or OTU-IDs
    IS_ASV = all(map(lambda x: re.sub("[ACGT]+", "", x.upper()) == "", counts.index))

    TYPES = ['16S','COG','EC','KO','PFAM','TIGRFAM','PHENO']

    def pre_execute(workdir, args):
        if IS_ASV:
            with open('%s/seqs.fna' % workdir, 'w') as f:
                for seq in args['counts'].index:
                    f.write('>%s\n%s\n' % (seq, seq))
        else:
            raise ValueError("picrust2 in ggmap does currently not support OTU tables.")

        pandas2biom('%s/table.biom' % workdir, counts)

    def commands(workdir, ppn, args):
        commands = []

        # Place reads into reference tree
        commands.append((
            'place_seqs.py '
            '-s %s/seqs.fna -o %s/out.tre -p %i '
            '--intermediate %s/intermediate/place_seqs') %
            (workdir, workdir, ppn, workdir))

        # Hidden-state prediction of gene families
        for _type in TYPES:
            # NSTI computation leads to identical results per _type.
            # Thus, we only do it for 16S.
            withNSTI = _type == '16S'
            commands.append((
                'hsp.py '
                '-i %s '
                '-t %s/out.tre '
                '-o %s/%s_predicted%s.tsv.gz '
                '-p %i '
                '%s') %
                (_type, workdir, workdir, _type, '_and_nsti' if withNSTI else '', ppn, '-n' if withNSTI else ''))
        # commands.append("cp /tmp/ana_picrust2/* -r %s" % workdir)

        # Generate metagenome predictions
        for _type in TYPES:
            if _type == '16S':
                continue
            commands.append((
                'metagenome_pipeline.py '
                '-i %s/table.biom '
                '-m %s/16S_predicted_and_nsti.tsv.gz '
                '-f %s/%s_predicted.tsv.gz '
                '-o %s/%s_metagenome_out ') %
                (workdir, workdir, workdir, _type, workdir, _type))

        # Pathway-level inference
        commands.append((
            'pathway_pipeline.py '
            '-i %s/EC_metagenome_out/pred_metagenome_unstrat.tsv.gz '
            '-o %s/pathways_out '
            '-p %i') %
            (workdir, workdir, ppn))

        # Add functional descriptions
        for _type in TYPES:
            if _type in ['16S', 'PHENO']:
                continue
            commands.append((
                'add_descriptions.py '
                '-i %s/%s_metagenome_out/pred_metagenome_unstrat.tsv.gz '
                '-m %s '
                '-o %s/%s_metagenome_out/pred_metagenome_unstrat_descrip.tsv.gz') %
                (workdir, _type, _type, workdir, _type))
        commands.append((
            'add_descriptions.py '
            '-i %s/pathways_out/path_abun_unstrat.tsv.gz '
            '-m METACYC '
            '-o %s/pathways_out/path_abun_unstrat_descrip.tsv.gz') %
            (workdir, workdir))

        return commands

    def post_execute(workdir, args):
        results = dict()

        results['counts'] = dict()
        for _type in TYPES:
            if _type in ['16S']:
                continue
            if _type == 'PHENO':
                results['counts'][_type] = pd.read_csv('%s/%s_metagenome_out/pred_metagenome_unstrat.tsv.gz' % (workdir, _type), compression='gzip', sep='\t', index_col=0)
            else:
                results['counts'][_type] = pd.read_csv('%s/%s_metagenome_out/pred_metagenome_unstrat_descrip.tsv.gz' % (workdir, _type), compression='gzip', sep='\t', index_col=0)
        results['counts']['MetaCyc_pathways'] = pd.read_csv('%s/pathways_out/path_abun_unstrat_descrip.tsv.gz' % workdir, compression='gzip', sep='\t', index_col=0)

        # split count table into real feature counts AND taxonomy series
        results['taxonomy'] = dict()
        for _type in results['counts'].keys():
            if 'description' in results['counts'][_type].columns:
                results['taxonomy'][_type] = results['counts'][_type]['description']
                del results['counts'][_type]['description']
            else:
                results['taxonomy'][_type] = None

        results['NSTI'] = {
            'features': pd.read_csv('%s/16S_predicted_and_nsti.tsv.gz' % workdir, compression='gzip', sep='\t', index_col=0, squeeze=True)['metadata_NSTI'],
            # all types return same NSTI samples values, thus we here arbitratily pick one
            'samples': pd.read_csv('%s/%s_metagenome_out/weighted_nsti.tsv.gz' % (workdir, 'COG'), compression='gzip', sep='\t', index_col=0, squeeze=True)}

        return results

    return _executor('picrust2',
                     {'counts': counts.fillna(0.0)},
                     pre_execute,
                     commands,
                     post_execute,
                     environment=settings.PICRUST2_ENV,
                     **executor_args)


def bugbase(counts, **executor_args):
    """BugBase is a microbiome analysis tool that determines high-level
       phenotypes present in microbiome samples.

    Parameters
    ----------
    counts : Pandas.DataFrame
        OTU counts
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    A dict of count tables, where every BugBase trait is one key-value pair,
    i.e. currently, following 9 traits get predicted:
      'Gram_Negative', 'Aerobic', 'Anaerobic', 'Stress_Tolerant',
      'Forms_Biofilms', 'Facultatively_Anaerobic', 'Potentially_Pathogenic',
      'Contains_Mobile_Elements', 'Gram_Positive'
    Input counts get normalized as with Picrust.

    Install
    -------
    conda create --name bugbase
    conda activate bugbase
    mkdir -p $CONDA_PREFIX/src
    cd $CONDA_PREFIX/src
    git clone https://github.com/knights-lab/BugBase
    export BUGBASE_PATH=$CONDA_PREFIX/src/BugBase
    sudo apt-get install gfortran libblas-dev liblapack-dev
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
    edit: $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
        #!/bin/bash
        module load R
        export PATH=$CONDA_PREFIX/src/BugBase/bin:$PATH
        export BUGBASE_PATH=$CONDA_PREFIX/src/BugBase
    bash $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    ./bin/run.bugbase.r
    """
    def pre_execute(workdir, args):
        if is_numeric_dtype(args['counts'].index) \
                or all(map(lambda x: x.isdigit(), args['counts'].index)):
            args['counts'].to_csv('%s/input.txt' % workdir, sep="\t")
        else:
            raise ValueError(
                ('Not all features are numerical, that might point to the fact'
                 ' that your count table does not stem from closed reference '
                 'picking vs. Greengenes.'))
        if args['counts'].max().max() < 1:
            raise ValueError('You need to insert frequencies, '
                             'not relative abundances')

    def commands(workdir, ppn, args):
        commands = []

        commands.append('run.bugbase.r -i %s/input.txt -a -o %s/results' %
                        (workdir, workdir))

        return commands

    def post_execute(workdir, args):
        normalized_counts = pd.read_csv(
            '%s/results/normalized_otus/16s_normalized_otus.txt' % workdir,
            sep="\t", index_col=0)
        # normalize to abundances
        normalized_counts /= normalized_counts.sum()
        contrib_otus = pd.read_csv(
            '%s/results/otu_contributions/contributing_otus.txt' % workdir,
            sep="\t", index_col=0)

        results = dict()
        for trait in contrib_otus.columns:
            results[trait] = pd.DataFrame(
                (normalized_counts.values.T * contrib_otus[trait].values).T,
                index=normalized_counts.index,
                columns=normalized_counts.columns)

        return results

    return _executor('bugbase',
                     {'counts': counts.fillna(0.0)},
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=1,
                     environment=settings.BUGBASE_ENV,
                     **executor_args)


def correlation_diversity_metacolumns(metadata, categorial, alpha_diversities,
                                      beta_diversities,
                                      **executor_args):
    """
    Parameters
    ----------
    metadata : pd.DataFrame
    categorial : [str]
        List of column names that contain categorial data
    alpha_diversities : pd.DataFrame
    beta_diversities : dict{name, skbio.distanceMatrix}
    """
    METHODS_ALPHA = ['pearson', 'spearman']
    METHODS_BETA = ['permanova', 'anosim']
    FAKE_COL = '__fake_qiime2_numcol'

    def pre_execute(workdir, args):
        # basic tests
        if len(set(args['cols_cat']) - set(args['meta'].columns)) > 0:
            raise ValueError(('Not all columns of metadata table are described'
                              ' in categorial or nonCategorial!'))

        # write alpha diversity values into files
        if args['alpha'] is not None:
            for metric in args['alpha'].keys():
                args['alpha'][metric].to_frame().to_csv(
                    '%s/alpha_%s.tsv' % (workdir, metric), sep="\t")
        if args['beta'] is not None:
            # write beta diversity matrices into files
            for metric in args['beta'].keys():
                args['beta'][metric].write(
                    '%s/beta_%s.tsv' % (workdir, metric))
        # escape values that Qiime2 might identify as numeric
        for c in args['cols_cat']:
            args['meta'][c] = args['meta'][c].apply(
                lambda x: '_%s' % x if not str(x).startswith('_') else x)
        # escape chars like /
        for c in args['cols_cat']:
            args['meta'][c] = args['meta'][c].apply(
                lambda x: x.replace('/', '_'))

        # write metadata into file
        # add one numeric fake column to make q2 visualizer operate correctly
        if FAKE_COL in args['meta'].columns:
            raise ValueError("Column name clash!")
        m = args['meta']
        m[FAKE_COL] = list(range(m.shape[0]))
        m.to_csv(
            '%s/meta.tsv' % workdir, sep='\t', index_label='sample_name')

    def commands(workdir, ppn, args):
        commands = {'pre': [], 'main': [], 'post': []}

        # import beta distance matrix into Qiime2 artifacts
        if args['beta'] is not None:
            for metric in args['beta'].keys():
                commands['pre'].append(
                    ('qiime tools import'
                     ' --input-path %s/beta_%s.tsv'
                     ' --output-path %s/beta_%s.qza'
                     ' --type "DistanceMatrix"') % (
                        workdir, metric, workdir, metric))

        array_i = 1
        if args['beta'] is not None:
            for metric in args['beta'].keys():
                for column in args['cols_cat']:
                    for method in METHODS_BETA:
                        commands['main'].append(
                            ('if [ ${%s} -eq %i ]; then qiime diversity beta-group-significance '
                             '--i-distance-matrix %s/beta_%s.qza '
                             '--m-metadata-file %s/meta.tsv '
                             '--m-metadata-column %s '
                             '--p-method %s '
                             '--output-dir %s/beta-group-significance_%s_%s_%s/; fi') % (
                                settings.VARNAME_PBSARRAY, array_i,
                                workdir, metric,
                                workdir,
                                column,
                                method,
                                workdir, metric, column, method))
                        commands['main'].append(
                             ('if [ ${%s} -eq %i ]; then qiime tools export '
                             '--input-path %s/beta-group-significance_%s_%s_%s/visualization.qzv '
                             '--output-path %s/beta-group-significance_%s_%s_%s/raw/; fi') % (
                                settings.VARNAME_PBSARRAY, array_i,
                                workdir, metric, column, method,
                                workdir, metric, column, method))
                        array_i += 1

        # store alpha diversities as Qiime2 artifacts
        if args['alpha'] is not None:
            for metric in args['alpha'].keys():
                commands['pre'].append(
                    ('qiime tools import '
                     '--input-path %s/alpha_%s.tsv '
                     '--output-path %s/alpha_%s.qza '
                     '--type "SampleData[AlphaDiversity]"') % (
                        workdir, metric, workdir, metric))
                commands['main'].append(
                    ('if [ ${%s} -eq %i ]; then qiime diversity alpha-group-significance '
                     '--i-alpha-diversity %s/alpha_%s.qza '
                     '--m-metadata-file %s/meta.tsv '
                     '--output-dir %s/alpha-group-significance_%s/; fi') % (
                        settings.VARNAME_PBSARRAY, array_i,
                        workdir, metric, workdir, workdir, metric))
                commands['main'].append(
                    ('if [ ${%s} -eq %i ]; then qiime tools export '
                     '--input-path %s/alpha-group-significance_%s/'
                     'visualization.qzv '
                     '--output-path %s/alpha-group-significance_%s/raw/; fi') % (
                        settings.VARNAME_PBSARRAY, array_i,
                        workdir, metric, workdir, metric))
                array_i += 1
            for metric in args['alpha'].keys():
                for method in METHODS_ALPHA:
                    commands['main'].append(
                        ('if [ ${%s} -eq %i ]; then qiime diversity alpha-correlation '
                         '--i-alpha-diversity %s/alpha_%s.qza '
                         '--m-metadata-file %s/meta.tsv '
                         '--p-method %s '
                         '--output-dir %s/alpha-correlation_%s_%s/; fi') % (
                            settings.VARNAME_PBSARRAY, array_i,
                            workdir, metric, workdir, method,
                            workdir, metric, method))
                    commands['main'].append(
                        ('if [ ${%s} -eq %i ]; then qiime tools export '
                         '--input-path %s/alpha-correlation_%s_%s/'
                         'visualization.qzv '
                         '--output-path %s/alpha-correlation_%s_%s/raw/; fi') % (
                            settings.VARNAME_PBSARRAY, array_i,
                            workdir, metric, method, workdir, metric, method))
                    array_i += 1

        return commands

    def post_execute(workdir, args):
        # file content is not strictly json. some HTML content contains double
        # quotes enclosed by single quotes. My strategy: remove all HTML context
        # then convert all single quotes into double quotes and parse as json
        regex = re.compile(r"\<table.*?\<\/table\>")

        results = []
        if args['alpha'] is not None:
            for metric in args['alpha'].keys():
                fp_asig = '%s/alpha-group-significance_%s/raw/' % (workdir, metric)
                for _, _, files in os.walk(fp_asig):
                    for file in files:
                        if file.startswith('column-') and file.endswith('.jsonp'):
                            column = '.'.join(("-".join(
                                file.split('-')[1:])).split('.')[:-1])
                            with open('%s/%s' % (fp_asig, file), 'r') as f:
                                # read file content
                                content = "".join(f.readlines())
                                # convert ' into "
                                content = content.replace("'", '"')
                                # remove load_data( ... ); wrapping
                                content = '[%s]' % content[len('load_data('):-2]
                                # remove HTML table with single and double quotes
                                content = regex.sub("", content)

                                testres = None
                                for entry in json.loads(content):
                                    if (type(entry) == dict) and len(set(['H','p']) & set(entry.keys())) >= 2:
                                        testres = entry
                                        break

                                assert(testres['p'] >= 0)
                                results.append({'div': 'alpha',
                                                'type': 'group-significance',
                                                'metric': metric,
                                                'column': column,
                                                'test-statistic': testres['H'],
                                                'test statistic name': 'H',
                                                'p-value': testres['p'],
                                                'test':
                                                'Kruskal-Wallis (all groups)'})
                    break

                regexAt = r'"testStat":\s+"?(-?\d*\.\d+|nan)"?'
                regexAp = r'"pVal":\s+"?(\d*\.\d+)"?'
                regexAs = r'"sampleSize":\s+"?(\d+)"?'
                for method in ['pearson', 'spearman']:
                    fp_acorr = '%s/alpha-correlation_%s_%s/raw/' % (
                        workdir, metric, method)
                    for _, _, files in os.walk(fp_acorr):
                        for file in files:
                            if file.startswith('column-') and \
                                    file.endswith('.jsonp'):
                                column = '.'.join(("-".join(
                                    file.split('-')[1:])).split('.')[:-1])
                                with open('%s/%s' % (fp_acorr, file), 'r') as f:
                                    content = "\n".join(f.readlines())
                                    results.append({'div': 'alpha',
                                                    'type': 'correlation',
                                                    'metric': metric,
                                                    'column': column,
                                                    'test-statistic':
                                                    re.findall(regexAt,
                                                               content)[0],
                                                    'p-value':
                                                    re.findall(regexAp,
                                                               content)[0],
                                                    'sampleSize':
                                                    re.findall(regexAs,
                                                               content)[0],
                                                    'test': method})
                    break

        regexBtn = r'<th>test statistic name</th>\s*<td>(.+?)</td>'
        regexBss = r'<th>sample size</th>\s*<td>(.+?)</td>'
        regexBts = r'<th>test statistic</th>\s*<td>(.+?)</td>'
        regexBpv = r'<th>p-value</th>\s*<td>(.+?)</td>'
        regexBnp = r'<th>number of permutations</th>\s*<td>(.+?)</td>'
        regexBng = r'<th>number of groups</th>\s*<td>(.+?)</td>'
        if args['beta'] is not None:
            for metric in args['beta'].keys():
                for method in ['permanova', 'anosim']:
                    for _, dirs, _ in os.walk(workdir):
                        for dir in dirs:
                            if dir.startswith(
                                    'beta-group-significance_%s' % metric) and \
                                    dir.endswith(method):
                                column = dir[
                                    len('beta-group-significance_%s' % metric) +
                                    1: -1 * (len(method)+1)]
                                with open('%s/%s/raw/index.html' % (
                                        workdir, dir), 'r') as f:
                                    content = ("".join(f.readlines()).replace(
                                        '\n', ''))
                                    results.append({'div': 'beta',
                                                    'type': 'group-significance',
                                                    'metric': metric,
                                                    'column': column,
                                                    'test statistic name':
                                                    re.findall(regexBtn,
                                                               content)[0],
                                                    'sampleSize':
                                                    re.findall(regexBss,
                                                               content)[0],
                                                    'test-statistic':
                                                    re.findall(regexBts,
                                                               content)[0],
                                                    'p-value':
                                                    re.findall(regexBpv,
                                                               content)[0],
                                                    'number of permutations':
                                                    re.findall(regexBnp,
                                                               content)[0],
                                                    'number of groups':
                                                    re.findall(regexBng,
                                                               content)[0],
                                                    'test': method})
                        break

        # add information about those columns of the metadata that have not
        # been tested because all their values were the same.
        for col in args['cols_onevalue']:
            results.append({'type': 'unique value',
                            'column': col,
                            'test statistic name': 'not executed',
                            'p-value': 1.0,
                            'test': 'not executed'})

        pd_results = pd.DataFrame(results)
        pd_results['p-value'] = pd_results['p-value'].apply(
            lambda x: np.nan if x == 'nan' else x).astype(float)

        if FAKE_COL in pd_results.columns:
            del pd_results[FAKE_COL]

        return pd_results

    # synchronize samples across metadata, alpha and beta diversity to the
    # smallest shared group
    idx_samples = set(metadata.index)
    if alpha_diversities is not None:
        idx_samples &= set(alpha_diversities.index)
    for metric in beta_diversities.keys():
        idx_samples &= set(beta_diversities[metric].ids)
    if (len(idx_samples) < metadata.shape[0]) |\
            ((alpha_diversities is not None) and (len(idx_samples) < alpha_diversities.shape[0])) |\
            any([len(idx_samples) < m.shape[0]
                 for m in beta_diversities.values()]):
        sys.stderr.write(
            'Reducing analysis to %i samples.\n' % len(idx_samples))

    idx_samples = list(idx_samples)
    # find columns that a) have only one value for all samples ...
    cols_onevalue = [col
                     for col in metadata.columns
                     if len(metadata.loc[idx_samples, col].unique()) == 1]
    # ... or b) are categorial, but have different values for all samples
    cols_alldiff = [col
                    for col in categorial
                    if len(metadata.loc[idx_samples, col].unique()) ==
                    metadata.loc[idx_samples, col].shape[0]]

    array_alpha = len(alpha_diversities.keys()) * (1 + len(METHODS_ALPHA))
    array_beta = len(beta_diversities.keys()) * \
        len(set(categorial) - set(cols_alldiff) - set(cols_onevalue)) * \
        len(METHODS_BETA)

    return _executor('corr-divmeta',
                     {'alpha': alpha_diversities.loc[idx_samples, :] if alpha_diversities is not None else None,
                      'beta': {k: m.filter(idx_samples)
                               for k, m in beta_diversities.items()},
                      'meta': metadata.loc[idx_samples,
                                           sorted(list(set(metadata.columns) -
                                                       set(cols_onevalue)))]
                      .sort_index(),
                      'cols_cat': sorted(list(set(categorial) -
                                              set(cols_alldiff) -
                                              set(cols_onevalue))),
                      'cols_onevalue': sorted(cols_onevalue)},
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=1,
                     array=array_alpha + array_beta,
                     environment=settings.QIIME2_ENV,
                     **executor_args)


def emperor(metadata, beta_diversities, fp_results, other_beta_diversities=None, infix="", run_tsne_umap=False, ppn=1, **executor_args):
    """Generates Emperor plots as qzv. Or procrustes if two distance metrics are given.

    Parameters
    ----------
    metadata : Pandas.DataFrame
        The metadata about samples to be plotted. Samples not included in
        metadata will be omitted from ordination and plotting!
    beta_diversities : dict(str: DistanceMatrix) OR dict(str: fp to qza)
        Dictionary of (multiple) beta diversity distance metrices.
        OR a dictionary of (multiple) QZA of beta diversity distance metrics,
        in case they are HUGE, i.e. multiple GB of filesize.
    fp_results : str
        Filepath to directory where to store generated emperor plot qzvs.
    other_beta_diversities : dict(str: DistanceMatrix)
        Default: None
        Dictionary of (multiple) beta diversity distance metrices as other
        distances for procrustes plots.
    infix : str
        Output filenames have pattern: "emperor%s_%s.gzv" % (infix, metric)
    run_tsne_umap : bool
        Default: False.
        Besides PCoA, also compute ordinations via t-SNE and UMAP.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    ?"""

    ORDINATIONS = ['pcoa']
    if run_tsne_umap:
        ORDINATIONS.extend(['tsne', 'umap'])

    array = len(ORDINATIONS) * len(beta_diversities)

    def pre_execute(workdir, args):
        samples = set(args['metadata'].index)
        for metric in args['beta_diversities'].keys():
            if isinstance(args['beta_diversities'][metric], DistanceMatrix):
                samples &= set(args['beta_diversities'][metric].ids)
            else:
                if (not args['beta_diversities'][metric].endswith('.qza')) or (not os.path.exists(args['beta_diversities'][metric])):
                    raise ValueError("The file '%s' you provided as '%s' distance metric, does not end with .qza OR does not exists!" % (args['beta_diversities'][metric], metric))
        if args['other_beta_diversities'] is not None:
            if sorted(args['beta_diversities'].keys()) != sorted(args['other_beta_diversities'].keys()):
                raise ValueError("Procrustes: reference and other beta diversity metrics do NOT contain the same metrics!")
            for metric in args['other_beta_diversities'].keys():
                if isinstance(args['other_beta_diversities'][metric], DistanceMatrix):
                    samples &= set(args['other_beta_diversities'][metric].ids)
                else:
                    if (not args['other_beta_diversities'][metric].endswith('.qza')) or (not os.path.exists(args['other_beta_diversities'][metric])):
                        raise ValueError("The file '%s' you provided as other '%s' distance metric, does not end with .qza OR does not exists!" % (args['other_beta_diversities'][metric], metric))

        if isinstance(list(args['beta_diversities'].values())[0], DistanceMatrix):
            if (args['metadata'].shape[0] != len(samples)):
                sys.stderr.write(
                    'Info: reducing number of samples for Emperor plot to %i\n' %
                    len(samples))
        else:
            sys.stderr.write(
                'Info: since you provide file-paths to *.qza\'s, we cannot merge metadata and distance matrixes. Please ensure matching IDs yourself!')

        # write metadata to tmp file
        args['metadata'].loc[list(samples), :].to_csv(
            workdir+'/metadata.tsv', sep="\t", index_label='sample_name')

        # write distance metrices to tmp files
        for metric in args['beta_diversities'].keys():
            if isinstance(args['beta_diversities'][metric], DistanceMatrix):
                os.makedirs('%s/%s' % (workdir, metric), exist_ok=True)
                args['beta_diversities'][metric].filter(samples).write(
                    '%s/%s/distance-matrix.tsv' % (workdir, metric))
        if args['other_beta_diversities'] is not None:
            if isinstance(args['other_beta_diversities'][metric], DistanceMatrix):
                for metric in args['other_beta_diversities'].keys():
                    os.makedirs('%s/other_%s' % (workdir, metric), exist_ok=True)
                    args['other_beta_diversities'][metric].filter(samples).write(
                        '%s/other_%s/distance-matrix.tsv' % (workdir, metric))

    def commands(workdir, ppn, args):
        commands = {'pre': [], 'main': [], 'post': []}

        for metric in args['beta_diversities'].keys():
            # note: skip importing if user provides *.qza file-paths
            if isinstance(args['beta_diversities'][metric], DistanceMatrix):
                # import DistanceMatrix
                commands['pre'].append(
                    ('qiime tools import '
                     '--input-path %s/%s '
                     '--type "DistanceMatrix" '
                     '--output-path %s/beta_%s.qza ') %
                    (workdir, metric, workdir, metric))
            if args['other_beta_diversities'] is not None:
                if isinstance(args['other_beta_diversities'][metric], DistanceMatrix):
                    # import DistanceMatrix for other
                    commands['pre'].append(
                        ('qiime tools import '
                         '--input-path %s/other_%s '
                         '--type "DistanceMatrix" '
                         '--output-path %s/other_beta_%s.qza ') %
                        (workdir, metric, workdir, metric))

        arrayid = 1
        for metric in args['beta_diversities'].keys():
            for ordname in ORDINATIONS:
                # compute ordination
                commands['main'].append(
                    ('if [ ${%s} -eq %i ]; then qiime diversity %s '
                     '--i-distance-matrix %s '
                     '--o-%s %s/%s_%s; fi ') %
                    (settings.VARNAME_PBSARRAY, arrayid,
                     ordname,
                     '%s/beta_%s.qza' % (workdir, metric) if isinstance(args['beta_diversities'][metric], DistanceMatrix) else args['beta_diversities'][metric],
                     ordname, workdir, ordname, metric))

                # export ordination to return insights
                commands['main'].append(
                    ('if [ ${%s} -eq %i ]; then qiime tools export '
                     '--input-path %s/%s_%s.qza '
                     '--output-path %s/%s_%s; fi ') %
                     (settings.VARNAME_PBSARRAY, arrayid,
                      workdir, ordname, metric,
                      workdir, ordname, metric))

                if args['other_beta_diversities'] is not None:
                    # compute other ordination
                    commands['main'].append(
                        ('if [ ${%s} -eq %i ]; then qiime diversity %s '
                         '--i-distance-matrix %s '
                         '--o-%s %s/other_%s_%s; fi ') %
                        (settings.VARNAME_PBSARRAY, arrayid,
                         ordname,
                         '%s/other_beta_%s.qza' % (workdir, metric) if isinstance(args['other_beta_diversities'][metric], DistanceMatrix) else args['other_beta_diversities'][metric],
                         ordname, workdir, ordname, metric))

                    # generate procrustes emperor plot
                    commands['main'].append(
                        ('if [ ${%s} -eq %i ]; then qiime emperor procrustes-plot '
                         '--i-reference-pcoa %s/%s_%s.qza '
                         '--i-other-pcoa %s/other_%s_%s.qza '
                         '--m-metadata-file %s/metadata.tsv '
                         '--o-visualization %s/emperor-%s-procrustes_%s.qzv; fi') %
                        (settings.VARNAME_PBSARRAY, arrayid,
                         workdir, ordname, metric,
                         workdir, ordname, metric,
                         workdir,
                         workdir, ordname, metric))
                else:
                    # generate emperor plot
                    commands['main'].append(
                        ('if [ ${%s} -eq %i ]; then qiime emperor plot '
                         '--i-pcoa %s/%s_%s.qza '
                         '--m-metadata-file %s/metadata.tsv '
                         '--o-visualization %s/emperor-%s_%s.qzv; fi') %
                         (settings.VARNAME_PBSARRAY, arrayid,
                          workdir, ordname, metric,
                          workdir,
                          workdir, ordname, metric))
                arrayid += 1

        return commands

    def post_execute(workdir, args):
        results = {'ordinations': dict(), 'visualizations': dict()}
        os.makedirs(fp_results, exist_ok=True)
        label_procrustes = ""
        if args['other_beta_diversities'] is not None:
            label_procrustes = '-procrustes'
        for ordname in ORDINATIONS:
            for metric in args['beta_diversities']:
                with open("%s/%s_%s/ordination.txt" % (workdir, ordname, metric), 'r') as f:
                    results['ordinations'][metric] = f.readlines()
                results['visualizations'][metric] = os.path.join(
                    fp_results, 'emperor-%s%s%s_%s.qzv' % (ordname, label_procrustes, infix, metric))
                shutil.move(
                    "%s/emperor-%s%s_%s.qzv" % (workdir, ordname, label_procrustes, metric),
                    results['visualizations'][metric])
        return results

    return _executor('emperor',
                     {'metadata': metadata,
                      'beta_diversities': beta_diversities,
                      'other_beta_diversities': other_beta_diversities},
                     pre_execute,
                     commands,
                     post_execute,
                     environment=settings.QIIME2_ENV,
                     ppn=1,
                     array=array,
                     **executor_args)


def empress(metadata: pd.DataFrame, beta_diversities, counts: pd.DataFrame, reference_tree: str, fp_results: str, fix_zero_len_branches=False, infix="", ppn=1, **executor_args):
    """Empress Plot

    Parameters
    ----------
    metadata : Pandas.DataFrame
        The metadata about samples to be plotted. Samples not included in
        metadata will be omitted from ordination and plotting!
    counts : Pandas.DataFrame

    beta_diversities : dict(str: DistanceMatrix)
        Dictionary of (multiple) beta diversity distance metrices.
    reference_tree : str
        Filepath to insertion tree
    fp_results : str
        Filepath to directory where to store generated emperor plot qzvs.
    infix : str
        Output filenames have pattern: "emperor%s_%s.gzv" % (infix, metric)
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    ?"""
    # general sanity checks
    (counts, metadata) = sync_counts_metadata(counts, metadata)
    def pre_execute(workdir, args):
        samples = set(args['metadata'].index)
        for metric in args['beta_diversities'].keys():
            samples &= set(args['beta_diversities'][metric].ids)

        if (args['metadata'].shape[0] != len(samples)):
            sys.stderr.write(
                'Info: reducing number of samples for Emperor plot to %i\n' %
                len(samples))

        # write metadata to tmp file
        args['metadata'].loc[samples, :].to_csv(
            workdir+'/metadata.tsv', sep="\t", index_label='sample_name')

        # write distance metrices to tmp files
        for metric in args['beta_diversities'].keys():
            os.makedirs('%s/%s' % (workdir, metric), exist_ok=True)
            args['beta_diversities'][metric].filter(samples).write(
                '%s/%s/distance-matrix.tsv' % (workdir, metric))

        pandas2biom('%s/input.biom' % workdir, args['counts'])

        if 'verbose' not in executor_args:
            verbose = sys.stderr
        else:
            verbose = executor_args['verbose']
        writeReferenceTree(args['reference_tree'], workdir,
                           fix_zero_len_branches, verbose=verbose,
                           name_analysis='empress')

    def commands(workdir, ppn, args):
        commands = []

        commands.append((
            'qiime tools import '
            '--input-path %s/input.biom '
            '--output-path %s/counts '
            '--type "FeatureTable[Frequency]"') % (workdir, workdir))
        commands.append(
            ('qiime tools import '
             '--input-path %s '
             '--output-path %s '
             '--type "Phylogeny[Rooted]"') %
            (workdir+'/reference.tree',
             workdir+'/reference_tree.qza'))

        # import diversity matrix as qiime2 artifact
        for metric in args['beta_diversities'].keys():
            commands.append(
                ('qiime tools import '
                 '--input-path %s '
                 '--type "DistanceMatrix" '
                 # '--source-format DistanceMatrixDirectoryFormat '
                 '--output-path %s ') %
                ('%s/%s' % (workdir, metric),
                 # " % Properties([\"phylogenetic\"])"
                 # if 'unifrac' in metric else '',
                 '%s/beta_%s.qza' % (workdir, metric)))
            # compute PcoA
            commands.append(
                ('qiime diversity pcoa '
                 '--i-distance-matrix %s '
                 '--o-pcoa %s ') %
                ('%s/beta_%s.qza' % (workdir, metric),
                 '%s/pcoa_%s' % (workdir, metric))
            )
            # create empress plot
            commands.append(
                ('qiime empress community-plot '
                 '--i-tree %s '
                 '--i-feature-table %s '
                 '--i-pcoa %s '
                 '--m-sample-metadata-file %s '
                 '--o-visualization %s') %
                ('%s/reference_tree.qza' % workdir,
                 '%s/counts.qza' % workdir,
                 '%s/pcoa_%s.qza' % (workdir, metric),
                 '%s/metadata.tsv' % workdir,
                 '%s/empress_%s.qzv' % (workdir, metric),
                 )
            )

        return commands

    def post_execute(workdir, args):
        results = dict()
        os.makedirs(fp_results, exist_ok=True)
        for metric in args['beta_diversities']:
            results[metric] = os.path.join(
                fp_results, 'empress%s_%s.qzv' % (infix, metric))
            shutil.move(
                "%s/empress_%s.qzv" % (workdir, metric),
                results[metric])
        return results

    return _executor('empress',
                     {'metadata': metadata,
                      'counts': counts,
                      'reference_tree': reference_tree,
                      'beta_diversities': beta_diversities},
                     pre_execute,
                     commands,
                     post_execute,
                     environment="qiime2-2020.11",
                     ppn=ppn,
                     array=1,
                     **executor_args)


def taxonomy_RDP(counts, fp_classifier, ppn=4, environment=settings.QIIME2_ENV, **executor_args):
    """Uses q2-feature-classifier to obtain taxonomic lineages for features
       in counts table.

    Paramaters
    ----------
    counts : Pandas.DataFrame
        OTU counts
    fp_classifier : str
        Filepath to Qiime2 pre-trained sklearn based classifier.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    Pandas.DataFrame with lineage strings and confidence for each feature."""

    def pre_execute(workdir, args):
        # store counts as a biom file
        with open(workdir+'/features.fasta', 'w') as f:
            for feature in args['features']:
                f.write('>%s\n%s\n' % (feature, feature))

    def commands(workdir, ppn, args):
        commands = []

        commands.append(
            ('qiime tools import '
             '--input-path %s '
             '--type "FeatureData[Sequence]" '
             # '--source-format BIOMV210Format '
             '--output-path %s ') %
            (workdir+'/features.fasta', workdir+'/input.qza'))
        commands.append(
            ('qiime feature-classifier classify-sklearn '
             '--i-reads %s '
             '--i-classifier %s '
             '--o-classification %s '
             '--p-n-jobs %i') %
            (workdir+'/input.qza',
             args['fp_classifier'],
             workdir+'/taxonomyRDP.qza',
             ppn))
        commands.append(
            ('qiime tools export '
             '--input-path %s/taxonomyRDP.qza '
             '--output-path %s') %
            (workdir, workdir))
        return commands

    def post_execute(workdir, args):
        taxonomy = pd.read_csv('%s/taxonomy.tsv' % workdir, sep="\t",
                               index_col=0)
        return taxonomy

    return _executor('taxRDP',
                     {'features': sorted(list(counts.index)),
                      'fp_classifier': fp_classifier},
                     pre_execute,
                     commands,
                     post_execute,
                     environment=environment,
                     ppn=ppn,
                     **executor_args)


def volatility(metadata: pd.DataFrame, alpha_diversity: pd.DataFrame,
               col_entity: str, col_group: str, col_event: str,
               col_alpha_metric: str='shannon', **executor_args):
    """"""
    def pre_execute(workdir, args):
        # store alpha diversity as tsv file
        alpha_diversity.to_csv(
            workdir+'/diversity.tsv', sep="\t", index_label='sample_name')
        # store metadata as tsv file
        metadata.to_csv(
            workdir+'/metadata.tsv', sep="\t", index_label='sample_name')

    def commands(workdir, ppn, args):
        commands = []

        commands.append(
            ('qiime longitudinal volatility '
             '--m-metadata-file %s '
             '--m-metadata-file %s '
             '--p-default-metric %s '
             '--p-default-group-column %s '
             '--p-state-column %s '
             '--p-individual-id-column %s '
             '--o-visualization %s/volatility.qzv ') %
            (workdir+'/metadata.tsv',
             workdir+'/diversity.tsv',
             col_alpha_metric,
             col_group,
             col_event,
             col_entity,
             workdir))
        commands.append(
             ('qiime tools export '
              '--input-path %s/volatility.qza '
              '--output-path %s') %
             (workdir, workdir))
        return commands

    def post_execute(workdir, args):
        # taxonomy = pd.read_csv('%s/taxonomy.tsv' % workdir, sep="\t",
        #                        index_col=0)
        # return taxonomy
        return None

    # if fp_classifier is not None:
    #     fp_classifier = os.path.abspath(fp_classifier)
    return _executor('volatility',
                     {'metadata': metadata,
                      'alpha_diversity': alpha_diversity},
                     pre_execute,
                     commands,
                     post_execute,
                     environment=settings.QIIME2_ENV,
                     ppn=1,
                     **executor_args)


def dada2_pacbio(demux: pd.DataFrame, fp_fastqprefix: str=None, seq_primer_fwd: str="AGRGTTYGATYMTGGCTCAG", seq_primer_rwd: str="RGYTACCTTGTTACGACTT",
                 ppn=10, pmem='30GB', walltime='6:00:00', **executor_args):
    """Uses dada2 to demultiplex PacBio amplicon sequences and returns feature table.

    Paramaters
    ----------
    demux : Pandas.DataFrame
        Demultiplexing information. Must contain columns:
        - sample_name: should match to your metadata for this experiment.
        - fp_fastq: relative path to sequencing file in fastq format
    fp_fastqprefix : str
        Default: None
        If given, a directory pointing to the fastq files that prefixes all
        relative file paths given in the demux table, column "fp_fastq".
    seq_primer_fwd : str
        Default: "AGRGTTYGATYMTGGCTCAG" = F27
        The forward primer sequence.
    seq_primer_rwd : str
        Default: "RGYTACCTTGTTACGACTT" = R1492
        The reverse primer sequence.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose, dirty

    Returns
    -------
    Pandas.DataFrame: feature table.

    Install
    -------
    conda create --name dada2_pacbio -y
    conda activate dada2_pacbio

    # there is a concurrency issue with one of the used libaries, see https://github.com/benjjneb/dada2/issues/684
    # to solve, we need to make sure to use the following conda package
    conda install r-rcppparallel=4.4.3=r35h0357c0b_2 -c conda-forge -y

    # install dependencies devtools will complain about
    conda install -c r -c conda-forge r-git2r r-httr r-gh r-usethis

    ### somehow basic libraries cannot be found by default. Thus, I came up with a dirty hack:
    # the linker is ~/miniconda3/envs/dada2_pacbio/x86_64-conda_cos6-linux-gnu/bin/ld I renamed it into ld_bin and created a new bash script named ld
    # within the script I am adding paths to the linker. Content of this wrapper is:
    #	#!/usr/bin/bash
    #	ld -L/usr/lib64/ -L/home/jansses/miniconda3/envs/dada2_pacbio/x86_64-conda_cos6-linux-gnu/lib/ $@
    # don't forget to chmod a+x

    # you should not update to later version if R asks for, because this will crash condas R packages
    R
    	install.packages("devtools")
    	library("devtools")
    	devtools::install_github("benjjneb/dada2", ref="v1.12")
    	if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
    	BiocManager::install("gridExtra")
    	BiocManager::install("phyloseq")
    """
    MANDATORY_DEMUX_COLUMNS = ['sample_name', 'fp_fastqprefix']
    DEMUX_INTERNAL_COLUMNS = ['__demux_fp_fastq']

    def pre_execute(workdir, args):
        # test if demux table contains necessary columns
        missing_columns = []
        for c in MANDATORY_DEMUX_COLUMNS:
            if c not in args['demux'].columns:
                missing_columns.append(c)
        if len(missing_columns) > 0:
            raise ValueError("You demux table does not contain all mandatory columns. I am missing column(s) '%s'!" % "', '".join(missing_columns))

        # test that given demux column names do not collide with internally created columns
        colliding_columns = []

        for c in DEMUX_INTERNAL_COLUMNS:
            if c in args['demux'].columns:
                colliding_columns.append(c)
        if len(colliding_columns) > 0:
            raise ValueError("Your demux table contains columns whose names collide with internal operations. Please rename columns '%s'." % "', '".join(colliding_columns))

        # merge prefix and individual sample input file paths + test if files are present
        def _join_test_fps(fp_fastq):
            fp = os.path.join("" if fp_fastqprefix is None else fp_fastqprefix, fp_fastq)
            if os.path.exists(fp):
                return os.path.abspath(fp)
            else:
                return None
        args['demux']['__demux_fp_fastq'] = args['demux']['fp_fastqprefix'].apply(_join_test_fps)

        if args['demux']['__demux_fp_fastq'].dropna().shape[0] < args['demux'].shape[0]:
            raise ValueError("%i of %i fastq input files cannot be found! You might have to specify or correct the 'fp_fastqprefix'\n  %s" % (args['demux'][pd.isnull(args['demux']['__demux_fp_fastq'])].shape[0], args['demux'].shape[0], '  \n'.join(args['demux'][pd.isnull(args['demux']['__demux_fp_fastq'])]['fp_fastqprefix'].values)))

        duplicate_samples = args['demux'][args['demux']['sample_name'].isin(
            [n for n, c in args['demux']['sample_name'].value_counts().iteritems() if c > 1])]
        if duplicate_samples.shape[0] > 0:
            raise ValueError('Your demux table contains following duplicates:\n%s' % duplicate_samples['sample_name'].value_counts())

        # store demux table as input for R call
        args['demux'][['sample_name'] + DEMUX_INTERNAL_COLUMNS].to_csv("%s/demux.csv" % workdir, sep="\t")

        # generate R code
        with open('%s/dada2_pacbio.R' % workdir, 'w') as f:
            f.write('library(dada2); packageVersion("dada2")\n')
            f.write('library(Biostrings)\n')
            f.write('library(ShortRead)\n')
            f.write('library(ggplot2)\n')
            f.write('library(reshape2)\n')
            f.write('library(gridExtra)\n')
            f.write('library(phyloseq)\n')

            f.write('path.out <- "Figures/"\n')
            f.write('path.rds <- "RDS/"\n')
            f.write('F27 <- "%s"\n' % args['seq_primer_fwd'])
            f.write('R1492 <- "%s"\n' % args['seq_primer_rwd'])
            f.write('rc <- dada2:::rc\n')
            f.write('theme_set(theme_bw())\n')

            f.write('\n# list all input files: fastq\n')
            f.write('fns <- %s\n' % ('c(\n%s)' % ',\n'.join(map(lambda x: '    "%s"' % x, args['demux']['__demux_fp_fastq'].values))))

            f.write('\n# set tmp filename for primer less fastq\n')
            f.write('nops <- %s\n' % ('c(\n%s)' % ',\n'.join(map(lambda x: '    "%s"' % str(os.path.abspath(os.path.join(workdir, '01_noprimer', x+'.fastq'))), args['demux']['sample_name'].values))))
            f.write('\n# perform primer removal\n')
            f.write('prim <- removePrimers(fns, nops, primer.fwd=F27, primer.rev=dada2:::rc(R1492), orient=TRUE)\n')

            f.write('\n# inspect length distribution\n')
            f.write('lens.fn <- lapply(nops, function(fn) nchar(getSequences(fn)))\n')
            f.write('lens <- do.call(c, lens.fn)\n')
            f.write('write.table(lens, file="%s/results_lengthdistribution.csv", sep="\\t")\n' % workdir)

            f.write('\n# filter\n')
            f.write('filts <- %s\n' % ('c(\n%s)' % ',\n'.join(map(lambda x: '    "%s"' % str(os.path.abspath(os.path.join(workdir, '02_filtered', x+'.fastq'))), args['demux']['sample_name'].values))))
            f.write('track <- filterAndTrim(nops, filts, minQ=3, minLen=1000, maxLen=1600, maxN=0, rm.phix=FALSE, maxEE=2)\n')
            #f.write('write.table(track, file="%s/results_track.csv", sep="\\t")\n' % workdir)

            f.write('\n# dereplicate sequences\n')
            f.write('drp <- derepFastq(filts, verbose=TRUE)\n')

            f.write('\n# learn error model\n')
            f.write('errmodel <- learnErrors(drp, errorEstimationFunction=PacBioErrfun, BAND_SIZE=32, multithread=TRUE)\n')
            f.write('write.table(getErrors(errmodel), file="%s/results_errors_table.csv")\n' % workdir)
            f.write('saveRDS(errmodel, "%s/dada2_error_model.rds")\n' % workdir)

            f.write('\n# Denoise\n')
            f.write('dd2 <- dada(drp, err=errmodel, BAND_SIZE=32, multithread=TRUE)\n')
            f.write('saveRDS(dd2, "%s/dada2_error_model_dd2.rds")\n' % workdir)
            f.write('st2 <- makeSequenceTable(dd2)\n')
            f.write('write.table(st2, "%s/results_feature-table.csv", sep="\t")\n' % workdir)
            f.write('write.table(cbind(sample_name=c(%s), ccs=prim[,1], primers=prim[,2], filtered=track[,2], denoised=sapply(dd2, function(x) sum(x$denoised))), file="%s/results_summary.csv", sep="\\t")\n' % (','.join(map(lambda x: '"%s"' % x, args['demux']['sample_name'].values)), workdir))
            f.write('\n')

    def commands(workdir, ppn, args):
        commands = []

        commands.append('R --vanilla < %s/dada2_pacbio.R' % workdir)

        return commands

    def post_execute(workdir, args):
        counts = pd.read_csv('%s/results_feature-table.csv' % workdir, sep="\t").T.fillna(0).astype(int)
        counts.columns = map(lambda x: x[:-6] if x.endswith('.fastq') else x, counts.columns)
        lendistr = pd.read_csv('%s/results_lengthdistribution.csv' % workdir, sep="\t", squeeze=True)
        tracking = pd.read_csv('%s/results_summary.csv' % workdir, sep="\t", index_col=1).rename(columns={
            'ccs': '01_rawreads',
            'primers': '02_primers_removed',
            'filtered': '03_quality_filtering',
            'denoised': '04_final_counts',
        })
        del tracking['sample_name']
        tracking.index.name = 'sample_name'
        return {'counts': counts,
                'read_length_distribution': lendistr,
                'stats': tracking.sort_values('04_final_counts')}

    def post_cache(cache_results):
        # generate plot for read length distribution
        lendistr = cache_results['results']['read_length_distribution']
        fig, axes = plt.subplots(1, 1)
        g = sns.distplot(lendistr, kde=False, bins=100, ax=axes)
        g.set_xlim(lendistr.mean() - (2*lendistr.std()), lendistr.mean() + (2*lendistr.std()))
        g.set_ylabel('frequency')
        g.set_xlabel('read length')
        g.set_title('Histogram of read lengths\nWe expect to see a strong peak around ~1450bp')
        cache_results['results']['read_length_distribution'] = fig

        return cache_results

    return _executor('dada2_pacbio',
                     {'demux': demux.copy(),
                      'seq_primer_fwd': seq_primer_fwd,
                      'seq_primer_rwd': seq_primer_rwd},
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache=post_cache,
                     ppn=ppn,
                     pmem=pmem,
                     walltime=walltime,
                     environment=settings.DADA2PACBIO_ENV,
                     **executor_args)


def metalonda(counts: pd.DataFrame, meta: pd.DataFrame, col_time: str, col_entities: str, col_phenotype: str,
              colors_phenotype: dict={}, taxonomy: pd.Series=None,
              num_intervals: int=20, num_permutations: int=100,
              rf_iterations: int=10, rf_train_test_ratio: float=0.5,
              ppn=12, pmem='10GB', walltime='2:00:00', **executor_args):
    """METAgenomic LONgitudinal Differential Abundance method.

    Paramaters
    ----------
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose, dirty

    Returns
    -------
    ?

    Install
    -------
    conda create -n MetaLonDA r-essentials r-base
    conda activate MetaLonDA
    conda install -c r -c conda-forge r-git2r r-httr r-gh r-usethis

    # you should not update to later version if R asks for, because this will crash condas R packages
    R
    	install.packages("devtools")
    	library("devtools")
    	if (!requireNamespace("BiocManager", quietly = TRUE))
    		install.packages("BiocManager")
    	BiocManager::install("DESeq2")
    	BiocManager::install("metagenomeSeq")
    	BiocManager::install("edgeR")
    	install.packages("MetaLonDA")
    """
    counts = counts.loc[:, set(counts.columns) & set(meta.index)]
    meta = meta.loc[set(counts.columns) & set(meta.index), :]

    # since MetaLonDA uses feature names for file names,
    # we need to rename those in order to prevent file system issues
    # e.g. if feature names are full length 16S sequences
    map_featurenames = pd.Series({feature: "feature%i" % (i+1) for (i, feature) in enumerate(counts.sort_index().index)})
    counts_orig = counts.copy()
    counts.index = map_featurenames.loc[counts.index]

    def pre_execute(workdir, args):
        if meta[args['col_phenotype']].unique().shape[0] != 2:
            raise ValueError("You have more / less than 2 phenotypes!")

        # test if demux table contains necessary columns
        missing_columns = []
        for c in [col_time, col_entities, col_phenotype]:
            if c not in args['meta'].columns:
                missing_columns.append(c)
        if len(missing_columns) > 0:
            raise ValueError("You metadata do not contain all specified columns. I am missing column(s) '%s'!" % "', '".join(missing_columns))

        args['counts'].to_csv('%s/counts.csv' % workdir, sep="\t")
        map_featurenames.to_csv('%s/map_featurenames.csv' % workdir, sep="\t", header=True)

        # generate R code
        with open('%s/metalonda.R' % workdir, 'w') as f:
            f.write('library(MetaLonDA); packageVersion("MetaLonDA")\n')
            f.write('counts <- as.matrix(read.csv("%s/counts.csv", sep="\\t", row.names=1, header=T))\n' % workdir)
            f.write('Group = factor(c(%s))\n' % ', '.join(map(lambda x: '"%s"' % x, args['meta'].loc[args['counts'].columns, args['col_phenotype']].values)))
            f.write('Time = c(%s)\n' % ', '.join(map(lambda x: '%s' % x, args['meta'].loc[args['counts'].columns, args['col_time']].values)))
            f.write('ID = factor(c(%s))\n' % ', '.join(map(lambda x: '"%s"' % x, args['meta'].loc[args['counts'].columns, args['col_entities']].values)))

            f.write(('output.metalonda.all = metalondaAll('
                        'Count = counts, '
                        'Time = Time, '
                        'Group = Group, '
                        'ID = ID, '
                        'n.perm = %i, '
                        'fit.method = "nbinomial", '
                        'num.intervals = %i, '
                        'parall = FALSE, '
                        'pvalue.threshold = 0.05, '
                        'adjust.method = "BH", '
                        'time.unit = "days", '
                        'norm.method = "none", '
                        'prefix = "%s/MetaLonDA_results", '
                        'ylabel = "Read Counts", '
                        'col = c("black", "green"))\n') % (args['num_permutations'], args['num_intervals'], workdir))

    def commands(workdir, ppn, args):
        commands = []

        commands.append('export TMPDIR=%s' % workdir)
        commands.append('R --vanilla < %s/metalonda.R' % workdir)

        return commands

    def post_execute(workdir, args):
        if 'verbose' not in executor_args:
            verbose = sys.stderr
        else:
            verbose = executor_args['verbose']

        # parse dominant intervals and re-map feature names
        intervalls = pd.read_csv('%s/MetaLonDA_results/MetaLonDA_TimeIntervals.csv' % workdir, sep=",", index_col=0)
        map_featurenames = pd.read_csv('%s/map_featurenames.csv' % workdir, sep="\t").rename(columns={'Unnamed: 0': 'feature', '0': 'mapped_name'}).set_index('mapped_name')
        intervalls.index = map_featurenames.loc[intervalls.index, 'feature']
        intervalls = intervalls.reset_index().set_index(['feature', 'start', 'end'])

        featurecounts = []
        counts_remapped = counts.copy()
        counts_remapped.index = map_featurenames.loc[counts_remapped.index, 'feature']
        for (feature, start, end), row in intervalls.iterrows():
            x = meta[[col_entities, col_time]].merge(counts_remapped.loc[feature, :], left_index=True, right_index=True)
            x = x[x[col_time].apply(lambda x: start <= x <= end)].groupby(col_entities)[feature].mean()
            x.name = '%s@%s@%s' % (feature, start, end)
            featurecounts.append(x)
        featurecounts = pd.concat(featurecounts, axis=1, sort=False).T.reindex(meta[args['col_entities']].unique(), axis=1).fillna(0)

        phenotypes = meta.groupby(args['col_entities'])[args['col_phenotype']].unique().apply(lambda x: x[0])

        if (rf_iterations > 0) and (verbose is not None):
            verbose.write("running random forest: ")
        rf_scores = []
        for i in range(rf_iterations):
            if verbose is not None:
                verbose.write(".")
            # 50% train, 50% test
            clf = RandomForestClassifier(n_estimators=1000, n_jobs=1)
            X_train, X_test, y_train, y_test = train_test_split(
                featurecounts.T,
                phenotypes,
                test_size=rf_train_test_ratio)

            # train the ML tool
            clf = clf.fit(X_train, y_train)
            # make predictions for test samples
            prediction = pd.Series(clf.predict(X_test), index=X_test.index)
            # assess accurracy
            rf_scores.append({'accurracy': clf.score(X_test, y_test), 'iteration': i+1})
        if (rf_iterations > 0) and (verbose is not None):
            verbose.write(" done.\n")

        return {'intervals': intervalls,
                'featurecounts': featurecounts,
                'phenotypes': phenotypes,
                'rf_scores': pd.DataFrame(rf_scores)
               }

    def post_cache(cache_results):
        intervals = cache_results['results']['intervals'].reset_index()

        for phenotype in meta[col_phenotype].unique():
            if phenotype not in colors_phenotype:
                colors_phenotype[phenotype] = (['green', 'blue']*4)[len(colors_phenotype)]
        fig, ax = plt.subplots(1,1, figsize=(10, 0.5*intervals['feature'].unique().shape[0]))
        feature_names = dict()
        for i, (feature, g) in enumerate(intervals.groupby('feature')):
            feature_names[feature] = {
                'name': '%i: %s...' % (len(feature_names), feature[:40]) if len(feature) > 40 else feature,
                'taxonomy': "" if taxonomy is None else [r for r in taxonomy[feature].split(';') if len(r.strip()[3:]) > 0][-1],
            }
            for j, (phenotype, g_pheno) in enumerate(g.groupby('dominant')):
                for _, row in g_pheno.sort_values('start').iterrows():
                    ax.hlines(i, row['start'], row['end'], lw=6, color=colors_phenotype[phenotype])
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([f['name'] for f in feature_names.values()])
        ax.set_ylim(-1, len(feature_names))
        ax.xaxis.grid(which='both')
        ax.yaxis.grid(which='both')
        if taxonomy is not None:
            ax_taxa = ax.twinx()
            ax_taxa.set_ylim(ax.get_ylim())
            ax_taxa.set_yticks(ax.get_yticks())
            ax_taxa.set_yticklabels([f['taxonomy'] for f in feature_names.values()])

        ax.set_xlabel(col_time)
        ax.set_title("MetaLonDA significant intervals, %i permutations and %i time intervals" % (num_permutations, num_intervals))
        ax.legend(
            handles=[mpatches.Patch(color=colors_phenotype[phenotype], label=phenotype) for phenotype in meta[col_phenotype].unique()],
            loc='upper left',
            bbox_to_anchor=(1.21, 1.05), title="dominant in:")
        cache_results['results']['plot_summary'] = ax

        fig, axes = plt.subplots(intervals['feature'].unique().shape[0], 1, figsize=(10, 5*intervals['feature'].unique().shape[0]), sharex=False, sharey=False)
        for i, (feat, _) in enumerate(intervals.groupby('feature')):
            data = meta[[col_entities, col_time, col_phenotype]].merge(counts_orig.loc[feat,:], left_index=True, right_index=True)
            sns.lineplot(data=data, y=feat, x=col_time, hue=col_phenotype, palette=colors_phenotype, estimator=None, units=col_entities, ax=axes[i])
            axes[i].set_title(feature_names[feat]['name'])
            if taxonomy is not None:
                axes[i].set_title(axes[i].get_title() + "\n" + feature_names[feat]['taxonomy'])
            axes[i].set_ylabel('reads')
            axes[i].xaxis.grid(which='both')
        cache_results['results']['plot_counts'] = ax

        return cache_results


    return _executor('metalonda',
                     {'counts': counts.fillna(0.0),
                      'meta': meta,
                      'col_time': col_time,
                      'col_entities': col_entities,
                      'col_phenotype': col_phenotype,
                      'num_intervals': num_intervals,
                      'num_permutations': num_permutations},
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache=post_cache,
                     ppn=ppn,
                     pmem=pmem,
                     walltime=walltime,
                     environment="ggmap_metalonda",
                     **executor_args)


def feast(counts: pd.DataFrame, metadata: pd.DataFrame,
          col_envname, col_type,
          #col_envID=None,
          EM_iterations=1000,
          ppn=4, pmem='4GB', **executor_args):
    """FEAST for microbial source tracking.

    Paramaters
    ----------
    counts : Pandas.DataFrame
        feature-table
    metadata : Pandas.DataFrame
        metadata for samples of feature-table.
        Need to contain columns for "envname", "type" and "envID".
    col_envname : str
        column name in metadata specifying the environment _name_. Like "infant",
        "adult gut", "soil", ...
    INACTIVE col_envID : str
        column name in metadata specifying the environment ID to group "Sinks" and "Sources".
    col_type : str
        column name in metadata specifiying if sample is either "Sink" or "Source".
    EM_iterations : int
        Default: 1000.
        Number of EM iterations. We recommend using this default value.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose, dirty

    Returns
    -------
    pd.DataFrame: feature-table, with cols for samples and rows for source environment contribution.
    """
    # general sanity checks
    (counts, metadata) = sync_counts_metadata(counts, metadata)
    check_column_presents(metadata, [col_envname, col_type])
    def _fix_sourcesink(value):
        if value.lower() == 'sink':
            return 'Sink'
        elif value.lower() == 'source':
            return 'Source'
        return value
    metadata[col_type] = metadata[col_type].apply(_fix_sourcesink)
    metadata[col_envname] = metadata[col_envname].fillna('unnamed Environment')
    metadata[col_envname] = metadata[col_envname].apply(lambda x: x.replace(' ', '_'))

    def pre_execute(workdir, args):
        #  check logic of column values
        sinksource = args['metadata'][args['col_type']].value_counts()
        if len(set(sinksource.index) & set(['Sink', 'Source'])) != 2:
            display(sinksource)
            raise ValueError("Column '%s' needs to define for each sample to be either 'Sink' or 'Source' (case sensitive)." % (col_type))

        idx_sources = list(args['metadata'][args['metadata'][args['col_type']] == 'Source'].index)
        for i, (idx, row) in enumerate(args['metadata'][args['metadata'][args['col_type']] == 'Sink'].iterrows()):
            args['counts'].loc[:, idx_sources + [idx]].to_csv('%s/feature-table_%i.txt' % (workdir, i+1), sep="\t", index=True, index_label=False)
            args['metadata'].loc[idx_sources + [idx], [col_envname, col_type]].rename(
                columns={args['col_type']: 'SourceSink', args['col_envname']: 'Env'}).to_csv(
                    '%s/metadata_%i.txt' % (workdir, i+1), sep="\t", index=True, index_label="SampleID", quoting=csv.QUOTE_NONNUMERIC)

    def commands(workdir, ppn, args):
        commands = []

        arr_var = '${%s}' % settings.VARNAME_PBSARRAY if args['metadata'][args['metadata'][args['col_type']] == 'Sink'].shape[0] > 1 else '1'
        commands.append((
            "Rscript --vanilla $CONDA_PREFIX/src/feast/feast_main.R "
            "-m %s/metadata_%s.txt "
            "-c %s/feature-table_%s.txt "
            "-s 0 "
            "-e %i "
            "-r %s/feast.results_%s.csv "
            "> %s/feast_%s.out 2> %s/feast_%s.err") % (workdir, arr_var, workdir, arr_var, args['EM_iterations'], workdir, arr_var, workdir, arr_var, workdir, arr_var))

        return commands

    def post_execute(workdir, args):
        # read feast results from file, one file per sink sample
        results = []
        for i, (idx, row) in enumerate(args['metadata'][args['metadata'][args['col_type']] == 'Sink'].iterrows()):
            results.append(pd.read_csv('%s/feast.results_%s.csv' % (workdir, i+1), sep="\t", index_col=0, names=[idx]))
        # even though source environments can consist of several samples, they don't get grouped by feast. Thus, I here sum their contribution
        results = pd.concat(results, sort=False, axis=1).reset_index().groupby('index').sum()
        results.index.name = "Source"
        results.rename(index={'unknown': 'Unknown'})

        # map every sink-sample to its environment name ...
        # results_env = results.copy().T
        # results_env[args['col_envname']] = list(map(lambda x: args['metadata'].loc[x, args['col_envname']], results.columns))

        # and take the mean for each source over all samples of a env group
        # results_env = results_env.groupby('Env').mean()
        # results_env.index.name = 'Sink'

        return results  # _env.T

    return _executor('feast',
                     {'counts': counts.fillna(0.0),
                      'metadata': metadata[[col_envname,col_type]],
                      'col_envname': col_envname,
                      'col_type': col_type,
                      #'col_envID': col_envID,
                      'EM_iterations': EM_iterations,
                      },
                     pre_execute,
                     commands,
                     post_execute,
                     #post_cache=post_cache,
                     environment=settings.FEAST_ENV,
                     array=metadata[metadata[col_type] == 'Sink'].shape[0],
                     ppn=ppn,
                     pmem=pmem,
                     **executor_args)


def sourcetracker2(counts: pd.DataFrame, metadata: pd.DataFrame,
          col_envname, col_type, samples_per_job: int=4,
          #col_envID=None,
          ppn=4, pmem='4GB', **executor_args):
    """SourceTracker2 for microbial source tracking.

    Paramaters
    ----------
    counts : Pandas.DataFrame
        feature-table
    metadata : Pandas.DataFrame
        metadata for samples of feature-table.
        Need to contain columns for "envname", "type" and "envID".
    col_envname : str
        column name in metadata specifying the environment _name_. Like "infant",
        "adult gut", "soil", ...
    col_type : str
        column name in metadata specifiying if sample is either "Sink" or "Source".
    samples_per_job : int
        Default: 4.
        To parallelize execution, sink samples are split into multiple metadata files (each also holding ALL sources)
        and are executed separately. This variable defined how many sink samples shall be at most in one run.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose, dirty

    Returns
    -------
    pd.DataFrame: feature-table, with cols for samples and rows for source environment contribution.
    """
    # general sanity checks
    (counts, metadata) = sync_counts_metadata(counts, metadata)
    check_column_presents(metadata, [col_envname, col_type])
    def _fix_sourcesink(value):
        if value.lower() == 'sink':
            return 'Sink'
        elif value.lower() == 'source':
            return 'Source'
        return value
    metadata[col_type] = metadata[col_type].apply(_fix_sourcesink)
    metadata[col_envname] = metadata[col_envname].fillna('unnamed Environment')
    metadata[col_envname] = metadata[col_envname].apply(lambda x: x.replace(' ', '_'))

    def pre_execute(workdir, args):
        #  check logic of column values
        sinksource = args['metadata'][args['col_type']].value_counts()
        if len(set(sinksource.index) & set(['Sink', 'Source'])) != 2:
            display(sinksource)
            raise ValueError("Column '%s' needs to define for each sample to be either 'Sink' or 'Source' (case sensitive)." % (col_type))

        pandas2biom('%s/counts.biom' % workdir, args['counts'])
        idx_sources = list(args['metadata'][args['metadata'][args['col_type']] == 'Source'].index)
        idx_sinks = list(args['metadata'][args['metadata'][args['col_type']] == 'Sink'].index)
        for i, chunk in enumerate(range(0, len(idx_sinks), samples_per_job)):
            args['metadata'].loc[idx_sources + idx_sinks[chunk:chunk+samples_per_job], [args['col_type'], args['col_envname']]].to_csv('%s/meta_%i.tsv' % (workdir, (i+1)), sep="\t")

    def commands(workdir, ppn, args):
        commands = []

        arr_var = '${%s}' % settings.VARNAME_PBSARRAY if args['metadata'][args['metadata'][args['col_type']] == 'Sink'].shape[0] > 1 else '1'
        commands.append((
            'sourcetracker2 '
            '--table_fp %s/counts.biom '
            '--mapping_fp %s/meta_%s.tsv '
            '--output_dir %s/result_%s '
            '--jobs %i '
            '--source_rarefaction_depth 0 '
            '--sink_rarefaction_depth 0 '
            '--source_sink_column "%s" '
            '--source_category_column "%s" '
            '--source_column_value "Source" '
            '--sink_column_value "Sink"') % (workdir, workdir, arr_var, workdir, arr_var, ppn, args['col_type'], args['col_envname']))

        return commands

    def post_execute(workdir, args):
        results = []
        for i in range(1, int(np.ceil(args['metadata'][args['metadata'][args['col_type']] == 'Sink'].shape[0] / samples_per_job))+1):
            results.append(pd.read_csv('%s/result_%i/mixing_proportions.txt' % (workdir, i), sep="\t", index_col=0))
        results = pd.concat(results, axis=1, sort=False)
        results.index.name = 'Source'
        results.columns.name = 'Sink'
        return results

    return _executor('sourcetracker2',
                     {'counts': counts.fillna(0.0),
                      'metadata': metadata[[col_envname,col_type]],
                      'col_envname': col_envname,
                      'col_type': col_type,
                      },
                     pre_execute,
                     commands,
                     post_execute,
                     environment=settings.SOURCETRACKER2_ENV,
                     array=int(np.ceil(metadata[metadata[col_type] == 'Sink'].shape[0] / samples_per_job)),
                     ppn=ppn,
                     pmem=pmem,
                     **executor_args)


def pldist(counts: pd.DataFrame, meta: pd.DataFrame, reference_tree: TreeNode, col_time: str, col_entities: str, ppn=1, pmem='5GB', **executor_args):
    """Paired and Longitudinal Ecological Dissimilarities.

    Paramaters
    ----------
    counts: pd.DataFrame
        feature-table
    meta: pd.DataFrame
        metadata
    reference_tree: skbio.TreeNode
        Insertion tree containing all feature-ids as tips
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose, dirty

    Returns
    -------
    {beta: dict(metric: skbio.DistanceMatrix),
     meta_entities: pd.DataFrame}
    """
    counts, meta = sync_counts_metadata(counts, meta)
    check_column_presents(meta, [col_time, col_entities])
    GAMMAS = ["0", "0.5", "1"]

    def pre_execute(workdir, args):
        # write sample metadata
        idx_name = 'level_0'
        if args['meta'].index.name is not None:
            idx_name = args['meta'].index.name
        meta_samples = args['meta'].reset_index().rename(columns={args['col_entities']: 'subjID', args['col_time']: 'time', idx_name: 'sampID'})[["subjID", "sampID", "time"]].sort_values('sampID')
        meta_samples.to_csv('%s/meta_samples.tsv' % workdir, sep="\t", index_label='sample')

        # write feature counts
        nonzero_counts = args['counts'][args['counts'].sum(axis=1) > 0]
        nonzero_counts.T.loc[meta_samples['sampID'].values, :].to_csv('%s/counts.tsv' % workdir, sep="\t", index_label='otu')

        # write tree
        tree = args['reference_tree'].shear(list(nonzero_counts.index))
        tree.prune()
        tree.write('%s/tree.newick' % workdir)

        # generate R code
        with open('%s/pldist.R' % workdir, 'w') as f:
            f.write('library(pldist)\n')
            f.write('library(ape)\n')
            f.write('ggmap.otus <- as.matrix(read.table("%s/counts.tsv", header=TRUE, sep = "\\t", row.names = 1, as.is=TRUE))\n' % workdir)
            f.write('ggmap.meta <- read.table("%s/meta_samples.tsv", header=TRUE, sep = "\\t", row.names = 1)\n' % workdir)
            f.write('ggmap.tree <- read.tree(file="%s/tree.newick")\n' % workdir)
            for clr in ['TRUE', 'FALSE']:
                f.write('res <- pldist(ggmap.otus, ggmap.meta, paired=FALSE, binary=FALSE, method="unifrac", tree=ggmap.tree, gam=c(%s), clr=%s)\n' % (','.join(GAMMAS), clr))
                for gamma in GAMMAS+['UW']:
                    f.write('write.table(res$D[,,"d_%s"], "%s/res_%sCLR_%s.tsv", sep="\t", row.names=TRUE, col.names=NA)\n' % (gamma, workdir, 'no' if clr == 'FALSE' else '', gamma))

    def commands(workdir, ppn, args):
        commands = []

        commands.append('R --vanilla < %s/pldist.R' % workdir)

        return commands

    def post_execute(workdir, args):
        # identify metadata columns that apply to host_subjects not to individual samples
        entity_unique_cols = None
        for entity, g in args['meta'].groupby(args['col_entities']):
            nunique = g.apply(pd.Series.nunique)
            if entity_unique_cols is None:
                entity_unique_cols = set(nunique[nunique == 1].index)
            else:
                entity_unique_cols &= set(nunique[nunique == 1].index)
        # exclude columns that are identical across all host_subjects
        entity_unique_cols -= {col for col, n in args['meta'].apply(pd.Series.nunique).iteritems() if n == 1}
        meta_entities = args['meta'][list(entity_unique_cols)].groupby(args['col_entities']).apply(lambda row: row.iloc[0])

        # load results
        pldist_dms = dict()
        for clr in ['TRUE', 'FALSE']:
            for gamma in GAMMAS+['UW']:
                name = ('noCLR' if clr == 'FALSE' else 'CLR') + '_' + gamma
                pldist_dms[name] = pd.read_csv('%s/res_%sCLR_%s.tsv' % (workdir, 'no' if clr == 'FALSE' else '', gamma), sep="\t", index_col=0)
                pldist_dms[name] = DistanceMatrix(pldist_dms[name], ids=pldist_dms[name].index)

        return {'beta': pldist_dms, 'meta_entities': meta_entities}

    return _executor('pldist',
                     {'counts': counts.fillna(0.0),
                      'meta': meta,
                      'reference_tree': reference_tree,
                      'col_time': col_time,
                      'col_entities': col_entities},
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=ppn,
                     pmem=pmem,
                     walltime="00:59:00",
                     environment=settings.PLDIST_ENV,
                     **executor_args)


def taxonomy(metadata : pd.DataFrame, counts : pd.DataFrame, taxonomy : pd.Series,
             fp_results, infix="", ppn=1, **executor_args):
    """"""

    # general sanity checks
    (counts, metadata) = sync_counts_metadata(counts, metadata)
    def pre_execute(workdir, args):
        # store counts as a biom file
        pandas2biom(workdir+'/input.biom', args['counts'])

        # write metadata to tmp file
        args['metadata'].to_csv(
            workdir+'/metadata.tsv', sep="\t", index_label='sample_name')

        # store taxonomy
        if (len(set(args['counts'].index) - set(args['taxonomy'].index)) > 0):
            raise ValueError("Not all features in your count table are covered by your taxonomy!")
        args['taxonomy'].name = 'Taxon'
        args['taxonomy'].loc[set(args['counts'].index) & set(args['taxonomy'].index)].to_frame().to_csv(
            workdir+'/taxonomy.tsv', sep="\t", index_label='Feature ID')

    def commands(workdir, ppn, args):
        commands = []

        commands.append((
            'qiime tools import '
            '--input-path %s/input.biom '
            '--output-path %s/counts '
            '--type "FeatureTable[Frequency]"') % (workdir, workdir))
        commands.append((
            'qiime tools import '
            '--input-path %s/taxonomy.tsv '
            '--output-path %s/taxonomy.qza '
            '--type "FeatureData[Taxonomy]"') % (workdir, workdir))
        commands.append((
            'qiime taxa barplot '
            '--i-table %s/counts.qza '
            '--i-taxonomy %s/taxonomy.qza '
            '--m-metadata-file %s/metadata.tsv '
            '--o-visualization %s/taxa_barplot.qzv ') % (
            workdir, workdir, workdir, workdir))

        return commands

    def post_execute(workdir, args):
        os.makedirs(fp_results, exist_ok=True)
        fp = os.path.join(fp_results, 'taxa_barplot%s.qzv' % infix)
        shutil.move("%s/taxa_barplot.qzv" % workdir, fp)
        sys.stdout.write("Resulting file moved to %s" % fp)
        return {'filepath': fp}

    return _executor('taxonomy',
                     {'counts': counts.fillna(0.0),
                      'metadata': metadata,
                      'taxonomy': taxonomy},
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=ppn,
                     environment=settings.QIIME2_ENV,
                     **executor_args)


def _update_metric_alpha(metric):
    if metric == 'PD_whole_tree':
        return 'faith_pd'
    return metric


def scnic(counts: pd.DataFrame,
          method: str='sparcc',
          min_reads_per_feature: float=500,
          min_mean_abundance_per_feature: float=2,
          comp_minnumber: int=5,
          comp_mincompsize: int=3,
          ppn=8,
          #col_envname, col_type,
          #col_envID=None,
          #EM_iterations=1000,
          #ppn=4, pmem='4GB',
          **executor_args):
    """q2-SCNIC.

    Paramaters
    ----------
    counts : Pandas.DataFrame
        feature-table
    method : str
        Default: sparcc.
        Choose from 'kendall', 'pearson', 'spearman', 'sparcc'.
    min_reads_per_feature : float
        Default: 500.
        Minimal number of reads per feature to be included for SCNIC analysis.
        First filtering step.
    min_mean_abundance_per_feature : float
        Default: 2.
        Minimal mean abundance of a feature across all samples.
        Second filtering step.
    comp_minnumber : int
        Default: 5.
        Minimal number of graph components to report.
    comp_mincompsize : int
        Default: 3.
        Minimal number of nodes per component.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose, dirty

    Returns
    -------
    ???
    """
    # filter such that only features persist that have more than min_reads_per_feature counts across all samples
    counts = counts[counts.sum(axis=1) >= min_reads_per_feature]
    # filter such that only features persist that have a mean abundance of at least
    counts = counts[counts.mean(axis=1) >= min_mean_abundance_per_feature]
    if 'verbose' in executor_args and executor_args['verbose'] is not None:
        executor_args['verbose'].write('feature-table density: %.1f%%\n' % (
            (1 - (float((counts == 0).sum().sum()) /
            float(counts.shape[0]*counts.shape[1])))*100
        ))

    def pre_execute(workdir, args):
        # filter feature table to avoid too many zeros, following:
        # Correlational analyses are hampered by having large numbers of zeroes.
        # Therefore we are first going to remove these from our data. In the
        # q2-SCNIC plugin a method called sparcc-filter to do this based on
        # the parameters used in Friedman et al. 23 This method removes all
        # samples with a feature abundance total below 500 and all features
        # with an average abundance less than 2 across all samples. You do not
        # need to use these parameters and can use any method you chose to do
        # this. Other methods for filtering feature tables are outlined here.

        # store counts as a biom file
        pandas2biom(workdir+'/input.biom', args['counts'])

    def commands(workdir, ppn, args):
        commands = []

        commands.append(
            ('qiime tools import '
             '--input-path %s '
             '--type "FeatureTable[Frequency]" '
             # '--source-format BIOMV210Format '
             '--output-path %s') %
            (workdir+'/input.biom', workdir+'/input'))

        commands.append(
            ('qiime SCNIC calculate-correlations '
             '--i-table %s/input.qza '
             '--p-method %s '
             '--o-correlation-table %s/correls_%s.qza '
             '--p-n-procs %i') %
            (workdir, args['method'], workdir, args['method'], 8 if ppn > 8 else ppn)
        )

        commands.append(
            ('qiime tools export '
             '--input-path %s/correls_%s.qza '
             '--output-path %s') %
            (workdir, args['method'], workdir))

        return commands

    def post_execute(workdir, args):
        correlations = pd.read_csv('%s/pairwise_comparisons.tsv' % workdir, sep="\t")
        correlations['abs_r'] = correlations['r'].abs()

        pv = pd.pivot_table(data=correlations, values='r', index='feature1', columns='feature2').reindex(args['counts'].index, axis=0).reindex(args['counts'].index, axis=1)
        pv = pv.fillna(0) + pv.fillna(0).T
        for c in pv.columns:
            pv.loc[c, c] = 1
        pv.index.name = method

        return {'correlations': correlations, 'r': pv}

    def post_cache(cache_results):
        G = nx.from_pandas_edgelist(cache_results['results']['correlations'], 'feature1', 'feature2', ['r'])
        G_wanted = None
        # descendingly iterate through absolute correlation values until graph has enough members to form comp_minnumber components of min_comp_size nodes each
        for i, corr_thresh in enumerate(cache_results['results']['correlations']['abs_r'].sort_values(ascending=False).unique()):
            # print(i, corr_thresh)
            G_sub = nx.Graph()
            G_sub.add_weighted_edges_from([(edge[0], edge[1], G[edge[0]][edge[1]]['r']) for edge in G.edges if abs(G[edge[0]][edge[1]]['r']) >= corr_thresh])

            if sum([1 for comp in nx.connected_components(G_sub) if len(comp) >= comp_mincompsize]) > comp_minnumber:
                break
            G_wanted = G_sub.copy()

        # remove components with less than comp_mincompsize nodes
        nodes_to_remove = set()
        for comp in nx.connected_components(G_wanted):
            if len(comp) < comp_mincompsize:
                nodes_to_remove |= comp
        G_wanted.remove_nodes_from(nodes_to_remove)


        # draw graph
        # create node colors
        node_color_map = dict()
        for i, comp in enumerate(nx.connected_components(G_wanted)):
            for n in comp:
                node_color_map[n] = sns.color_palette()[i % len(sns.color_palette())]
            display(cache_results['results']['r'].loc[comp, comp])

        edges, weights = zip(*nx.get_edge_attributes(G_wanted, 'weight').items())

        pos = nx.spring_layout(G_wanted, weight='r', k=0.3)

        nx.draw(G_wanted,
                pos=pos,
                with_labels=True,
                node_color=[node_color_map[node] for node in G_wanted.nodes],
                edgelist=edges,
                edge_color=weights,
                edge_cmap=plt.cm.PuOr,
                width=5.0
               )
        cache_results['results']['graph'] = G_wanted

        return cache_results

    return _executor('scnic',
                     {'counts': counts.fillna(0.0),
                      'method': method,
                      'min_reads_per_feature': min_reads_per_feature,
                      'min_mean_abundance_per_feature': min_mean_abundance_per_feature,
                      },
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache=post_cache,
                     ppn=ppn,
                     environment='qiime2-2019.10',
                     array=1,
                     **executor_args)

def ancom(counts: pd.DataFrame, rank, taxonomy: pd.Series, grouping: pd.Series, min_mean_abundance_per_feature: float=0.005,
          ppn=1, pmem='4GB', **executor_args):
    """Execute ANCOM analysis through Qiime2.

    Paramaters
    ----------
    counts: pd.DataFrame
        The feature table.
    rank: str or (str, str)
        A taxonomic rank to which features shall be collapsed.
        Use 'raw' to avoid collapsing.
        Use a tuple e.g. ('Family', 'Genus') to collapse on merged taxonomy
        ranks.
    taxonomy: pd.Series
        Taxonomic lineages for each feature in counts. Can be None if rank is
        'raw'.
    grouping: pd.Series
        The metadata column on which samples of the feature table shall be
        grouped into exactly two groups.
    min_mean_abundance_per_feature: float
        only report features, that have on average a relative abundance of
        min_mean_abundance_per_feature
    palette: dict, optional
        You can pass a color dictionary to the underlying sns.barplot function
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose,
        dirty

    Returns
    -------
    """

    COL = 'COLANCOM'
    # general sanity checks
    (counts, grouping) = sync_counts_metadata(counts, grouping.dropna())
    if grouping.value_counts().shape[0] < 2:
        raise ValueError("Your grouping information has only ONE group. It must be exactly two!")
    if grouping.value_counts().shape[0] > 2:
        raise ValueError("Your grouping information has more than two groups. It must be exactly two!")

    if rank is None:
        raise ValueError("Rank cannot be empty, choose from %s or set to 'raw'." % settings.RANKS)
    if (taxonomy is None) and rank != 'raw':
        raise ValueError("You must provide a taxonomy pd.Series if 'rank' is not 'raw'!")
    counts = collapseCounts_objects(counts, rank, taxonomy)[0]

    def pre_execute(workdir, args):
        def _is_numeric(element):
            try:
                float(element)
                return True
            except ValueError:
                return False

        pandas2biom('%s/counts.biom' % workdir, args['counts'])
        args['grouping'].name = COL
        args['grouping'] = args['grouping'].apply(lambda x: '_%s' % x if _is_numeric(x) else x)
        args['grouping'].to_csv('%s/grouping.tsv' % workdir, sep="\t", index_label="sample_name")

    def commands(workdir, ppn, args):
        commands = []

        commands.append(
            ('qiime tools import '
             '--input-path %s '
             '--type "FeatureTable[Frequency]" '
             '--output-path %s') %
            (workdir+'/counts.biom', workdir+'/counts'))
        commands.append(
            ('qiime composition add-pseudocount '
             '--i-table %s '
             '--o-composition-table %s ') %
            (workdir+'/counts.qza', workdir+'/counts_pseudo.qza'))
        commands.append(
            ('qiime composition ancom '
             '--i-table %s '
             '--m-metadata-file %s '
             '--m-metadata-column %s '
             '--o-visualization %s ') %
            (workdir+'/counts_pseudo.qza', workdir+'/grouping.tsv', COL, workdir+'/ancom.qzv'))
        commands.append(
            ('qiime tools export '
             '--input-path %s/ancom.qzv '
             '--output-path %s/ancom_results/')
            % (workdir, workdir))

        return commands

    def post_execute(workdir, args):
        results = dict()
        results['ancom'] = pd.read_csv('%s/ancom_results/ancom.tsv' % workdir,
                                       sep="\t", index_col=0)
        results['data'] = pd.read_csv('%s/ancom_results/data.tsv' % workdir,
                                      sep="\t", index_col=0)
        results['percent_abundances'] = pd.read_csv('%s/ancom_results/percent-abundances.tsv' % workdir,
                                                    sep="\t", index_col=0, header=[0,1])
        # normalize feature table
        feat = args['counts'] / args['counts'].sum()
        # restructure data
        feat = feat.stack().reset_index().rename(columns={'level_1': 'sample_name', 0: 'relAbundance'})

        merge_on = args['counts'].index.name
        if rank == 'raw':
            feat = feat.rename(columns={'level_0': 'raw'})
            if merge_on is None:
                merge_on = 'raw'

        # assign grouping values
        feat = feat.merge(args['grouping'].to_frame(), left_on='sample_name', right_index=True, how='left')
        # add 'Reject null hypothesis'

        feat = feat.merge(results['ancom'][['Reject null hypothesis']], left_on=merge_on, right_index=True, how='left')
        results['features'] = feat

        results['col_feature'] = merge_on
        results['col_grouping'] = args['grouping'].name

        return results

    def post_cache(cache_results, palette=None, feature_order=None, hue_order=None):
        feat_sigdiff = cache_results['results']['ancom'][cache_results['results']['ancom']['Reject null hypothesis']].index

        data = cache_results['results']['features'][
            cache_results['results']['features'][cache_results['results']['col_feature']].isin(feat_sigdiff)]

        srt_feat = data.groupby(
            cache_results['results']['col_feature'])['relAbundance'].mean(numeric_only=True).sort_values(ascending=False)

        report_taxa = cache_results['results']['ancom'].copy()
        report_taxa = report_taxa.merge((cache_results['results']['features'].groupby(cache_results['results']['col_feature'])['relAbundance'].mean(numeric_only=True) >= min_mean_abundance_per_feature).to_frame(), left_index=True, right_index=True, how='left')
        report_taxa = report_taxa.groupby(['Reject null hypothesis', 'relAbundance']).size().to_frame().reset_index().sort_values(by=['Reject null hypothesis', 'relAbundance'], ascending=[False, False]).rename(columns={
            0: '#%s' % cache_results['results']['col_feature'],
            'relAbundance': '>= %f mean rel. abundance' % min_mean_abundance_per_feature,
            'Reject null hypothesis': 'significantly different'
        })
        display(report_taxa)

        srt_feat = srt_feat[srt_feat >= min_mean_abundance_per_feature]
        if report_taxa[report_taxa['significantly different'] & report_taxa['>= %f mean rel. abundance' % min_mean_abundance_per_feature]].shape[0] > 0:
            fig, axes = plt.subplots(1,1,figsize=(5, 0.4 * report_taxa[report_taxa['significantly different'] & report_taxa['>= %f mean rel. abundance' % min_mean_abundance_per_feature]].iloc[0,-1]))
            cache_results['plot'] = sns.barplot(
                data=cache_results['results']['features'][cache_results['results']['features']['Reject null hypothesis']],
                x='relAbundance', y=cache_results['results']['col_feature'], hue=cache_results['results']['col_grouping'],
                order=srt_feat.index if feature_order is None else feature_order,
                hue_order=hue_order,
                orient='h', ax=axes, palette=palette)
            cache_results['results']['figure'] = fig

        cache_results['results']['reported_features'] = srt_feat
        cache_results['results']['summary'] = report_taxa

        return cache_results

    return _executor('ancom',
                     {'counts': counts.fillna(0.0),
                      'grouping': grouping.sort_values(),
                      },
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache,
                     environment=settings.QIIME2_ENV,
                     ppn=ppn,
                     pmem=pmem,
                     **executor_args)

def tempted(counts: pd.DataFrame, sample_metadata: pd.DataFrame,
            pivot_samples: pd.Series, fp_results: str, infix: str="",
            col_subject: str='host_subject_id', col_collection_date: str='collection_timestamp',
            drop_samples_lessThanCounts: int=1, drop_feature_lessInRatioSamples: float=0.05, drop_subjects_lessTimepoints: int=2,
            apply_clr: bool=True, apply_svd: bool=True,
            components: int=5,
            ppn=1, pmem='4GB', **executor_args):
    """Run tempted analysis and return Emperor plots.

    Parameters
    ----------
    counts : pd.DataFrame
        Feature in rows x Samples in columns read counts.
    sample_metadata : pd.DataFrame
        Metadata about samples (not about subjects, i.e. multiple samples have the same subject)
    pivot_samples: pd.Series
        A pd.Series that holds one sample_name per subject to point to the sample/timepoint that shall form the baseline, i.e. time point zero.
        Used to compute the delta between each timepoint and this sample.
    fp_results: str
        Filepath for generated Emperor plot.
    infix: str
        Infix for emperor file name.
    col_subject: str
        the column name in the metadata that gives the subject name, i.e. used as grouper for multiple timepoints per subject
    col_collection_date: str
        the column name in the metadata that holds an actual date of sample collections. These dates will be used to compute time deltas.
    drop_samples_lessThanCounts: int
        Samples with less than this number of feature counts in total will be dropped.
    drop_feature_lessInRatioSamples: float
        Features that occure in less than this percentage of samples will be dropped.
    drop_subjects_lessTimepoints: int
        Subjects with less than this number of samples/timepoints will be dropped.
    apply_clr: bool
        Transform feature counts?
    apply_svd: bool
        Centralize feature counts?
    components: int
        Number of components for tempted computation = PC axes?!
    """
    meta = sample_metadata.copy()
    counts_internal = counts.copy()

    COL = 'COLTEMPTED'
    if COL in meta.columns:
        raise ValueError("Column named %s already found in your sample_metadata!" % COL)

    # general sanity checks
    check_column_presents(meta, [col_subject, col_collection_date])
    #(counts, sample_metadata) = sync_counts_metadata(counts, sample_metadata.dropna(subset=[col_subject, col_collection_date]))
    if len(set(pivot_samples.values) - set(meta.index)):
        raise ValueError("Not all samples in your pivot_samples are included in sample_metadata (after syncing with counts)! Missing samples are:\n  - %s" % '\n  - '.join(sorted(list(set(pivot_samples.values) - set(meta.index)))))

    if 'verbose' in executor_args and executor_args['verbose'] is not None:
        executor_args['verbose'].write('You start tempted with    %i subjects, %i samples and %i features.\n' % (len(meta[col_subject].unique()), counts.shape[1], counts.shape[0]))

    # filter out samples with less than drop_samples_lessThanCounts reads
    counts_internal = counts_internal.T[counts_internal.sum() >= drop_samples_lessThanCounts].T

    # filter ASVs not occuring in at least drop_feature_lessInRatioSamples% of all samples
    asvs = [asv for asv, numsamples in (counts_internal > 0).sum(axis=1).items() if numsamples >= counts_internal.shape[1] * drop_feature_lessInRatioSamples]
    counts_internal = counts_internal.loc[asvs, :]

    # compute actual time deltas per subject
    for subject, g in meta.groupby(col_subject):
        idx_baseline = pivot_samples.loc[subject]
        if isinstance(idx_baseline, pd.Series):
            # multiple samples point to the baseline!!
            # check if both samples point to the very same date
            dates = meta.loc[idx_baseline.values, col_collection_date].unique()
            if len(dates) != 1:
                raise ValueError("Your pivot_samples hold multiple samples for subject %s with different dates!!\n%s" % (subject, dates))
            idx_baseline = idx_baseline.iloc[0]
        try:
            # time might be given as date or as float, try date first ...
            _baseline = pd.to_datetime(str(meta.loc[idx_baseline, col_collection_date]))
        except ValueError as e:
            # ... resort to float later
            _baseline = float(meta.loc[idx_baseline, col_collection_date])

        x = None
        try:
            # time might be given as date or as float, try date first ...
            times = g[col_collection_date].apply(lambda x: pd.to_datetime(str(x)))
            x = (times - _baseline).dt.days.apply(float)
        except ValueError as e:
            # ... resort to float later
            x = g[col_collection_date].apply(float) - _baseline
        meta.loc[x.index, COL] = x.loc[x.index]

    # filter out subjects with less than drop_subjects_lessTimepoints samples
    (counts_internal, meta) = sync_counts_metadata(counts_internal, meta.dropna(subset=[col_subject, col_collection_date]), verbose=None)
    subjects = [hsid for (hsid, num_samples) in meta.groupby(col_subject).size().items() if num_samples >= drop_subjects_lessTimepoints]
    meta = meta[meta[col_subject].isin(subjects)]
    (counts_internal, meta) = sync_counts_metadata(counts_internal, meta.dropna(subset=[col_subject, col_collection_date]), verbose=None)

    if 'verbose' in executor_args and executor_args['verbose'] is not None:
        executor_args['verbose'].write('After filtering, you have %i subjects, %i samples and %i features.\n' % (len(meta[col_subject].unique()), counts_internal.shape[1], counts_internal.shape[0]))

    meta_missing = meta[pd.isnull(meta[COL])]
    if meta_missing.shape[0] > 0:
        raise ValueError("There are %i samples in %i subjects having NaN in metadata-column '%s'!" % (meta_missing.shape[0], len(meta_missing[col_subject].unique())), col_collection_date)

    def pre_execute(workdir, args):
        m_sorted = args['sample_metadata'].sort_values(by=[col_subject, COL])
        m_sorted.to_csv('%s/sample_metadata.csv' % workdir, sep=",", index_label="sample_name")
        args['counts'].loc[:, m_sorted.index].T.to_csv('%s/counts_T.csv' % workdir, sep=",", index_label="feature")

        old_index = args['sample_metadata'].index.name
        if old_index is None:
            old_index = 'index'
        m = args['sample_metadata'].groupby(col_subject).head(1).reset_index()
        del m[old_index]
        m.set_index(col_subject).rename(columns={'sample_name': 'old_sample_name'}).to_csv('%s/subject_metadata.tsv' % workdir, sep="\t", index_label="sample_name")

        with open('%s/R.script' % workdir, 'w') as R:
            R.write("# for data \n")
            R.write("library(readr) # read tsv\n")
            R.write("#library(qiime2R) # read in Qiime artifacts\n")
            R.write("library(dplyr) # data formatting\n")
            R.write("library(yaml) # for read_qza() in qiime2R\n")
            R.write("library(tidyr)\n\n")
            R.write("# for computing\n")
            R.write("library(reticulate) # run py codes\n")
            R.write("library(phyloseq) # phyloseq object\n")
            R.write("library(vegan) # distance matrix\n")
            R.write("library(PERMANOVA) # permanova\n")
            R.write("library(randomForest) # random forest\n")
            R.write("library(PRROC) # roc and pr\n")
            R.write("library(tempted)\n\n")
            R.write("library(microTensor)\n")
            R.write("# for plotting\n")
            R.write("library(ggpubr)\n")
            R.write("library(ggplot2)\n")
            R.write("library(gridExtra)\n")
            R.write("library(RColorBrewer)\n")
            R.write("library(plotly)\n\n")

            R.write("# load data\n")
            R.write("meta <- read.csv(\"%s/sample_metadata.csv\", row.names=1)\n" % workdir)
            R.write("counts <- read.csv(\"%s/counts_T.csv\", row.names=1)\n\n" % workdir)
            R.write("# run tempted\n")
            R.write("datlist <- format_tempted(counts, meta$%s, meta$%s, threshold=%f, pseudo=0.5, transform='%s')\n" % (
                COL, col_subject, 1-drop_feature_lessInRatioSamples, 'clr' if apply_clr else 'none'))
            if apply_svd:
                R.write("svd <- svd_centralize(datlist, 1)\n")
            else:
                R.write("svd <- datlist\n")
            R.write("res_tempted <- tempted(svd$datlist, r = %s, resolution = 101, smooth=1e-6)\n\n" % components)

            R.write("# export data\n")
            R.write("write.table(res_tempted$A_hat, file=\"%s/pcs.csv\", quote=FALSE, sep=\"\\t\", col.names=NA)\n" % workdir)
            R.write("write.table(res_tempted$Lambda, file=\"%s/lambda.csv\", col.names=\"Lambda\", sep=\",\")\n" % workdir)
            R.write("write.table(res_tempted$r_square, file=\"%s/r_square.csv\", col.names=\"r_square\", sep=\",\")\n" % workdir)

        # dry = executor_args['dry'] if 'dry' in executor_args else True
        # cluster_run(['cat %s/R.script | R --vanilla' % workdir], environment=settings.TEMPTED_ENV,
        #             jobname='R_tempted',
        #             result="%s/r_square.csv" % workdir,
        #             ppn=1,
        #             pmem=pmem,
        #             walltime='1:00:00',
        #             dry=dry,
        #             wait=True, use_grid=executor_args.get('use_grid', True))

    def commands(workdir, ppn, args):
        commands = []

        # execute the R tempted program in it's own conda env
        commands.append(get_conda_activate_cmd(executor_args.get('use_grid', True), settings.TEMPTED_ENV))
        commands.append('cat %s/R.script | R --vanilla' % workdir)
        #commands.append('%s cat %s/R.script | R --vanilla' % (get_conda_activate_cmd(executor_args.get('use_grid', True), settings.TEMPTED_ENV), workdir))

        # collect tempted results and create an PCoA like file
        commands.extend([
            'echo "Eigvals\t%i" > %s/ordination.txt' % (components, workdir),
            'tail -n +2 %s/lambda.csv | cut -f 2 -d "," | xargs | tr " " "\t" >> %s/ordination.txt' % (workdir, workdir),
            'echo "" >> %s/ordination.txt' % workdir,
            'echo "Proportion explained\t%i" >> %s/ordination.txt' % (components, workdir),
            'tail -n +2 %s/r_square.csv | cut -f 2 -d "," | xargs | tr " " "\t" >> %s/ordination.txt' % (workdir, workdir),
            'echo "" >> %s/ordination.txt' % workdir,
            'echo "Species\t0\t0" >> %s/ordination.txt' % workdir,
            'echo "" >> %s/ordination.txt' % workdir,
            'echo "Site\t%i\t%i" >> %s/ordination.txt' % (len(subjects), components, workdir),
            'tail -n +2 %s/pcs.csv >> %s/ordination.txt' % (workdir, workdir),
            'echo "" >> %s/ordination.txt' % workdir,
            'echo "Biplot\t0\t0" >> %s/ordination.txt' % workdir,
            'echo "" >> %s/ordination.txt' % workdir,
            'echo "Site constraints\t0\t0" >> %s/ordination.txt' % workdir,
        ])

        commands.append(get_conda_activate_cmd(executor_args.get('use_grid', True), settings.QIIME2_ENV))

        commands.append('qiime tools import --input-path %s/ordination.txt --output-path %s/ordination.qza --type PCoAResults' % (workdir, workdir))

        commands.append('qiime emperor plot --i-pcoa %s/ordination.qza --m-metadata-file %s/subject_metadata.tsv  --o-visualization %s/emperor_tempted%s.qzv' % (workdir, workdir, workdir, infix))

        return commands

    def post_execute(workdir, args):
        results = dict()

        os.makedirs(fp_results, exist_ok=True)
        shutil.copy(
            "%s/emperor_tempted%s.qzv" % (workdir, infix), fp_results)
        results['emperor'] = fp_results

        results['ordination'] = OrdinationResults.read(open('%s/ordination.txt' % workdir))

        results['subject_metadata'] = pd.read_csv('%s/subject_metadata.tsv' % workdir, sep="\t", index_col=0)

        return results

    return _executor('tempted',
                     {'counts': counts_internal.fillna(0.0),
                      'sample_metadata': meta,
                      },
                     pre_execute,
                     commands,
                     post_execute,
                     environment=settings.QIIME2_ENV,
                     ppn=ppn,
                     pmem=pmem,
                     **executor_args)


def decontam(counts_raw: pd.DataFrame, sample_metadata: pd.DataFrame, taxonomy: pd.Series,
             col_concentration: str, col_sampletype: str="sample_type", name_control_sample: str="blank",
             cols_batch: [str]=[],
             threshold: float=0.5,
             rank: str='Genus',
             ppn=1, pmem='4GB', environment=settings.QIIME2_ENV, **executor_args):
    """Run decontam to remove kitome contamination in low biomass samples. See https://doi.org/10.1186/s40168-018-0605-2
    Also see: https://jordenrabasco.github.io/Q2_Decontam_Tutorial.html
    - does not handle cross contamination / spill over
    - best use with DNA concentrations AND blanks (= negative control samples)
    - "five to six negative control samples are sufficient to identify most contamination" quote from above paper
    - consider batch effects, i.e. diff runs / diff plates
    - apply per sample type: "we recommend applying decontam independently to samples collected from different environments" quote from above paper

    Parameters
    ----------
    counts_raw : pd.DataFrame
        Raw reads! i.e. prior to rarefaction, normalization. Feature in rows x Samples in columns read counts.
    sample_metadata : pd.DataFrame
        Metadata about samples.
    col_concentration : str
        the column name in the metadata that gives DNA concentrations prior library construction. Used to find inverse correlations of contaminant features.
        If None, decontam falls back to the "prevalence" mode (only using blanks), instead of "combined".
    col_sample_type : str
        the column name in the metadata that tells decontam if a sample is a negative control or true biological sample.
    cols_batch : [str]
        List of column names in the metadata, which will be used to group samples into batches.
    name_control_sample: str
        the 'name' of control samples in col_sample_type
    threshold: float
        this indicates the metadata column which will subset the input table
    """
    meta = sample_metadata.copy()
    counts_internal = counts_raw.copy()

    # general sanity checks
    if type(cols_batch) != list:
        cols_batch = [cols_batch]
    check_column_presents(meta, [col_concentration, col_sampletype] + cols_batch)

    # sync counts and metadata + drop samples lacking col information
    (counts_internal, meta) = sync_counts_metadata(counts_internal, meta[([col_concentration] if col_concentration is not None else []) + [col_sampletype] + cols_batch].dropna(axis=0, how='any'))

    # test if all values can be interpreted as concentrations
    if col_concentration is not None:
        meta[col_concentration] = meta[col_concentration].astype(float)
        if meta[col_concentration].min() < 0:
            raise ValueError("Some concentrations are negative!")

    env_types = meta[col_sampletype].unique()
    if len(env_types) > 2:
        raise ValueError("You have %i different sample types: '%s'. Better use decontam per environment and/or remove samples that you don't want to subject to decontam!" % (len(env_types), "', '".join(env_types)))
    if len(env_types) == 1:
        raise ValueError("You only provide samples that you consider blanks, but not a single biological samples!")
    if name_control_sample not in env_types:
        raise ValueError("The identifier '%s' for your negative control samples cannot be found in the the metadata!" % name_control_sample)

    grps = [('all', meta)]
    if cols_batch != []:
        grps = []
        batch_sizes = dict()
        cb = cols_batch
        if len(cols_batch) == 1:
            cb = cols_batch[0]
        for n, g in meta.groupby(cb):
            batch_sizes[n] = g.shape[0]
            batch_controls = g[g[col_sampletype] == name_control_sample]
            if batch_controls.shape[0] < 1:
                raise ValueError("Your batch %s lacks negative control samples!" % str(n))
            if batch_controls.shape[0] == g.shape[0]:
                raise ValueError("Your batch %s consists of only control samples!" % str(n))
            grps.append((n, g))
        empty_batch_sizes = {n: s for (n, s) in batch_sizes.items() if s <= 0}
        if len(empty_batch_sizes) > 0:
            raise ValueError("%i of your batch groups have zero samples!" % len(empty_batch_sizes))

        if 'verbose' not in executor_args:
            verbose = sys.stderr
        else:
            verbose = executor_args['verbose']
        if verbose:
            verbose.write("Decontam: splitting into %i batches:\n  - %s\n" % (len(grps), "\n  - ".join([str(n) for (n, g) in grps])))

    def pre_execute(workdir, args):
        with open('%s/rep-seq.fasta' % (workdir), 'w') as f:
            for seq in args['counts'].index:
                f.write('>%s\n%s\n' % (seq, seq))

        for i, (batchname, g) in enumerate(grps):
            grp_counts = args['counts'].loc[:, g.index]
            grp_counts = grp_counts[grp_counts.sum(axis=1) > 0]
            pandas2biom('%s/counts_%i.biom' % (workdir, i), grp_counts)

            g.to_csv('%s/metadata_%i.tsv' % (workdir, i), sep="\t", index_label="sample_name")

    def commands(workdir, ppn, args):
        commands = []

        commands.append(
            ('qiime tools import '
             '--input-path %s '
             '--type "FeatureData[Sequence]" '
             '--output-path %s') %
            ('%s/rep-seq.fasta' % (workdir), '%s/rep-seq.qza' % (workdir)))

        mode, param_concentrations = 'combined', '--p-freq-concentration-column "%s"' % col_concentration
        if col_concentration is None:
            mode, param_concentrations = 'prevalence', ''
        for i, (batchname, g) in enumerate(grps):
            commands.append(
                ('qiime tools import '
                 '--input-path %s '
                 '--type "FeatureTable[Frequency]" '
                 '--output-path %s') %
                ('%s/counts_%i.biom' % (workdir, i), '%s/counts_%i' % (workdir, i)))

            commands.append(
                ('qiime quality-control decontam-identify '
                 '--i-table %s '
                 '--m-metadata-file %s '
                 '--p-method %s '
                 ' %s '
                 '--p-prev-control-column "%s" '
                 '--p-prev-control-indicator "%s" '
                 '--o-decontam-scores %s') %
                ('%s/counts_%i.qza' % (workdir, i),
                 '%s/metadata_%i.tsv' % (workdir, i),
                 mode,
                 param_concentrations,
                 col_sampletype,
                 args['name_control_sample'],
                 '%s/decontam-score_%i.qza' % (workdir, i)))

            commands.append(
                ('qiime tools export '
                 '--input-path %s/decontam-score_%i.qza '
                 '--output-path %s/decontam_results_%i/')
                % (workdir, i, workdir, i))

        return commands

    def post_execute(workdir, args):
        results = []
        batch_counts = []
        stats = []
        for i, (batchname, g) in enumerate(grps):
            batch_counts.append(biom2pandas('%s/counts_%i.biom' % (workdir, i)))
            res_decontam = pd.read_csv('%s/decontam_results_%i/stats.tsv' % (workdir, i), sep="\t", index_col=0)
            res = res_decontam.merge(batch_counts[i].sum(axis=1).rename('reads'), left_index=True, right_index=True)
            for (colname, value) in zip(cols_batch, list(batchname)):
                res[colname] = value
            res['batch_grp'] = i
            results.append(res)

            stat = {
                'affected_samples': batch_counts[i].columns,
                'total_num_asvs': res.shape[0],
                'total_num_reads': res['reads'].sum(),
                'non_plotted_num_reads': res[pd.isnull(res['p'])]['reads'].sum(),
                'batch_name': batchname,
                'batch_grp': i,
                'num_control_samples': g[g[col_sampletype] == name_control_sample].shape[0],
                'name_control_samples': name_control_sample,
                'num_biol_samples': g[g[col_sampletype] != name_control_sample].shape[0],
                'name_biol_samples': g[g[col_sampletype] != name_control_sample][col_sampletype].unique()[0]
                }
            for (colname, value) in list(zip(cols_batch, batchname)):
                stat.update({colname: value})
            stats.append(stat)
        stats = pd.DataFrame(stats)

        return {'decontam': pd.concat(results), 'batch_counts': batch_counts, 'stats': stats}

    def post_cache(cache_results):
        if 'verbose' not in executor_args:
            verbose = sys.stderr
        else:
            verbose = executor_args['verbose']

        # compute these stats dynamically, to account for changing threshold
        for i, (batchname, g) in enumerate(grps):
            res = cache_results['results']['decontam'][cache_results['results']['decontam']['batch_grp'] == i]
            idx_batch = cache_results['results']['stats'][cache_results['results']['stats']['batch_grp'] == i].index[0]
            cache_results['results']['stats'].at[idx_batch, 'lost_asvs'] = res[res['p'] < threshold].index
            cache_results['results']['stats'].loc[idx_batch, 'lost_num_asvs'] = len(res[res['p'] < threshold].index)
            cache_results['results']['stats'].loc[idx_batch, 'percent_lost_asvs'] = len(res[res['p'] < threshold].index) / res.shape[0] * 100
            cache_results['results']['stats'].loc[idx_batch, 'lost_num_reads'] = res[res['p'] < threshold]['reads'].sum()
            cache_results['results']['stats'].loc[idx_batch, 'percent_lost_reads'] = res[res['p'] < threshold]['reads'].sum() / res['reads'].sum() * 100

        if verbose:
            verbose.write('  loosing %i of %i = %.2f%% features\n  loosing %i of %i = %.2f%% reads\n' % (
                cache_results['results']['stats']['lost_num_asvs'].sum(), cache_results['results']['stats']['total_num_asvs'].sum(), cache_results['results']['stats']['lost_num_asvs'].sum() / cache_results['results']['stats']['total_num_asvs'].sum() * 100,
                cache_results['results']['stats']['lost_num_reads'].sum(), cache_results['results']['stats']['total_num_reads'].sum(), cache_results['results']['stats']['lost_num_reads'].sum() / cache_results['results']['stats']['total_num_reads'].sum() * 100))

        SIZE = 5
        fig, axes = plt.subplots(len(grps), 3, figsize=(SIZE * 3, SIZE * len(grps)))
        fig.subplots_adjust(hspace=0.4, wspace=0.5)

        num_bins = 50
        tax_colors = dict()
        for i, (batchname, g) in enumerate(grps):
            stats = cache_results['results']['stats'][cache_results['results']['stats']['batch_grp'] == i]

            res = cache_results['results']['decontam'][cache_results['results']['decontam']['batch_grp'] == i]
            ax = None
            if len(grps) > 1:
                ax = axes[i][0]
            else:
                ax = axes[0]
            for tcont in ['contaminant', 'non-contaminant']:
                data = res[res['p'] < threshold]
                if tcont == 'non-contaminant':
                    data = res[res['p'] >= threshold]
                _ = ax.hist(data['p'], bins=num_bins, range=(0, 1), weights=data['reads'], log=True, facecolor='red' if tcont == 'contaminant' else 'blue' , rwidth=0.88, label=tcont[0].upper() + tcont[1:] + ' Reads')
            ax.axvline(x=threshold, label='Threshold', color='black', linestyle='dashed')
            ax.set_ylabel('Number of Reads')
            ax.set_xlabel('Score')
            if i == 0:
                ax.legend()
            ax.set_title('%s\n%i Non-Contaminant reads with p=NA (not plotted)' % (str(batchname), stats['non_plotted_num_reads'].iloc[0]))

            ax = None
            if len(grps) > 1:
                ax = axes[i][1]
            else:
                ax = axes[1]
            data = cache_results['results']['decontam'][(cache_results['results']['decontam']['batch_grp'] == i) & pd.notnull(cache_results['results']['decontam']['p'])]
            sum_reads = data['reads'].sum()
            lost = []
            for mybin in np.linspace(0, 1, num=num_bins + 1):
                cont = data[data['p'] < mybin]
                lost.append({
                    'accum_reads': cont['reads'].sum(),
                    'accum_asvs': cont.shape[0],
                    'accum_norm_reads': cont['reads'].sum() / sum_reads,
                    'accum_norm_asvs': cont.shape[0] / data.shape[0],
                    'max_p': mybin, 'type': 'contaminant'})
            lost = pd.DataFrame(lost)
            ln_reads = ax.plot(lost['max_p'].values, lost['accum_reads'], label='Number of Reads', color='orange')
            ax.set_ylabel('Number of Reads')
            rax = ax.twinx()
            ln_asvs = rax.plot(lost['max_p'].values, lost['accum_asvs'], label='Number of ASVs', color='green')
            rax.set_ylabel('Number of ASVs')
            ln_thresh = ax.axvline(x=threshold, label='Threshold', color='black', linestyle='dashed')
            comb = ln_reads + ln_asvs
            ax.legend(comb, [l.get_label() for l in comb], loc=0)
            ax.set_xlabel("Score")
            ax.set_title('loosing %i of %i = %.2f%% features\nloosing %i of %i = %.2f%% reads' % (
                stats['lost_num_asvs'].iloc[0], stats['total_num_asvs'].iloc[0], stats['percent_lost_asvs'].iloc[0],
                stats['lost_num_reads'].iloc[0], stats['total_num_reads'].iloc[0], stats['percent_lost_reads'].iloc[0]
            ))

            ax = None
            if len(grps) > 1:
                ax = axes[i][2]
            else:
                ax = axes[2]
            taxonomy_cont = taxonomy.copy()
            taxa_contaminants = set(data[data['p'] < threshold].index)
            taxonomy_cont.loc[list(set(taxonomy_cont.index) - taxa_contaminants)] = '; '.join(['%s__non-contaminant' % r[0].lower().replace('k', 'd') for r in settings.RANKS])
            taxa_contaminants = [t for t in collapseCounts_objects(cache_results['results']['batch_counts'][i], rank, taxonomy_cont, out=None)[0].index if '__non-contaminant' not in t]
            _, _, _, _, tax_colors = plotTaxonomy(cache_results['results']['batch_counts'][i], meta.loc[cache_results['results']['batch_counts'][i].columns, :], rank=rank, file_taxonomy=taxonomy_cont, ax=ax, minreadnr=1, plottaxa=taxa_contaminants, out=None, colors=tax_colors, group_l1=col_sampletype)
            # ax.set_title('using %i "%s" samples' % (stats['num_control_samples'], stats['name_control_samples'].values[0]))
            # ax.set_xlabel('%i "%s" samples' % (stats['num_biol_samples'], stats['name_biol_samples'].values[0]))

        cache_results['figure'] = fig

        return cache_results

    return _executor('decontam',
                     {'counts': counts_internal.fillna(0.0),
                      'sample_metadata': meta,
                      'name_control_sample': name_control_sample,
                      },
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache,
                     environment=environment,
                     ppn=ppn,
                     pmem=pmem,
                     **executor_args)

def QC(dir_fastqs:str,
       pattern_fwdfiles:str="*_R1_*.fastq.gz",
       r1r2_replace:(str, str)=("_R1_", "_R2_"),
       no_rev_seqs:bool=False,
       ppn:int=1, pmem:str='2GB', environment:str=settings.SPIKE_ENV,
       **executor_args):
    """Generates multiQC reports for all fastq files in a directory.

    Parameters
    ----------
    dir_fastqs : str
        Filepath to directory which contains one or more Sequencing fastq files.
    pattern_fwdfiles : str
        Unix pattern identify fastq files.
    r1r2_replace : (str, str)
        Source and Target infix to instruct how matching reverse fastQ files
        can be identified from forward fastQ files.
    no_rev_seqs : Boolean
        If True, no R2 (=reverse) reads are expected, i.e. processed
    """
    files_fwd = []
    files_rev = []
    for fp_fastq in sorted(glob(os.path.join(dir_fastqs, "**", pattern_fwdfiles), recursive=True)):
        if os.path.basename(fp_fastq).startswith('Undetermined_S0_'):
            continue

        files_fwd.append(os.path.abspath(fp_fastq))
        if not no_rev_seqs:
            fp_rev = os.path.join(os.path.dirname(fp_fastq), os.path.basename(fp_fastq).replace(r1r2_replace[0], r1r2_replace[1]))
            if not os.path.exists(fp_rev):
                raise ValueError('Cannot find reverse file for forward file:\n  fwd: %s\n   rev: %s\nCheck r1r2_replace pattern!' % (fp_fastq, fp_rev))
            files_rev.append(os.path.abspath(fp_rev))
    if (len(files_fwd) + len(files_rev)) <= 0:
        raise ValueError("No fwd fastQ files found. Check dir_fastqs and pattern_fwdfiles!")
    if (not no_rev_seqs) & (len(files_fwd) != len(files_rev)):
        raise ValueError("Number of fwd (%i) and rev (%i) files no not match!" % (len(files_fwd), len(files_rev)))

    def pre_execute(workdir, args):
        with open("%s/commands.txt" % workdir, "w") as f:
            os.makedirs('%s/forward/' % workdir, exist_ok=True)
            for fp_fwd in files_fwd:
                f.write('fastqc --noextract --outdir %s --threads 1 %s\n' % (
                    '%s/forward/' % workdir, fp_fwd))
            if not args['no_rev_seqs']:
                os.makedirs('%s/reverse/' % workdir, exist_ok=True)
                for fp_rev in files_rev:
                    f.write('fastqc --noextract --outdir %s --threads 1 %s\n' % (
                        '%s/reverse/' % workdir, fp_rev))

    def commands(workdir, ppn, args):
        commands = {'pre': [], 'main': [], 'post': []}

        commands['main'] = [
            'var_fastqcCMD=`head -n ${%s} %s/commands.txt | tail -n 1 | cut -f 1`' % (
                settings.VARNAME_PBSARRAY, workdir),
            '$var_fastqcCMD'
             ]

        directions = ['forward']
        if not args['no_rev_seqs']:
            directions.append('reverse')
        for direction in directions:
            fp_outdir = '%s/%s/multiqc_data/' % (workdir, direction)
            fp_report = '%s/%s/multiqc_report.html' % (workdir, direction)
            commands['post'].append('multiqc --filename %s --outdir %s --flat %s/%s' % (
                fp_report, fp_outdir, workdir, direction
            ))

        return commands

    def post_execute(workdir, args):
        dry = executor_args['dry'] if 'dry' in executor_args else True

        directions = ['forward']
        if not args['no_rev_seqs']:
            directions.append('reverse')

        results = dict()
        fps_reports = glob('%s/**/multiqc_report.html' % workdir, recursive=True)
        if len(fps_reports) < 1:
            raise ValueError("Cannot find any multiqc HTML report. Something must be broken. Please inspect your log files and/or re-execute.")
        for fp_report in fps_reports:
            direction = os.path.dirname(fp_report).split('/')[-1]
            with open(fp_report, 'r') as f:
                # save full html report here ...
                results['multiqc_%s' % direction] = ''.join(f.readlines())

        return results

    def post_cache(cache_results):
        for direction in ['forward', 'reverse']:
            if 'multiqc_%s' % direction in cache_results['results'].keys():
                # ... and parse after cache for better debugging should HTML tags change with multiQC version changes.
                for line in cache_results['results']['multiqc_%s' % direction].split('\n'):
                    if 'id="mqc_fastqc_per_base_sequence_quality_plot' in line or \
                       'id="mqc_fastqc_per_base_sequence_quality_plot_1' in line or \
                       'id="fastqc_per_base_sequence_quality_plot' in line:
                        png_search = re.search('img src="(.*?)".*/></div>', line)
                        g = Image(url=png_search[1])
                        cache_results['results']['mean-quality-scores_%s' % direction] = g

        display(cache_results['results']['mean-quality-scores_forward'])
        if 'mean-quality-scores_reverse' in cache_results['results'].keys():
            display(cache_results['results']['mean-quality-scores_reverse'])
        return cache_results

    return _executor('qc',
                     {'dir_fastqs': os.path.abspath(dir_fastqs),
                      'pattern_fwdfiles': pattern_fwdfiles,
                      'r1r2_replace': r1r2_replace,
                      'no_rev_seqs': no_rev_seqs
                      },
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache,
                     environment=environment,
                     ppn=ppn,
                     pmem=pmem,
                     array=(len(files_fwd) + len(files_rev)),
                     **executor_args)


def trimprimers(dir_fastqs:str,
       primerseq_fwd:str,
       primerseq_rev:str,
       dir_target:str=None,
       pattern_fwdfiles:str="*_R1_*.fastq.gz",
       r1r2_replace:(str, str)=("_R1_", "_R2_"),
       no_rev_seqs:bool=False,
       ppn:int=1, pmem:str='4GB', environment:str=settings.SPIKE_ENV,
       **executor_args):
    """Trimms all fastq files in a directory.

    Parameters
    ----------
    dir_fastqs : str
        Filepath to directory which contains one or more Sequencing fastq files.
    primerseq_fwd : str
        Forward primer nucleotide sequence.
    primerseq_rev : str
        Reverse primer nucleotide sequence.
    dir_target : str
        Filepath of target directory into which trimmed files shall be moved.
        Default: None, i.e. remove files with temporary directory.
    pattern_fwdfiles : str
        Unix pattern identify fastq files.
    r1r2_replace : (str, str)
        Source and Target infix to instruct how matching reverse fastQ files
        can be identified from forward fastQ files.
    no_rev_seqs : Boolean
        If True, no R2 (=reverse) reads are expected, i.e. processed
    """
    KNOWNPRIMER = {
        'GTGCCAGCMGCCGCGGTAA': {
            'gene': '16s',
            'region': 'V4',
            'orientation': 'fwd',
            'position': '515f',
            'reference': 'Caporaso et al.',
            'doi': '10.1073/pnas.1000080107'},
        'GGACTACHVGGGTWTCTAAT': {
            'gene': '16s',
            'region': 'V4',
            'orientation': 'rev',
            'position': '806r',
            'reference': 'Caporaso et al.',
            'doi': '10.1073/pnas.1000080107'},

        'GTGYCAGCMGCCGCGGTAA': {
            'gene': '16s',
            'region': 'V4',
            'orientation': 'rev',
            'position': '806r',
            'reference': 'Parada et al.',
            'doi': '10.1111/1462-2920.13023'},
        'GGACTACNVGGGTWTCTAAT': {
            'gene': '16s',
            'region': 'V4',
            'orientation': 'rev',
            'position': '806r',
            'reference': 'Apprill et al.',
            'doi': '10.3354/ame01753'},

        'CCTACGGGNGGCWGCAG': {
            'gene': '16s',
            'region': 'V34',
            'orientation': 'fwd',
            'position': '341f',
            'reference': 'Klindworth et al.',
            'doi': '10.1093/nar/gks808'},
        'GACTACHVGGGTATCTAATCC': {
            'gene': '16s',
            'region': 'V34',
            'orientation': 'rev',
            'position': '785r',
            'reference': 'Klindworth et al.',
            'doi': '10.1093/nar/gks808'},

        'GTGYCAGCMGCCGCGGTAA': {
            'gene': '16s',
            'region': 'V45',
            'orientation': 'fwd',
            'position': '515f',
            'reference': 'Parada et al. 2016',
            'doi': '10.1111/1462-2920.13023'},
        'CCGYCAATTYMTTTRAGTTT': {
            'gene': '16s',
            'region': 'V45',
            'orientation': 'rev',
            'position': '926r',
            'reference': 'Parada et al. 2016',
            'doi': '10.1111/1462-2920.13023'},

        "CCTAYGGGDBGCWGCAG": {
            'gene': '16s',
            'region': 'V34',
            'orientation': 'fwd',
            'position': '341f',
            'reference': 'Quick-16S Plus NGS Library Prep Kit (V3-V4, UDI)',
            'doi': 'https://www.zymoresearch.de/products/quick-16s-plus-ngs-library-prep-kit-v3-v4-udi'},
        "GACTACNVGGGTMTCTAATCC": {
            'gene': '16s',
            'region': 'V34',
            'orientation': 'rev',
            'position': '806r',
            'reference': 'Quick-16S Plus NGS Library Prep Kit (V3-V4, UDI)',
            'doi': 'https://www.zymoresearch.de/products/quick-16s-plus-ngs-library-prep-kit-v3-v4-udi'},
    }
    if primerseq_fwd.upper() not in KNOWNPRIMER.keys():
        print("Forward primer sequence unknown.")
    if primerseq_rev.upper() not in KNOWNPRIMER.keys():
        print("Reverse primer sequence unknown.")

    files_fwd = []
    files_rev = []
    for fp_fastq in sorted(glob(os.path.join(dir_fastqs, "**", pattern_fwdfiles), recursive=True)):
        if os.path.basename(fp_fastq).startswith('Undetermined_S0_'):
            continue

        files_fwd.append(os.path.abspath(fp_fastq))
        if not no_rev_seqs:
            fp_rev = os.path.join(os.path.dirname(fp_fastq), os.path.basename(fp_fastq).replace(r1r2_replace[0], r1r2_replace[1]))
            if not os.path.exists(fp_rev):
                raise ValueError('Cannot find reverse file for forward file:\n  fwd: %s\n   rev: %s\nCheck r1r2_replace pattern!' % (fp_fastq, fp_rev))
            files_rev.append(os.path.abspath(fp_rev))
    if (len(files_fwd) + len(files_rev)) <= 0:
        raise ValueError("No fwd fastQ files found in %s. Check dir_fastqs and pattern_fwdfiles!" % dir_fastqs)
    if (not no_rev_seqs) & (len(files_fwd) != len(files_rev)):
        raise ValueError("Number of fwd (%i) and rev (%i) files no not match!" % (len(files_fwd), len(files_rev)))

    def pre_execute(workdir, args):
        with open("%s/commands.txt" % workdir, "w") as f:
            if not args['no_rev_seqs']:
                for fp_fwd, fp_rev in zip(files_fwd, files_rev):
                    f.write('cutadapt --json %s/%s.report.json -g %s -G %s -n 2 -o %s/%s -p %s/%s -m 1 %s %s\n' % (
                        workdir, os.path.basename(fp_fwd),
                        primerseq_fwd, primerseq_rev,
                        workdir, os.path.basename(fp_fwd), workdir, os.path.basename(fp_rev),
                        fp_fwd, fp_rev))
            else:
                for fp_fwd in files_fwd:
                    f.write('cutadapt --json %s/%s.report.json -g %s -n 1 -o %s/%s -m 1 %s\n' % (
                        workdir, os.path.basename(fp_fwd),
                        primerseq_fwd,
                        workdir, os.path.basename(fp_fwd),
                        fp_fwd))

    def commands(workdir, ppn, args):
        commands = [
            'var_cutadaptCMD=`head -n ${%s} %s/commands.txt | tail -n 1 | cut -f 1`' % (
                settings.VARNAME_PBSARRAY, workdir),
            '$var_cutadaptCMD'
             ]
        return commands

    def post_execute(workdir, args):
        results = dict()

        res = []
        for fp_fwd in tqdm(files_fwd, 'collect trimming stats'):
            fp_report = os.path.join(workdir, os.path.basename(fp_fwd) + '.report.json')
            report = json.load(open(fp_report, 'r'))
            stats = report['basepair_counts']
            stats['sample_name'] = os.path.basename(report['input']['path1'])
            stats['is_paired'] = report['input']['paired']
            res.append(stats)
        results['report'] = pd.DataFrame(res)
        results['report'].columns = list(map(lambda x: x.replace('read1', 'forward').replace('read2', 'reverse'), results['report'].columns))

        if dir_target is not None:
            os.makedirs(dir_target, exist_ok=True)
            for fp_orig in files_fwd + files_rev:
                fp_src = os.path.join(workdir, os.path.basename(fp_orig))
                fp_target = os.path.join(dir_target, os.path.basename(fp_orig))
                if executor_args.get('verbose', sys.stderr) is not None:
                    print("moving trimmed file '%s' into '%s'" % (os.path.basename(fp_orig), dir_target), file=executor_args.get('verbose', sys.stderr))
                shutil.move(fp_src, fp_target)
        return results

    def post_cache(cache_results):
        summary = pd.DataFrame(index=['forward', 'reverse'], data=np.nan, columns=['read', 'written', 'ratio'])
        for direction in ['forward', 'reverse']:
            summary.loc[direction, 'read'] = cache_results['results']['report']['input_%s' % direction].mean()
            summary.loc[direction, 'written'] = cache_results['results']['report']['output_%s' % direction].mean()
            summary.loc[direction, 'ratio'] = (cache_results['results']['report']['output_%s' % direction] / cache_results['results']['report']['input_%s' % direction]).mean()
        summary = summary.dropna(axis=0, how='all')
        cache_results['results']['summary'] = summary

        if executor_args.get('verbose', sys.stderr) is not None:
            for l in cache_results['conda_list']:
                if l.startswith('cutadapt'):
                    print(l.strip())

        display(cache_results['results']['summary'])
        return cache_results

    return _executor('trim',
                     {'primerseq_fwd': primerseq_fwd,
                      'primerseq_rev': primerseq_rev,
                      'dir_fastqs': os.path.abspath(dir_fastqs),
                      'pattern_fwdfiles': pattern_fwdfiles,
                      'r1r2_replace': r1r2_replace,
                      'no_rev_seqs': no_rev_seqs
                      },
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache,
                     environment=environment,
                     ppn=ppn,
                     pmem=pmem,
                     array=len(files_fwd),
                     **executor_args)


def blast_isolates(fp_fasta_isolates:str,
       counts:pd.DataFrame, fp_taxonomy_trained_classifier_isolates:str=None,
       taxonomy_asvs:pd.Series=None,
       ppn:int=2, pmem:str='4GB', environment:str=settings.ISOLATEASVS_ENV,
       **executor_args):
    """Takes one multiple fastA file and searchs ASVs against these isolate sequences.

    Parameters
    ----------
    fp_taxonomy_trained_classifier_isolates : str
        Optional, but highly recommended! File path to sklearn pre-trained taxonomy classifier
        to assign lineages to isolate sequences. This might help identify isolates that are
        reverse complements!
    """
    INFIX_REVCOM='reverse_complement__'
    min_isolate_length = max(150, len(counts.index[0]) * 2)

    def __external_select_isolate_sequences(workdir):
        import pandas as pd
        with open('%s/isolate_clusters.clstr' % workdir, 'r') as f:
            sequse = []
            # collect information about sequence use in clusters
            cluster = None
            cluster_sizes = dict()
            current_cluster_size = 0
            # get cluster size in terms of number of sequences
            for line in f.readlines():
                if line.strip() == "":
                    continue
                elif line.startswith('>Cluster '):
                    if cluster is not None:
                        cluster_sizes[cluster] = current_cluster_size
                        current_cluster_size = 0
                    cluster = line[1:].strip()
                else:
                    seqname = line.split('>')[1].split('...')[0]
                    is_reverse_complement = seqname.startswith('reverse_complement__')
                    seqname = seqname.replace('reverse_complement__', '')
                    is_representative = line.strip().endswith('... *')
                    sequse.append({'cluster': cluster, 'escaped_sequence_id': seqname, 'is_cluster_representative': is_representative, 'is_reverse_complement': is_reverse_complement})
                    current_cluster_size += 1
            sequse = pd.DataFrame(sequse).merge(pd.Series(cluster_sizes, name='cluster_size'), left_on='cluster', right_index=True)

            # sort sequences by 1) cluster size 2) being representative and 3) preferring original direction
            # group by cluster, start with largest, pick representative sequence as first member of positive set; all other cluster member as negative set
            # iterate clusters but only add new positive sequences, if not already used by other clusters, i.e. being member in the negative set
            seqs_positive = set()
            seqs_negative = set()
            for cluster, g in sequse.sort_values(by=['cluster_size', 'is_cluster_representative', 'is_reverse_complement'], ascending=[False, False, True]).groupby('cluster', sort=False):
                g_filtered = g[~g['escaped_sequence_id'].isin(seqs_negative)]
                seqs_positive |= set(g_filtered['escaped_sequence_id'][:1])
                seqs_negative |= set(g_filtered['escaped_sequence_id'][1:])

            # once we know the positive isolate sequence, we go back to all isolate sequence use (original and rev comp are in these list)
            # keep the above sorting and pick the "better" version (forward or reverse complement) per isolate sequence
            sequse = sequse[sequse['escaped_sequence_id'].isin(seqs_positive)].groupby('escaped_sequence_id').head(1)

        with open('%s/selected_isolate_sequences.fasta' % workdir, 'w') as w:
            with open('%s/intermediate_isolates.fasta' % workdir, 'r') as f:
                name, seq = None, None
                for line in f.readlines():
                    if line.startswith('>'):
                        name = line.strip()[1:]
                    else:
                        seq = line.strip()
                        if name in sequse['escaped_sequence_id'].values:
                            w.write('>%s\n%s\n' % (name, seq))

    def _create_length_warning(too_short, fp_fasta_isolates, num_ok_seqs):
        return ("WARNING: Your isolate sequence collection (%s) contains %i of %i "
                "sequences that are too short (<%i bases)! They won't be used for analysis. Please consider pruning your "
                "collection!\n\n%s") % (
                    os.path.basename(fp_fasta_isolates),
                    len(too_short),
                    len(too_short) + num_ok_seqs,
                    min_isolate_length,
                    '\n'.join(map(lambda seq: '>%s%s\n%s\n' % (seq.metadata['id'], seq.metadata['description'], str(seq)), too_short)))

    def pre_execute(workdir, args):
        # double check that fasta names are unique in library
        map_isolate_names = []
        too_short = []
        for i, seq in enumerate(read(args['fp_fasta_isolates'], format='fasta')):
            if len(str(seq)) < min_isolate_length:
                too_short.append(seq)
                continue
            map_isolate_names.append(' '.join([seq.metadata['id'], seq.metadata['description']]))
        id_occurrences = pd.Series(map_isolate_names).value_counts()
        if id_occurrences[id_occurrences > 1].shape[0] > 0:
            raise ValueError("Your isolate collection contains the following %i duplicate names:\n%s" % (
                id_occurrences[id_occurrences > 1].shape[0],
                '  \n'.join(sorted(id_occurrences[id_occurrences > 1].index))))

        # write new fasta files with escaped sequence names
        info_isolates = pd.DataFrame(index=map_isolate_names, data=None)
        info_isolates.index.name = 'original_name'
        with open('%s/intermediate_isolates.fasta' % workdir, 'w') as f:
            for i, seq in enumerate(read(args['fp_fasta_isolates'], format='fasta')):
                if len(str(seq)) < min_isolate_length:
                    continue
                orig_seq_id = ' '.join([seq.metadata['id'], seq.metadata['description']])
                info_isolates.loc[orig_seq_id, 'escaped_id'] = 'isolate_seq_%i' % (i+1)
                info_isolates.loc[orig_seq_id, 'escaped_id_reverse_complement'] = INFIX_REVCOM + info_isolates.loc[orig_seq_id, 'escaped_id']
                info_isolates.loc[orig_seq_id, 'sequence'] = str(seq)
                info_isolates.loc[orig_seq_id, 'sequence_reverse_complement'] = str(DNA(seq).reverse_complement())
                f.write('>%s\n%s\n' % (info_isolates.loc[orig_seq_id, 'escaped_id'], str(seq)))
        if len(too_short) > 0:
            msg = _create_length_warning(too_short, args['fp_fasta_isolates'], len(map_isolate_names))
            print(msg)
            with open('%s/length_warning.txt' % workdir, 'w') as f:
                f.write(msg)

        if fp_taxonomy_trained_classifier_isolates is not None:
            seqs = []
            for seq in read(args['fp_fasta_isolates'], format='fasta'):
                if len(str(seq)) < min_isolate_length:
                    continue
                seqs.append(str(seq))
                seqs.append(str(DNA(str(seq).upper()).reverse_complement()))

            res_tax_isolates = taxonomy_RDP(pd.Series(seqs, index=seqs), args['fp_taxonomy_trained_classifier_isolates'], dry=executor_args.get('dry', True), wait=True, use_grid=executor_args.get('use_grid', True), dirty=executor_args.get('dirty', False))
            for seq, lineage in res_tax_isolates['results']['Taxon'].items():
                taxa_idx = info_isolates[info_isolates['sequence'] == seq].index
                if len(taxa_idx) > 0:
                    info_isolates.loc[taxa_idx, 'lineage'] = lineage
                taxa_idx = info_isolates[info_isolates['sequence_reverse_complement'] == seq].index
                if len(taxa_idx) > 0:
                    info_isolates.loc[taxa_idx, 'lineage_reverse_complement'] = lineage
            for idx, row in info_isolates.iterrows():
                info_isolates.loc[idx, 'probably_reverse_complement'] = row['lineage'].split(';') < row['lineage_reverse_complement'].split(';')
        else:
            print("You might want to provide a pre-trained sklearn taxonomy classifier file path via 'fp_taxonomy_trained_classifier_isolates', such that isolate sequences can be assigned a lineage. It also helps to identify reverse complement isolate sequences!", file=sys.stderr)
        # dump all information about isolate sequences in one file
        info_isolates.to_csv('%s/info_isolates.csv' % workdir, sep="\t")

        map_asv_names = dict()
        with open('%s/asvs.fasta' % workdir, "w") as f:
            for i, asv in enumerate(sorted(args['asvs'].values)):
                f.write('>asv_%i\n%s\n' % (i+1, asv))
                map_asv_names['asv_%i' % (i+1)] = asv
        map_asv_names = pd.Series(map_asv_names)
        map_asv_names.to_frame().rename(columns={0: 'asv_sequences'}).to_csv('%s/map_asv_names.csv' % workdir, index_label='escaped_asv', sep="\t")

        with open('%s/awk_prog.txt' % workdir, 'w') as f:
            f.write('BEGIN{RS=">Cluster ";FS="\\n"}NR>1{if (($0!~/reverse_complement.+\*/) && (NF > 3))print ">Cluster "$0}\n')

        with open('%s/script_align.sh' % workdir, 'w') as f:
            f.write(
"""
#!/bin/bash

IFS=$'\\x0a'
for cluster in `cat %s/dubious_clusters.seqids`
do
	clusterID=`echo "$cluster" | cut -d " " -f 1`
	seqIDs=`echo "$cluster" | cut -d " " -f 2-`

	echo "Processing $clusterID";

	fp_seqs="%s/${clusterID}.mfa"
	echo "" > $fp_seqs
	for seqid in `echo "$seqIDs" | tr " " "\\n"`
	do
		grep "^>$seqid$" -A 1 %s/both_isolates.fasta >> $fp_seqs
	done

	fp_alignment="%s/${clusterID}.msa"
	mafft $fp_seqs > $fp_alignment
done
""" % (workdir, workdir, workdir, workdir) + "\n")

        with open('%s/select_isolate_sequences.py' % workdir, 'w') as f:
            f.write('import sys\n')
            import inspect
            f.write(inspect.getsource(__external_select_isolate_sequences).replace('\n    ', '\n').replace('    def ', 'def '))
            f.write('__external_select_isolate_sequences(sys.argv[1])\n')

    def commands(workdir, ppn, args):
        commands = []

        # create reverse complement sequences of isolates and merge with isolates
        commands.append((
            'seqtk '
            'seq '
            '-r '
            '%s/intermediate_isolates.fasta '
            '> %s/revcomp_isolates.fasta') % (workdir, workdir))

        # fix names of revcomp isolate sequences
        commands.append((
            'sed '
            '-e \"s/^>isolate_seq_/>reverse_complement__isolate_seq_/\" '
            '%s/revcomp_isolates.fasta '
            '> %s/revcomp_isolates_renamed.fasta') % (workdir, workdir))

        # merge isolates and revcomp-isolates into one file
        commands.append('cat %s/intermediate_isolates.fasta %s/revcomp_isolates_renamed.fasta > %s/both_isolates.fasta' % (workdir, workdir, workdir))

        # cluster sequences via cd-hit
        commands.append(
            ('cd-hit '
             '-i %s/both_isolates.fasta ' # input filename in fasta format, required, can be in .gz format
             '-o %s/isolate_clusters ' # output filename, required
             '-c 0.99 ' # sequence identity threshold, default 0.9 this is the default cd-hit's "global sequence identity" calculated as: number of identical amino acids or bases in alignment divided by the full length of the shorter sequence
             '-T %i ' # number of threads
             '-d 999 ' # length of description in .clstr file, default 20
             '-p 1 ' # print alignment overlap in .clstr file
             '-g 1 ' # by cd-hit's default algorithm, a sequence is clustered to the first cluster that meet the threshold (fast cluster). If set to 1, the program will cluster it into the most similar cluster that meet the threshold
             '-aS 0.95 ' # alignment coverage for the shorter sequence, default 0.0 if set to 0.9, the alignment must covers 90% of the sequence
             '-A 50 ' # minimal alignment coverage control for the both sequences
             '-uL 0.05 ' # maximum unmatched percentage for the longer sequence, default 1.0 if set to 0.1, the unmatched region (excluding leading and tailing gaps) must not be more than 10% of the sequence
             '-uS 0.05 ' # maximum unmatched percentage for the shorter sequence, default 1.0 if set to 0.1, the unmatched region (excluding leading and tailing gaps) must not be more than 10% of the sequence
             '-sc 1 ' # sort clusters by size (number of sequences), default 0, output clusters by decreasing length if set to 1, output clusters by decreasing size
             ) % (workdir, workdir, ppn))

        # filter cd-hit clusters such that they are not represented by reverse complement sequences AND have more than one member
        commands.append(
            ('awk -f %s/awk_prog.txt '
             '%s/isolate_clusters.clstr '
             '> %s/dubious_clusters.clstr') % (workdir, workdir, workdir))

        # collect cluster member names
        commands.append('sed -e "s#Cluster \\(.*\\)#Cluster\\1#" %s/dubious_clusters.clstr | sed -e "s#.*>\\(.*\\)\\.\\.\\..*#\\1#g" | tr "\\n" " " | tr ">" "\\n" > %s/dubious_clusters.seqids' % (workdir, workdir))

        # use grep to extract all sequences for each cluster and mafft to create multiple sequence alignments
        commands.append('bash %s/script_align.sh' % workdir)

        ## extract only forward cluster representatives
        #commands.append('grep "^>isolate_seq_" -A 1 %s/both_isolates.fasta > %s/fwd_cluster_repseq.fasta' % (workdir, workdir))
        commands.append('python3 %s/select_isolate_sequences.py %s' % (workdir, workdir))

        # construct blast database
        commands.append(
            ('makeblastdb '
             '-in %s/selected_isolate_sequences.fasta '
             '-dbtype nucl '
             '-input_type fasta '
             '-title isolates '
             '-out %s/selected_isolate_sequences_blastDB') % (workdir, workdir))

        # the actual isolate asv search
        commands.append(
            ('blastn '
             '-query %s/asvs.fasta '
             '-task blastn '
             '-db %s/selected_isolate_sequences_blastDB '
             '-out %s/asv_search.tsv '
             '-outfmt \"6 std qlen slen nident qcovs\" '
             '-max_target_seqs 5 '
             '-num_threads %i') % (workdir, workdir, workdir, ppn))

        # # the actual isolate asv search
        # commands.append(
        #     ('blastn '
        #      '-query %s/asvs.fasta '
        #      '-task blastn -db %s/isolates_blastDB -out %s/asv_search.tsv -outfmt 6 -max_target_seqs 5 -num_threads %i') % (workdir, workdir, workdir, ppn))

        return commands

    def post_execute(workdir, args):
        results = dict()

        def _get_dubious_cluster(workdir, info_isolates):
            def __get_original_seq_name(escaped_id, info_isolates):
                seqlabel = ''.join(map(str.strip, info_isolates[info_isolates['escaped_id'] == escaped_id.replace(INFIX_REVCOM, '')].index))
                if INFIX_REVCOM in escaped_id:
                    return 'reverse complement of "%s"' % seqlabel
                else:
                    return seqlabel

            max_label_len = max(map(len, info_isolates.index)) + len('reverse complement of ')

            res = ""
            dubious_cluster_files = sorted(glob('%s/Cluster*.msa' % workdir), key=lambda x: int(os.path.basename(x).replace('Cluster', '').replace('.msa', '')))
            num_members = 0
            for fp_cluster_msa in dubious_cluster_files:
                newmsa = "# ==== " + os.path.basename(fp_cluster_msa) + " "
                newmsa += '=' * (max_label_len - len(newmsa) + 3) + "\n"

                if 'lineage' in info_isolates.columns:
                    seqlabel_lineages = []
                    for seq in read(fp_cluster_msa, format='fasta'):
                        idx_isolate = info_isolates[info_isolates['escaped_id'] == seq.metadata['id'].replace(INFIX_REVCOM, '')].index
                        assert len(idx_isolate) == 1
                        idx_isolate = idx_isolate[0]

                        orig_name = __get_original_seq_name(seq.metadata['id'], info_isolates)
                        if info_isolates.loc[idx_isolate, 'probably_reverse_complement']:
                            seqlabel_lineages.append((orig_name, info_isolates.loc[idx_isolate, 'lineage_reverse_complement'] + '  -- lineage inferred from reverse complement!'))
                        else:
                            seqlabel_lineages.append((orig_name, info_isolates.loc[idx_isolate, 'lineage']))

                    max_len_labels = max(map(lambda x: len(x[0]), seqlabel_lineages))
                    for seqlabel, lineage in seqlabel_lineages:
                        newmsa += '## %s%s : %s\n' % (seqlabel, ' ' * (max_len_labels - len(seqlabel)), lineage)

                for seq in read(fp_cluster_msa, format='fasta'):
                    num_members += 1
                    seqlabel = __get_original_seq_name(seq.metadata['id'], info_isolates)
                    newmsa += '>%s%s%s\n' % (seqlabel, ' ' * (max_label_len-len(seqlabel)+1), str(seq))
                newmsa += '\n'

                res += newmsa

            if len(dubious_cluster_files) > 0:
                print(("WARNING: Your isolate sequence collection (%s) contains %i "
                       "sequences (we used forward and reverse complement versions) that are dubiously similar to each other. "
                       "We clustered those into %i clusters. Please carefully "
                       "inspect clustering results and consider pruning your "
                       "collection!\nE.g. print(res['results']['dubious_cluster'])") % (os.path.basename(fp_fasta_isolates), num_members, len(dubious_cluster_files)))

            return res

        results['info_isolates'] = pd.read_csv('%s/info_isolates.csv' % workdir, sep="\t", index_col=0)
        results['info_asvs'] = pd.read_csv('%s/map_asv_names.csv' % workdir, sep="\t", index_col=0)

        results['dubious_cluster'] = _get_dubious_cluster(workdir, results['info_isolates'])

        results['blast_hits'] = pd.read_csv('%s/asv_search.tsv' % workdir, sep="\t",
                            names=['qaccver', 'saccver', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend',
                                   'sstart', 'send', 'evalue', 'bitscore', 'qlen', 'slen', 'nident', 'qcovs'])

        fp_length_warning = '%s/length_warning.txt' % workdir
        if os.path.exists(fp_length_warning):
            with open(fp_length_warning, 'r') as f:
                results['too_short_isolate_sequences'] = ''.join(f.readlines())

        return results

    def post_cache(cache_results):
        if 'too_short_isolate_sequences' in cache_results['results'].keys():
            print(cache_results['results']['too_short_isolate_sequences'])

        hits = cache_results['results']['blast_hits']
        filtered_hits = hits[(hits['evalue'] < 10**-10) &           # ignore too bad hits
                             (hits['length'] > hits['qlen'] - 10) & # at most 10 nucleotids of ASV shall be missing in match
                             (hits['mismatch'] <= 5) &              # no more than 5 mismatches
                             (hits['pident'] >= 97)]                # assuming 97% sequence identity is species radius

        cols_hitquality = ['evalue', 'bitscore', 'qlen', 'pident', 'nident', 'qcovs', 'mismatch', 'gapopen', 'qaccver']
        sort_hitquality = [True, False, False, False, False, False, True, True, True]
        cache_results['results']['blast_hits_filtered'] = filtered_hits.sort_values(by=cols_hitquality, ascending=sort_hitquality).groupby('qaccver').head(1)

        def _get_taxonomy(row):
            lineage = row['lineage']
            if row['probably_reverse_complement']:
                lineage = row['lineage_reverse_complement']
            lineage += "; i__%s" % row.name.strip()
            return lineage

        if fp_taxonomy_trained_classifier_isolates is not None:
            cache_results['results']['info_isolates']['taxonomy'] = cache_results['results']['info_isolates'].apply(_get_taxonomy, axis=1).values
        else:
            cache_results['results']['info_isolates']['taxonomy'] = cache_results['results']['info_isolates'].reset_index()['original_name'].apply(lambda x: 'd__; p__; c__; o__; f__; g__; s__; i__%s' % x).values

        cache_results['results']['taxonomy'] = cache_results['results']['info_asvs'].merge(
            cache_results['results']['blast_hits_filtered'][['qaccver', 'saccver']].merge(
                cache_results['results']['info_isolates'][['escaped_id', 'taxonomy']],
                left_on='saccver',
                right_on='escaped_id',
                how='left')[['qaccver', 'taxonomy']],
            left_index=True,
            right_on='qaccver',
            how='left').fillna('').reset_index().set_index('asv_sequences')[['qaccver', 'taxonomy']].rename(columns={'taxonomy': 'Taxon_isolate'})

        if taxonomy_asvs is not None:
            cache_results['results']['taxonomy']['Taxon'] = cache_results['results']['taxonomy'].merge(taxonomy_asvs.rename('Taxon_asv'), left_index=True, right_index=True).apply(lambda row: row['Taxon_isolate'] if row['Taxon_isolate'] != "" else row['Taxon_asv'] + '; i__', axis=1)
            del cache_results['results']['taxonomy']['Taxon_isolate']
        else:
            cache_results['results']['taxonomy'].rename(columns={'Taxon_isolate': 'Taxon'})
        # else:
        #     help = cache_results['results']['info_isolates'].reset_index()
        #     help['original_name'] = help['original_name'].apply(lambda x: 'd__; p__; c__; o__; f__; g__; s__; i__%s' % x)
        #     cache_results['results']['taxonomy'] = cache_results['results']['info_asvs'].merge(
        #         cache_results['results']['blast_hits_filtered'][['qaccver', 'saccver']].merge(
        #             help[['original_name','escaped_id']],
        #             left_on='saccver',
        #             right_on='escaped_id')[['qaccver', 'original_name']],
        #         left_index=True,
        #         right_on='qaccver',
        #         how='left').fillna('').reset_index().set_index('asv_sequences').rename(columns={'original_name': 'Taxon'})
        #     del cache_results['results']['taxonomy']['index']

        return cache_results

    return _executor('blastASVs',
                     {'fp_fasta_isolates': os.path.abspath(fp_fasta_isolates),
                      'asvs': counts.index,
                      'fp_taxonomy_trained_classifier_isolates': None if fp_taxonomy_trained_classifier_isolates is None else os.path.abspath(fp_taxonomy_trained_classifier_isolates),
                      },
                     pre_execute,
                     commands,
                     post_execute,
                     post_cache,
                     environment=environment,
                     ppn=ppn,
                     pmem=pmem,
                     **executor_args)


def _parse_timing(workdir, jobname):
    """If existant, parses timing information.

    Parameters
    ----------
    workdir : str
        Path to tmp workdir of _executor containing cr_ana_<jobname>.t* file
    jobname : str
        Name of ran job.

    Parameters
    ----------
    None if file could not be found. Otherwise: [str]
    """
    files_timing = [workdir + '/' + d
                    for d in next(os.walk(workdir))[2]
                    if 'cr_ana_%s.t' % jobname in d]
    for file_timing in files_timing:
        with open(file_timing, 'r') as content_file:
            return content_file.readlines()
        # stop after reading first found file, since there should only be one
        break
    return None


def _md5(filepath):
    """Returns md5sum of file path"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _executor(jobname, cache_arguments, pre_execute, commands, post_execute,
              post_cache=None, post_cache_arguments=dict(),
              dry=True, use_grid=True, ppn=10, nocache=False,
              pmem='20GB', environment=settings.QIIME_ENV, walltime='4:00:00',
              wait=True, timing=True, verbose=sys.stderr, array=1,
              dirty=False):
    """

    Parameters
    ----------
    jobname : str
    cache_arguments : []
    pre_execute : function
    commands : [] or dict:{'pre': [], 'main': [], 'post': []}
    post_execute : function
    post_cache : function
        A function that is called, after results have been loaded from cache /
        were generated. E.g. drawing rarefaction curves.
    environment : str

    ==template arguments that should be copied to calling analysis function==
    dry : bool
        Default: True.
        If True: only prepare working directory and create necessary input
        files and print the command that would be executed in a non dry run.
        For debugging. Workdir is not deleted.
        "pre_execute" is called, but not "post_execute".
    use_grid : bool
        Default: True.
        If True, use qsub to schedule as a grid job, otherwise run locally.
    nocache : bool
        Default: False.
        Normally, successful results are cached in .anacache directory to be
        retrieved when called a second time. You can deactivate this feature
        (useful for testing) by setting "nocache" to True.
    wait : bool
        Default: True.
        Wait for results.
    walltime : str
        Default: "12:00:00".
        hh:mm:ss formated wall runtime on cluster.
    ppn : int
        Default: 10.
        Number of CPU cores to be used.
    pmem : str
        Default: '8GB'.
        Resource request for cluster jobs. Multiply by ppn!
    timing : bool
        Default: True
        Use '/usr/bin/time' to log run time of commands.
    verbose : stream
        Default: sys.stderr
        To silence this function, set verbose=None.
    array : int
        Default: 1 = deactivated.
        Only for Torque submits: make the job an array job.
        You need to take care of correct use of ${PBS_JOBID} !
    dirty : bool
        Defaul: False.
        If True, temporary working directory will not be removed.

    Returns
    -------
    """
    DIR_CACHE = '.anacache'
    FILE_STATUS = 'finished.info'
    results = {'results': None,
               'workdir': None,
               'qid': None,
               'file_cache': None,
               'cached_inputs': dict(),
               'timing': None,
               'cache_version': 20200826,
               'created_on': None,
               'conda_env': 'unknown',
               'jobname': jobname}

    # create an ID function if no post_cache function is supplied
    def _id(x):
        return x
    if post_cache is None:
        post_cache = _id

    # phase 1: compute signature for cache file
    # convert skbio.DistanceMatrix object to a sorted version of its data for
    # hashing
    cache_args_original = dict()
    for arg in cache_arguments.keys():
        if type(cache_arguments[arg]) == DistanceMatrix:
            cache_args_original[arg] = cache_arguments[arg]
            dm = cache_arguments[arg]
            cache_arguments[arg] = dm.filter(sorted(dm.ids)).data
        if (type(cache_arguments[arg]) == dict):
            if (len({type(v) for v in cache_arguments[arg].values()} ^
                    set([DistanceMatrix])) == 0):
                cache_args_original[arg] = cache_arguments[arg]
                cache_arguments[arg] = collections.OrderedDict(
                    {k: dm.filter(sorted(dm.ids)).data
                     for k, dm
                     in cache_arguments[arg].items()})
        if (type(cache_arguments[arg]) == pd.Series):
            cache_args_original[arg] = cache_arguments[arg]
            cache_arguments[arg] = cache_arguments[arg].sort_index()
        if (type(cache_arguments[arg]) == pd.DataFrame):
            cache_args_original[arg] = cache_arguments[arg]
            cache_arguments[arg] = cache_arguments[arg].loc[
                sorted(cache_arguments[arg].index),
                sorted(cache_arguments[arg].columns)]
        if isinstance(cache_arguments[arg], Table):
            cache_args_original[arg] = cache_arguments[arg]
            cache_arguments[arg] = sorted(list(cache_arguments[arg].ids('sample'))) + \
                                   sorted(list(cache_arguments[arg].ids('observation'))) + \
                                   [cache_arguments[arg].get_table_density()]
        if (type(cache_arguments[arg]) == str) and os.path.exists(cache_arguments[arg]):
            cache_args_original[arg] = cache_arguments[arg]
            if os.path.isfile(cache_arguments[arg]):
                # if argument can be used as a file path and the file actually exists...
                # ... than use the md5sum of the file instead of the filepath for cache fingerprint
                # Thus, moving the file will not affect the cache fingerprint
                cache_arguments[arg] = _md5(cache_arguments[arg])
            else:
                # assume path is directory
                cache_arguments[arg] = os.path.abspath(cache_arguments[arg])

        # for better debugging, write hash sum for each input argument in result object
        if cache_arguments[arg] is None:
            results['cached_inputs'][arg] = None
        else:
             results['cached_inputs'][arg] = hashlib.md5(str(cache_arguments[arg]).encode()).hexdigest()

    _input = collections.OrderedDict(sorted(cache_arguments.items()))
    results['file_cache'] = "%s/%s.%s" % (DIR_CACHE, hashlib.md5(
        str(_input).encode()).hexdigest(), jobname)

    # convert back cache arguments if necessary
    for arg in cache_args_original.keys():
        cache_arguments[arg] = cache_args_original[arg]

    # phase 2: if cache contains matching file, load from cache and return
    if os.path.exists(results['file_cache']) and (nocache is not True):
        if verbose:
            verbose.write("Using existing results from '%s'. \n" %
                          results['file_cache'])
        f = open(results['file_cache'], 'rb')
        results = pickle.load(f)
        f.close()
        return post_cache(results, **post_cache_arguments)

    # phase 3: search in TMP dir if non-collected results are
    # ready or are waited for
    dir_tmp = tempfile.gettempdir()
    if use_grid:
        dir_tmp = os.environ['HOME'] + '/TMP/'
        if not os.path.exists(dir_tmp):
            raise ValueError('Temporary directory "%s" does not exist. '
                             'Please create it and restart.' % dir_tmp)

    # collect all tmp workdirs that contain the right cache signature
    pot_workdirs = []
    for _dir in next(os.walk(dir_tmp))[1]:
        # a potential working directory needs to have the matching job name
        if _dir.startswith('ana_%s_' % results['jobname']):
            # for shared computers, make sure you have permission to read dir contents
            if not os.access(os.path.join(dir_tmp, _dir), os.R_OK):
                continue
            potwd = os.path.join(dir_tmp, _dir)
            # and a matching cache file signature
            if results['file_cache'].split('/')[-1] in next(os.walk(potwd))[2]:
                pot_workdirs.append(potwd)
    finished_workdirs = []
    for wd in pot_workdirs:
        all_finished = os.path.exists('%s/finished.info' % wd)
        # for i in range(array):
        #     exp_finish_suffix = ""
        #     if array > 1:
        #         exp_finish_suffix = str(int(i+1))
        #     if (array == 1):
        #         if (settings.GRIDNAME == 'JLU'):
        #             if use_grid:
        #                 exp_finish_suffix = 'undefined'
        #             else:
        #                 exp_finish_suffix = '1'
        #         else:
        #             exp_finish_suffix = '1'
        #     if not os.path.exists('%s/finished.info%s' % (wd, exp_finish_suffix)):
        #         all_finished = False
        #         break
        if all_finished:
            finished_workdirs.append(wd)
    if len(pot_workdirs) > 0 and len(finished_workdirs) <= 0:
        if verbose:
            verbose.write(
                ('Found %i temporary working directories, but non of '
                 'them have finished (missing "finished.info" file). If no job is currently running,'
                 ' you might want to delete these directories and res'
                 'tart:\n  %s\n') % (len(pot_workdirs),
                                     "\n  ".join(pot_workdirs)))
        return results
    if len(finished_workdirs) > 0:
        # arbitrarily pick first found workdir
        results['workdir'] = finished_workdirs[0]
        if verbose:
            verbose.write('found matching working dir "%s"\n' %
                          results['workdir'])
    else:
        # create a temporary working directory
        prefix = 'ana_%s_' % jobname
        results['workdir'] = tempfile.mkdtemp(prefix=prefix, dir=dir_tmp)
        if verbose:
            verbose.write("Working directory is '%s', cachefile is '%s'. " %
                          (results['workdir'], results['file_cache']))
        # leave an empty file in workdir with cache file name to later
        # parse results from tmp dir
        f = open("%s/%s" % (results['workdir'],
                            results['file_cache'].split('/')[-1]), 'w')
        f.close()

        pre_execute(results['workdir'], cache_arguments)

        lst_commands = commands(results['workdir'], ppn, cache_arguments)
        # convert to new dict structure instead of flat list
        if isinstance(lst_commands, list):
            lst_commands = {'main': lst_commands, 'pre': [], 'post': []}
        # device creation of a file _after_ execution of the job in workdir
        final_cmd = 'touch %s/%s' % (results['workdir'], FILE_STATUS)
        lst_commands['post'].append(final_cmd)

        results['qid'] = cluster_run(
            lst_commands, 'ana_%s' % jobname, results['workdir']+'mock',
            environment, ppn=ppn, wait=wait, dry=dry,
            pmem=pmem, walltime=walltime,
            file_qid=results['workdir']+'/cluster_job_id.txt',
            file_condaenvinfo=results['workdir']+'/conda_info.txt',
            timing=timing,
            file_timing=results['workdir']+('/timing${%s}.txt' % settings.VARNAME_PBSARRAY),
            array=array, use_grid=use_grid)
        if dry:
            return results
        if wait is False:
            return results

    results['results'] = post_execute(results['workdir'],
                                      cache_arguments)
    results['created_on'] = datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y-%m-%d %H:%M:%S')

    results['conda_env'] = environment
    with open(results['workdir']+'/conda_info.txt', 'r') as f:
        results['conda_list'] = f.readlines()

    results['timing'] = []
    for timingfile in next(os.walk(results['workdir']))[2]:
        if timingfile.startswith('timing'):
            with open(results['workdir']+'/'+timingfile, 'r') as content_file:
                results['timing'] += content_file.readlines()

    if results['results'] is not None:
        if not dirty:
            shutil.rmtree(results['workdir'])
            if verbose:
                verbose.write(" Was removed.\n")

    os.makedirs(os.path.dirname(results['file_cache']), exist_ok=True)
    f = open(results['file_cache'], 'wb')
    pickle.dump(results, f)
    f.close()

    return post_cache(results, **post_cache_arguments)
