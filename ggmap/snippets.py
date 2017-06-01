import pandas as pd
import biom
from biom.util import biom_open
from mpl_toolkits.basemap import Basemap
from itertools import repeat, chain
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import os
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
import sys
from itertools import combinations
from skbio.stats.distance import permanova


RANKS = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']


def biom2pandas(file_biom, withTaxonomy=False, astype=int):
    """ Converts a biom file into a Pandas.DataFrame

    Parameters
    ----------
    file_biom : str
        The path to the biom file.
    withTaxonomy : bool
        If TRUE, returns a second Pandas.DataFrame with lineage information for
        each feature, e.g. OTU or deblur-sequence. Default: FALSE
    astype : type
        datatype into each value of the biom table is casted. Default: int.
        Use e.g. float if biom table contains relative abundances instead of
        raw reads.

    Returns
    -------
    A Pandas.DataFrame holding holding numerical values from the biom file.
    If withTaxonomy is TRUE then a second Pandas.DataFrame is returned, holding
    lineage information about each feature.

    Raises
    ------
    IOError
        If file_biom cannot be read.
    ValueError
        If withTaxonomy=TRUE but biom file does not hold taxonomy information.
    """
    try:
        table = biom.load_table(file_biom)
        counts = pd.DataFrame(table.matrix_data.T.todense().astype(astype),
                              index=table.ids(axis='sample'),
                              columns=table.ids(axis='observation')).T
        if withTaxonomy:
            otu_ids = table.ids(axis='observation')
            if table.metadata(otu_ids[0], axis='observation') is not None:
                if 'taxonomy' in table.metadata(otu_ids[0],
                                                axis='observation'):
                    mapping = {i: table.metadata(id=i, axis='observation')
                               ['taxonomy']
                               for i in otu_ids}
                    taxonomy = pd.DataFrame(mapping,
                                            index=map(str.lower, RANKS)).T
                    return counts, taxonomy
            raise ValueError('No taxonomy information found in biom file.')
        else:
            return counts
    except IOError:
        raise IOError('Cannot read file "%s"' % file_biom)


def pandas2biom(file_biom, table):
    """ Writes a Pandas.DataFrame into a biom file.

    Parameters
    ----------
    file_biom: str
        The filename of the BIOM file to be created.
    table: a Pandas.DataFrame
        The table that should be written as BIOM.

    Raises
    ------
    IOError
        If file_biom cannot be written.

    TODO
    ----
        1) also store taxonomy information
    """
    try:
        bt = biom.Table(table.values,
                        observation_ids=table.index,
                        sample_ids=table.columns)
        with biom_open(file_biom, 'w') as f:
            bt.to_hdf5(f, "example")
    except IOError:
        raise IOError('Cannot write to file "%s"' % file_biom)


def parse_splitlibrarieslog(filename):
    """ Parse the log of a QIIME split_libraries_xxx.py run.

    Especially deal with multiple input files, i.e. several sections in log.

    Parameters
    ----------
    filename : str
        The filename of the log to parse.

    Returns
    -------
    A Pandas.DataFrame containing two column with 'counts' and sample name for
    each sample in the log file.
    (We might see duplicate sample names from multiple input files, thus we
     cannot make the sample name the index.)

    Raises
    ------
    IOError
        If filename cannot be read.
    """
    try:
        counts = []
        f = open(filename, 'r')
        endOfFile = False
        while not endOfFile:
            # find begin of count table
            while True:
                line = f.readline()
                if 'Median sequence length:' in line:
                    break
            # collect counts
            while True:
                line = f.readline()
                if line == '\n':
                    break
                samplename, count = line.split()
                counts.append({'sample': samplename, 'counts': count})
            # check if file contains more blocks
            while True:
                line = f.readline()
                if 'Input file paths' in line:
                    break
                if line == '':
                    endOfFile = True
                    break
        return pd.DataFrame(sorted(counts,
                                   key=lambda x: int(x['counts']),
                                   reverse=True), dtype=int)
    except IOError:
        raise IOError('Cannot read file "%s"' % filename)


def drawMap(points, basemap=None):
    """ Plots coordinates of metadata to a worldmap.

    Parameters
    ----------
    points : a special data structure
    basemap : if not the whole earth, pass the rectangle to be plotted

    Returns
    -------
    Nothing
    """
    map = basemap
    if basemap is None:
        map = Basemap(projection='robin', lon_0=180, resolution='c')
    # Fill the globe with a blue color
    map.drawmapboundary(fill_color='lightblue', color='white')
    # Fill the continents with the land color
    map.fillcontinents(color='lightgreen', lake_color='lightblue', zorder=1)
    map.drawcoastlines(color='gray', zorder=1)

    for z, set_of_points in enumerate(points):
        coords = set_of_points['coords'][['latitude', 'longitude']].dropna()
        x, y = map(coords.longitude.values, coords.latitude.values)
        size = 50
        if 'size' in set_of_points:
            size = set_of_points['size']
        alpha = 0.5
        if 'alpha' in set_of_points:
            alpha = set_of_points['alpha']
        map.scatter(x, y, marker='o', color=set_of_points['color'], s=size,
                    zorder=2+z, alpha=alpha)


def _repMiddleValues(values):
    """ Takes a list of values and repeats each value once.

    Example: [1,2,3] -> [1,1,2,2,3,3]

    Parameters
    ----------
    values : [a]

    Returns
    -------
    [a] where each element has been duplicated.
    """

    return list(chain.from_iterable(repeat(v, 2) for v in values))


def _shiftLeft(values):
    """ All list elements are shifted left. Leftmost element is lost, new right
        elements is last element +1.

    Parameters
    ----------
    values : [int]

    Returns
    -------
    [int]: A list where first element is lost, elements shift one position to
    the left and last element is last input element +1.
    """
    return values[1:]+[values[-1]+1]


def plotTaxonomy(file_otutable,
                 metadata,
                 group_l0=None,
                 group_l1=None,
                 group_l2=None,
                 rank='Phylum',
                 file_taxonomy=('/projects/emp/03-otus/reference/'
                                '97_otu_taxonomy.txt'),
                 verbose=True,
                 reorder_samples=False,
                 print_sample_labels=False,
                 minreadnr=50,
                 plottaxa=None,
                 fct_aggregate=None):
    """
    Parameters
    ----------
    file_otutable : file
        Path to a biom OTU table
    metadata : pandas.DataFrame
        metadata
    file_taxonomy : file
        Path to a GreenGenes taxonomy file.
    reorder_samples : Bool
        True = sort samples in each group according to abundance of most
        abundant taxon
    print_sample_labels : Bool
        True = print sample names on x-axis. Use only for small numbers of
        samples!
    minreadnr : int
        min number of reads a taxon need to have to be plotted
    plottaxa : [str]
        Only plot abundances for taxa IDs provided. If None, all taxa are
        plotted. Default: None
    rank : str
        Set taxonomic level to collapse abundances. Use 'raw' to de-activate
        collapsing.
    fct_aggregate : function
        A numpy function to aggregate over several samples.
    """

    NAME_LOW_ABUNDANCE = 'low abundance'

    # Parameter checks: check that grouping fields are in metadata table
    for i, field in enumerate([group_l0, group_l1, group_l2]):
        if field is not None:
            if field not in metadata.columns:
                raise ValueError(('Column "%s" for grouping level %i is not '
                                  'in metadata table!') % (field, i))

    # check that rank is a valid taxonomic rank
    if rank not in RANKS + ['raw']:
        raise ValueError('"%s" is not a valid taxonomic rank. Choose from %s' %
                         (rank, ", ".join(RANKS)))

    # check that taxonomy file exists
    if not os.path.exists(file_taxonomy) and rank != 'raw':
        raise IOError('Taxonomy file not found!')

    # check that biom table can be read
    if not os.path.exists(file_otutable):
        raise IOError('OTU table file not found')
    rawcounts = biom2pandas(file_otutable)

    # restrict to those samples for which we have metadata AND counts
    meta = metadata.loc[[idx
                         for idx in metadata.index
                         if idx in rawcounts.columns], :]
    counts = rawcounts.loc[:, meta.index]
    if verbose:
        print('%i samples left with metadata and counts.' % meta.shape[0])

    # assign taxonomy and collapse at given rank
    if rank != 'raw':
        lineages = pd.read_csv(file_taxonomy, sep="\t", header=None,
                               names=['otuID', 'taxonomy'],
                               usecols=[0, 1])  # only parse two first columns
        lineages['otuID'] = lineages['otuID'].astype(str)
        lineages.set_index('otuID', inplace=True)
        # add taxonomic lineage information to the counts as column "taxonomy"
        rank_counts = pd.merge(counts, lineages, how='left', left_index=True,
                               right_index=True)

        # split lineage string into individual taxa names on ';' and remove
        # surrounding whitespaces. If rank does not exist return r+'__' instead
        def _splitranks(x, rank):
            try:
                return [t.strip() for t in x.split(";")][RANKS.index(rank)]
            except IndexError:
                return RANKS[RANKS.index(rank)].lower()[0] + "__"
        # add columns for each tax rank, such that we can groupby later on
        rank_counts[rank] = rank_counts['taxonomy'].apply(lambda x:
                                                          _splitranks(x, rank))
        # sum counts according to the selected rank
        rank_counts = rank_counts.reset_index().groupby(rank).sum()
        # get rid of the old index, i.e. OTU ids, since we have grouped by some
        # rank
        if verbose:
            print('%i taxa left after collapsing to %s.' %
                  (rank_counts.shape[0], rank))
    else:
        rank_counts = counts

    lowAbundandTaxa = rank_counts[(rank_counts.sum(axis=1) < minreadnr)].index
    highAbundantTaxa = rank_counts[(rank_counts.sum(axis=1) >=
                                    minreadnr)].index

    # normalize to 1 in each sample
    rank_counts /= rank_counts.sum(axis=0)

    # filter low abundant taxa
    if len(lowAbundandTaxa) > 0:
        lowReadTaxa = rank_counts.loc[lowAbundandTaxa, :].sum(axis=0)
        lowReadTaxa.name = NAME_LOW_ABUNDANCE
        rank_counts = rank_counts.loc[highAbundantTaxa, :]
        rank_counts = rank_counts.append(lowReadTaxa)
        if verbose:
            print('%i taxa left after filtering low abundant.' %
                  (rank_counts.shape[0]-1))

    # restrict to those taxa that are asked for in plottaxa
    if plottaxa is not None:
        rank_counts = rank_counts.loc[plottaxa, :]
        if verbose:
            print('%i taxa left after restricting to provided list.' %
                  (rank_counts.shape[0]))

    # all for plotting
    # sort taxa according to sum of abundance
    taxaidx = list(rank_counts.mean(axis=1).sort_values(ascending=False).index)
    if len(lowAbundandTaxa) > 0:
        taxaidx = [taxon
                   for taxon in taxaidx
                   if taxon != NAME_LOW_ABUNDANCE] + [NAME_LOW_ABUNDANCE]
    rank_counts = rank_counts.loc[taxaidx, :]

    # aggregate over samples
    if fct_aggregate is not None:
        levels = [f for f in [group_l2, group_l1, group_l0] if f is not None]
        if len(levels) < 1:
            raise ValueError("Cannot aggregate samples, "
                             "if no grouping is given!")
        # return rank_counts, meta, levels, None
        grs = dict()
        newmeta = dict()
        for n, g in meta.groupby(list(reversed(levels))):
            for sampleid in g.index:
                if isinstance(n, tuple):
                    grs[sampleid] = "###".join(list(map(str, n)))
                else:
                    grs[sampleid] = str(n)
            if isinstance(n, tuple):
                x = dict(zip(reversed(levels), n))
            else:
                x = {levels[0]: n}
            x['num'] = g.shape[0]
            if isinstance(n, tuple):
                newmeta["###".join(list(map(str, n)))] = x
            else:
                newmeta[str(n)] = x
        rank_counts = rank_counts.T.groupby(by=grs).agg(fct_aggregate).T
        meta = pd.DataFrame(newmeta).T
        group_l0, group_l1, group_l2 = None, group_l0, group_l1

    # prepare abundances for plot
    vals = rank_counts.cumsum()

    # collect information about how to plot data
    graphinfo = pd.DataFrame(data=None, index=vals.columns)
    if group_l0 is None:
        meta['help_plottaxonomy_level_0'] = 'all'
        grps0 = meta.groupby('help_plottaxonomy_level_0')
    else:
        grps0 = meta.groupby(group_l0)
    for i0, (n0, g0) in enumerate(grps0):
        graphinfo.loc[g0.index, 'group_l0'] = n0

        grps1 = [('all', g0)]
        if group_l1 is not None:
            grps1 = g0.groupby(group_l1)
        offset = 0
        for i1, (n1, g1) in enumerate(grps1):
            sample_idxs = vals.iloc[0, :].loc[g1.index]
            if reorder_samples:
                sample_idxs = sample_idxs.sort_values(ascending=False)
            sample_idxs = sample_idxs.index
            if group_l2 is not None:
                help_sample_idxs = []
                for n2, g2 in g0.loc[g1.index, :].groupby(group_l2):
                    reorderd = [idx for idx in sample_idxs if idx in g2.index]
                    help_sample_idxs.extend(reorderd)
                    graphinfo.loc[reorderd, 'group_l2'] = n2
                sample_idxs = help_sample_idxs
            graphinfo.loc[sample_idxs, 'group_l1'] = n1
            graphinfo.loc[sample_idxs, 'xpos'] = range(offset,
                                                       offset+len(sample_idxs))
            offset += len(sample_idxs)
            if i1 < len(grps1):
                offset += max(1, int(g0.shape[0]*0.05))

    # define colors for taxons
    availColors = sns.color_palette('Paired', 100)
    colors = {NAME_LOW_ABUNDANCE: 'white'}
    for i in range(0, vals.shape[0]):
        taxon = vals.index[i]
        if taxon not in colors:
            colors[taxon] = availColors[len(colors) % len(availColors)]

    # plot the actual thing
    fig, axarr = plt.subplots(len(grps0), 1)
    for ypos, (n0, g0) in enumerate(graphinfo.groupby('group_l0')):
        if group_l0 is None:
            ax = axarr
        else:
            ax = axarr[ypos]
        for i in range(0, vals.shape[0]):
            taxon = vals.index[i]
            y_prev = None
            for j, (name, g1_idx) in enumerate(graphinfo.loc[g0.index, :]
                                               .groupby('group_l1')):
                if i == 0:
                    y_prev = [0] * g1_idx.shape[0]
                else:
                    y_prev = vals.loc[:, g1_idx.sort_values(by='xpos').index]\
                        .iloc[i-1, :]
                y_curr = vals.loc[:, g1_idx.sort_values(by='xpos').index]\
                    .iloc[i, :]
                xpos = g1_idx.sort_values(by='xpos')['xpos']

                ax.fill_between(_shiftLeft(_repMiddleValues(xpos)),
                                _repMiddleValues(y_prev),
                                _repMiddleValues(y_curr),
                                color=colors[taxon])

        # decorate graph with axes labels ...
        if print_sample_labels:
            ax.set_xticks(graphinfo.loc[g0.index, :]
                          .sort_values(by='xpos')['xpos']+.5)
            # determine sample lables, which might be aggregated
            data = graphinfo[['xpos']]
            if fct_aggregate is not None:
                data = graphinfo[['xpos']].merge(meta[['num']],
                                                 left_index=True,
                                                 right_index=True)
            labels = []
            for idx, row in data.sort_values(by='xpos').iterrows():
                if '###' in idx:
                    label = "%s" % idx.split('###')[-1]
                else:
                    label = idx
                if 'num' in row.index:
                    label += " (n=%i)" % row['num']
                labels.append(label)
            ax.set_xticklabels(labels, rotation='vertical')
            ax.xaxis.set_ticks_position("bottom")
        else:
            ax.set_xticks([])

        # crop graph to actually plotted bars
        ax.set_xlim(0, graphinfo.loc[g0.index, 'xpos'].max()+1)
        ax.set_ylim(0, rank_counts.sum().max())
        ax.set_axis_bgcolor('white')

        if group_l0 is None:
            ax.set_ylabel('relative abundance')
        else:
            ax.set_ylabel("%s\n(n=%i)" % (n0, g0.shape[0]))
        #  ax.set_yticks([])

        # print labels on top of the groups
        if len(graphinfo.loc[g0.index, 'group_l1'].unique()) > 1:
            ax2 = ax.twiny()
            labels = []
            pos = []
            for n, g in graphinfo.loc[g0.index, :].groupby('group_l1'):
                pos.append(g['xpos'].mean()+0.5)
                labels.append(str(n)+"\n(n=%i)" % g.shape[0])
            ax2.set_xticks(pos)
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticklabels(labels)
            ax2.xaxis.set_ticks_position("top")
            ax2.xaxis.grid()

        # print labels for group level 2
        if group_l2 is not None:
            ax3 = ax.twiny()
            ax3.set_xlim(ax.get_xlim())
            pos = []
            labels = []
            poslabel = []
            for n, g in graphinfo.loc[g0.index, :].groupby(['group_l1',
                                                            'group_l2']):
                pos.append(g.sort_values('xpos').iloc[0, :].loc['xpos'])
                poslabel.append(g['xpos'].mean())
                labels.append(str(g.sort_values('xpos').iloc[0, :]
                              .loc['group_l2']) + ("\n(n=%i)" % g.shape[0]))
            ax3.set_xticks(np.array(poslabel)+.5, minor=False)
            ax3.set_xticks(np.array(pos), minor=True)
            ax3.set_xticklabels(labels, rotation='vertical')
            ax3.xaxis.set_ticks_position("bottom")
            ax3.xaxis.grid(False, which='major')
            ax3.xaxis.grid(True, which='minor', color="black")

        # draw boxes around each group
        if len(graphinfo.loc[g0.index, 'group_l1'].unique()) > 1:
            for n, g in graphinfo.loc[g0.index, :].groupby('group_l1'):
                ax.add_patch(
                    mpatches.Rectangle(
                        (g['xpos'].min(), 0.0),   # (x,y)
                        g['xpos'].max()-g['xpos'].min()+1,          # width
                        1.0,          # height
                        fill=False,
                        edgecolor="gray",
                        linewidth=1,
                    )
                )

        # display a legend
        if ypos == 0:
            l_patches = [mpatches.Patch(color=colors[tax], label=tax)
                         for tax in vals.index]
            if l_patches[-1]._label == NAME_LOW_ABUNDANCE:
                l_patches[-1]._label = "+%i %s taxa" % (len(lowAbundandTaxa),
                                                        NAME_LOW_ABUNDANCE)
            ax.legend(handles=l_patches,
                      loc='upper left',
                      bbox_to_anchor=(1.01, 1.05))
            font0 = FontProperties()
            font0.set_weight('bold')
            title = 'Rank: %s' % rank
            if fct_aggregate is not None:
                title = ('Aggregrated "%s"\n' % fct_aggregate.__name__) + title
            ax.get_legend().set_title(title=title, prop=font0)

    if verbose:
        print("raw counts:", rawcounts.shape[1])
        print("raw meta:", metadata.shape[0])
        print("meta with counts:", meta.shape)
        print("counts with meta:", counts.shape[1])

    return fig, rank_counts, graphinfo, vals


def cluster_run(cmds, jobname, result, environment=None,
                walltime='4:00:00', nodes=1, ppn=10, pmem='8GB',
                qsub='/opt/torque-4.2.8/bin/qsub', dry=True):
    """ Submits a job to the cluster.

    Paramaters
    ----------
    cmds : [str]
        List of commands to be run on the cluster.
    jobname : str
        A name for the cluster job.
    result : path
        A file or dir holding results of a sucessful run. Don't re-submit if
        result exists.
    environment : str
        Name of a conda environment to activate.
    walltime : str
        Format hh:mm:ss maximal CPU time for the job. Default: '4:00:00'.
    nodes : int
        Number of nodes onto the job should be distributed. Defaul: 1
    ppn : int
        Number of cores within one node onto which the job should be
        distributed. Default 10.
    pmem : str
        Format 'xGB'. Memory requirement per ppn for the job, e.g. if ppn=10
        and pmem=8GB the node must have at least 80GB free memory.
        Default: '8GB'.
    qsub : path
        Path to the qsub binary. Default: /opt/torque-4.2.8/bin/qsub
    dry : bool
        Only print command instead of executing it. Good for debugging.
        Default = True
    """

    if result is None:
        raise ValueError("You need to specify a result path.")
    parent_res_dir = "/".join(result.split('/')[:-1])
    if not os.access(parent_res_dir, os.W_OK):
        raise ValueError("Parent result directory '%s' is not writable!" %
                         parent_res_dir)
    if os.path.exists(result):
        sys.stderr.write("%s already computed\n" % jobname)
        return "Result already present!"
    if jobname is None:
        raise ValueError("You need to set a jobname!")
    if len(jobname) <= 1:
        raise ValueError("You need to set non empty jobname!")

    if not isinstance(cmds, list):
        cmds = [cmds]
    for cmd in cmds:
        if "'" in cmd:
            raise ValueError("One of your commands contain a ' char. "
                             "Please remove!")
    job_cmd = " && ".join(cmds)

    # compose qsub specific details
    pwd = subprocess.check_output(["pwd"]).decode('ascii').rstrip()
    ge_cmd = (("%s -d '%s' -V -l "
               "walltime=%s,nodes=%i:ppn=%i,pmem=%s -N cr_%s") %
              (qsub, pwd, walltime, nodes, ppn, pmem, jobname))

    full_cmd = "echo '%s' | %s" % (job_cmd, ge_cmd)
    env_present = None
    if environment is not None:
        # check if environment exists
        with subprocess.Popen("conda env list | grep %s -c" % environment,
                              shell=True,
                              stdout=subprocess.PIPE) as env_present:
            if (env_present.wait() != 0):
                raise ValueError("Conda environment '%s' not present." %
                                 environment)
        full_cmd = "source activate %s && %s" % (environment, full_cmd)

    if dry is False:
        with subprocess.Popen(full_cmd,
                              shell=True, stdout=subprocess.PIPE) as task_qsub:
            qid = task_qsub.stdout.read().decode('ascii').rstrip()
            sys.stderr.write("Now wait until %s job finishes.\n" % qid)
            return qid
    else:
        print(full_cmd)
        return None


def detect_distant_groups(beta_dm, metric_name, groupings, min_group_size=5,
                          num_permutations=999):
    """Given metadata field, test for sig. group differences in beta distances.

    Parameters
    ----------
    beta_dm : skbio.stats.distance._base.DistanceMatrix
        The beta diversity distance matrix for the samples
    metric_name : str
        Please provide the metric name used to create beta_dm. This is only
        for visualization purposes.
    groupings : pandas.core.series.Series
        A group label per sample.
    min_group_size : int
        A minimal group size to be considered. Smaller group labels will be
        ignored. Default: 5.
    num_permutations : int
        Number of permutations to use for permanova test.

    Returns
    -------
    dict with following keys:
        network :          a dict of dicts to list for every pair of group
                           labels its 'p-value' and 'avgdist'
        n_per_group :      a pandas.core.series.Series reporting the remaining
                           number of samples per group
        min_group_size :   passes min_group_size
        num_permutations : passes num_permutations
        metric_name :      passes metric_name
    """

    # remove samples whose grouping in NaN
    groupings = groupings.dropna()

    # remove samples not in the distance matrix
    groupings = groupings.loc[list(beta_dm.ids)]

    # remove groups with less than minNum samples per group
    groups = [name
              for name, counts
              in groupings.value_counts().iteritems()
              if counts >= min_group_size]

    network = dict()
    for a, b in combinations(groups, 2):
        group = groupings[groupings.isin([a, b])]
        group_dm = beta_dm.filter(group.index)
        res = permanova(group_dm, group, permutations=num_permutations)

        if a not in network:
            network[a] = dict()
        network[a][b] = {'p-value': res["p-value"],
                         'avgdist':
                         np.mean([group_dm[x, y]
                                  for x in group[group == a].index
                                  for y in group[group == b].index])}

    ns = groupings.value_counts()
    return ({'network': network,
             'n_per_group': ns[ns.index.isin(groups)],
             'min_group_size': min_group_size,
             'num_permutations': num_permutations,
             'metric_name': metric_name})
