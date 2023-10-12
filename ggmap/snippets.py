from typing import Dict
import pandas as pd
import biom
from biom.util import biom_open
from itertools import repeat, chain
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import os
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
import sys
import time
from itertools import combinations
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.stats import mannwhitneyu, kruskal
import networkx as nx
import warnings
import matplotlib.cbook
import random
from tempfile import mkstemp
import pickle
from ggmap import settings
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import math
from skbio.sequence import DNA
from skbio.tree import TreeNode

if 'PROJ_LIB' not in os.environ:
    os.environ['PROJ_LIB'] = os.path.join(*([os.path.sep] + sys.executable.split(os.path.sep)[:-2] + ['share', 'proj']))
from mpl_toolkits.basemap import Basemap

settings.init()
plt.rcParams['svg.fonttype'] = 'none'


def biom2pandas(file_biom, withTaxonomy=False, astype=int):
    """ Converts a biom file into a Pandas.DataFrame

    Parameters
    ----------
    file_biom : str
        The path to the biom file.
    withTaxonomy : bool
        If TRUE, returns a second Pandas.Series with lineage information for
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
            try:
                md = table.metadata_to_dataframe('observation')
                levels = [col
                          for col in md.columns
                          if col.startswith('taxonomy_')]
                if levels == []:
                    raise ValueError(('No taxonomy information found in '
                                      'biom file.'))
                else:
                    taxonomy = md.apply(lambda row:
                                        ";".join([row[l] for l in levels]),
                                        axis=1)
                    return counts, taxonomy
            except KeyError:
                raise ValueError(('Biom file does not have any '
                                  'observation metadata!'))
        else:
            return counts
    except IOError:
        raise IOError('Cannot read file "%s"' % file_biom)


def pandas2biom(file_biom, table, taxonomy=None, err=sys.stderr):
    """ Writes a Pandas.DataFrame into a biom file.

    Parameters
    ----------
    file_biom: str
        The filename of the BIOM file to be created.
    table: a Pandas.DataFrame
        The table that should be written as BIOM.
    taxonomy : pandas.Series
        Index is taxons corresponding to table, values are lineage strings like
        'k__Bacteria; p__Actinobacteria'
    err : StringIO
        Stream onto which errors / warnings should be printed.
        Default is sys.stderr
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

        # add taxonomy metadata if provided, i.e. is not None
        if taxonomy is not None:
            if not isinstance(taxonomy, pd.core.series.Series):
                raise AttributeError('taxonomy must be a pandas.Series!')
            idx_missing_intable = set(table.index) - set(taxonomy.index)
            if len(idx_missing_intable) > 0:
                err.write(('Warning: following %i taxa are not in the '
                           'provided taxonomy:\n%s\n') % (
                          len(idx_missing_intable),
                          ", ".join(idx_missing_intable)))
                missing = pd.Series(
                    index=idx_missing_intable,
                    name='taxonomy',
                    data='k__missing_lineage_information')
                taxonomy = taxonomy.append(missing)
            idx_missing_intaxonomy = set(taxonomy.index) - set(table.index)
            if (len(idx_missing_intaxonomy) > 0) and err:
                err.write(('Warning: following %i taxa are not in the '
                           'provided count table, but in taxonomy:\n%s\n') % (
                          len(idx_missing_intaxonomy),
                          ", ".join(idx_missing_intaxonomy)))

            t = dict()
            for taxon, linstr in taxonomy.iteritems():
                # fill missing rank annotations with rank__
                orig_lineage = {annot[0].lower(): annot
                                for annot
                                in (map(str.strip, linstr.split(';')))
                                if annot != ""}
                lineage = []
                for rank in settings.RANKS:
                    rank_char = rank[0].lower()
                    if rank_char in orig_lineage:
                        lineage.append(orig_lineage[rank_char])
                    else:
                        lineage.append(rank_char+'__')
                t[taxon] = {'taxonomy': ";".join(lineage)}
            bt.add_metadata(t, axis='observation')

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


def get_great_circle_distance(p1, p2):
    """Compute great circle distance for two points.

    Parameters
    ----------
    pX : (float, float)
        Latitude, Longitude of coordinate

    Returns
    -------
    float: great circle distance in km
    """
    x1 = math.radians(p1[0])
    y1 = math.radians(p1[1])
    x2 = math.radians(p2[0])
    y2 = math.radians(p2[1])

    # Compute using the Haversine formula.

    a = math.sin((x2-x1)/2.0) ** 2.0 \
        + (math.cos(x1) * math.cos(x2) * (math.sin((y2-y1)/2.0) ** 2.0))

    # Great circle distance in radians
    angle2 = 2.0 * math.asin(min(1.0, math.sqrt(a)))

    # Convert back to degrees.
    angle2 = math.degrees(angle2)

    # Each degree on a great circle of Earth is 60 nautical miles.
    distance = 60.0 * angle2

    return distance * 1.852


def drawMap(points, basemap=None, ax=None, no_legend=False,
            color_fill_land='lightgreen', color_border_land='gray',
            color_water='lightblue', draw_country_borders=False,
            draw_latitudes=False):
    """ Plots coordinates of metadata to a worldmap.

    Parameters
    ----------
    points : a set if dicts, with mandatory key
        'coords', which itself needs to be a Pandas DataFrame with columns
            'latitude' and
            'longitude'.
        Optional keys are:
        'color' = color of points drawn onto the map (defaults to 'red'),
        'size' = diameter of drawn points (defaults to 50),
        'alpha' = transparency of points (defaults to 0.5)
        'label' = a name for the group of points, useful if more than one dict
                  is supplied
    basemap : Default is None, i.e. the whole world is plotted. By providing a
        basemap object, you can restrict the plotted map to a specific
        rectangle, e.g. to Alaska region with:
            Basemap(llcrnrlat=43.,
                    llcrnrlon=168.,
                    urcrnrlat=63.,
                    urcrnrlon=-110,
                    resolution='i',
                    projection='cass',
                    lat_0 = 90.,
                    lon_0 = -155.
    ax : plt.axis
        Default is none, i.e. create a new figure. Otherwise, provide axis onto
        which shall be drawn.
    no_legend : bool
        Default is False. Set to True to suppress drawing a legend.
    draw_country_borders: bool
        Default: False.
    draw_latitudes: False
        Default: False.
        If True, for every point a "horizontal" line for this latitude is drawn.
        Place for improvement: if many points are plotted this will surely crowed the map.
        Would be better to provide a list of latitudes?!

    Returns
    -------
    plt.axis onto which was plotted.

    Raises
    ------
    ValueError if provided list of dicts do not contain keys 'coords' or
    coords DataFrame is lacking columns 'latitude' or 'longitude'.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    map = None
    if basemap is None:
        map = Basemap(projection='robin', lon_0=180, resolution='c', ax=ax)
    else:
        map = basemap

    # Fill the globe with a blue color
    map.drawmapboundary(fill_color=color_water, color='white')
    # Fill the continents with the land color
    map.fillcontinents(color=color_fill_land, lake_color=color_water, zorder=1)
    map.drawcoastlines(color=color_border_land, zorder=1)

    if draw_country_borders:
        map.drawcountries(color=color_border_land)
    if draw_latitudes:
        map.drawparallels(list({lat for p in points for lat in p['coords']['latitude'].values}), zorder=1, color=color_border_land)

    l_patches = []
    for z, set_of_points in enumerate(points):
        if 'coords' not in set_of_points:
            raise ValueError('You need to provide key'
                             ' "coords" for every dict!')
        if 'latitude' not in set_of_points['coords'].columns:
            raise ValueError('Given "coords" need to have column "latitude"')
        if 'longitude' not in set_of_points['coords'].columns:
            raise ValueError('Given "coords" need to have column "longitude"')
        coords = set_of_points['coords'][['latitude', 'longitude']].dropna()
        x, y = map(coords.longitude.values, coords.latitude.values)
        size = 50
        if 'size' in set_of_points:
            size = set_of_points['size']
        alpha = 0.5
        if 'alpha' in set_of_points:
            alpha = set_of_points['alpha']
        color = 'red'
        if 'color' in set_of_points:
            color = set_of_points['color']
        map.scatter(x, y, marker='o', color=color, s=size,
                    zorder=2+z, alpha=alpha)
        if 'label' in set_of_points:
            l_patches.append(mpatches.Patch(color=color,
                                            label=set_of_points['label']))

    if (len(l_patches) > 0) & (no_legend is not True):
        ax.legend(handles=l_patches, loc='upper left',
                  bbox_to_anchor=(1.01, 1))

    return ax


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


def _get_sample_numbers(num_samples, fields, names):
    """Given a table about the number of available samples, this function
       returns the number of samples for the given group.

    Parameters
    ----------
    num_samples : pd.DataFrame
        Number of samples per group for all set groups.
    field : [str]
        grouping names, must be included in metadata and therefore implicitly
        in num_samples
    names : [str]
        Group name.

    Returns
    -------
    int : number of samples for the given group.

    """
    x = num_samples
    for field, name in zip(fields, names):
        if field is not None:
            x = x[x[field] == name]
    return x[0].sum()


def _collapse_counts(counts_taxonomy, rank, out=sys.stdout):
    # check that rank is a valid taxonomic rank
    if rank not in settings.RANKS + ['raw']:
        raise ValueError('"%s" is not a valid taxonomic rank. Choose from %s' %
                         (rank, ", ".join(settings.RANKS)))

    if rank != 'raw':
        # split lineage string into individual taxa names on ';' and remove
        # surrounding whitespaces. If rank does not exist return r+'__' instead
        def _splitranks(x, rank):
            try:
                return [t.strip()
                        for t
                        in x.split(";")][settings.RANKS.index(rank)]
            except AttributeError:
                # e.g. if lineage string is missing
                settings.RANKS[settings.RANKS.index(rank)].lower()[0] + "__"
            except IndexError:
                return settings.RANKS[
                    settings.RANKS.index(rank)].lower()[0] + "__"

        # add columns for each tax rank, such that we can groupby later on
        counts_taxonomy[rank] = counts_taxonomy['taxonomy'].apply(
            lambda x: _splitranks(x, rank))
        # sum counts according to the selected rank
        counts_taxonomy = counts_taxonomy.reset_index().groupby(rank).sum(numeric_only=True)
        # get rid of the old index, i.e. OTU ids, since we have grouped by some
        # rank

        if out:
            out.write('%i taxa left after collapsing to %s.\n' %
                      (counts_taxonomy.shape[0], rank))
    else:
        sample_cols = set(counts_taxonomy.columns) - set(['taxonomy'])
        counts_taxonomy = counts_taxonomy.loc[:, sample_cols]

    return counts_taxonomy


def collapseCounts_objects(counts, rank, taxonomy, out=sys.stdout):
    """
    Parameters
    ----------
    counts : pd.DataFrame
        Feature table in raw format, i.e. index is OTU-IDs or deblur seqs,
        while columns are samples.
    rank : str
        Set taxonomic level to collapse abundances. Use 'raw' to de-activate
        collapsing.
    taxonomy : pd.Series
        Index are OTU-IDs or deblur seqs, values are ; separated taxonomic
        lineages.
    verbose : bool
        Default is true. Report messages if true.
    out : StringIO
        Buffer onto which messages should be written. Default is sys.stdout.

    Returns
    -------
    pd.DataFrame of collapsed counts.
    """
    tax = taxonomy.copy()
    tax.name = 'taxonomy'

    return _collapse_counts(
        pd.merge(
            counts, tax.to_frame(),
            how='left', left_index=True, right_index=True),
        rank,
        out=out), tax


def collapseCounts(file_otutable, rank,
                   file_taxonomy=None,
                   verbose=True, out=sys.stdout, astype=int):
    """Collapses features of an OTU table according to their taxonomic
       assignment and a given rank.

    Parameters
    ----------
    file_otutable : file
        Path to a biom OTU table
    rank : str
        Set taxonomic level to collapse abundances. Use 'raw' to de-activate
        collapsing.
    file_taxonomy : file
        Taxonomy information is read from the biom file. Except you provide an
        alternative taxonomy in terms of a two column file. First column must
        contain feature ID (OTUid or sequence), second colum is the ; separated
        lineage string.
        Default is None, i.e. using taxonomy from biom file.
    verbose : bool
        Default is true. Report messages if true.
    out : StringIO
        Buffer onto which messages should be written. Default is sys.stdout.
    astype : type
        datatype into each value of the biom table is casted. Default: int.
        Use e.g. float if biom table contains relative abundances instead of
        raw reads.

    Returns
    -------
    Pandas.DataFrame: counts of collapsed taxa.
    """
    # check that biom table can be read
    if not os.path.exists(file_otutable):
        raise IOError('OTU table file not found')

    counts, taxonomy, rank_counts = None, None, pd.DataFrame()
    if file_taxonomy is None:
        counts, taxonomy = biom2pandas(file_otutable, withTaxonomy=True,
                                       astype=astype)
        taxonomy.name = 'taxonomy'
        rank_counts = pd.merge(counts, taxonomy.to_frame(), how='left',
                               left_index=True, right_index=True)
    else:
        # check that taxonomy file exists
        if (not os.path.exists(file_taxonomy)) and (rank != 'raw'):
            raise IOError('Taxonomy file not found!')

        rank_counts = biom2pandas(file_otutable, astype=astype)
        if rank != 'raw':
            taxonomy = pd.read_csv(file_taxonomy, sep="\t", header=None,
                                   names=['otuID', 'taxonomy'],
                                   usecols=[0, 1])  # only parse 2 first cols
            taxonomy['otuID'] = taxonomy['otuID'].astype(str)
            taxonomy.set_index('otuID', inplace=True)
            # add taxonomic lineage information to the counts as
            # column "taxonomy"
            rank_counts = pd.merge(rank_counts, taxonomy, how='left',
                                   left_index=True, right_index=True)

    return _collapse_counts(rank_counts, rank, out=out), taxonomy


def plotTaxonomy(file_otutable,
                 metadata,
                 group_l0=None,
                 group_l1=None,
                 group_l2=None,
                 rank='Phylum',
                 file_taxonomy=settings.FILE_REFERENCE_TAXONOMY,
                 verbose=True,
                 reorder_samples=False,
                 reorder_taxa=True,
                 print_sample_labels=False,
                 sample_label_column=None,
                 print_meanrelabunances=False,
                 minreadnr=50,
                 plottaxa=None,
                 plotTopXtaxa=None,
                 fct_aggregate=None,
                 no_top_labels=False,
                 grayscale=False,
                 out=sys.stdout,
                 taxonomy_from_biom=False,
                 no_sample_numbers=False,
                 colors=None,
                 min_abundance_grayscale=0,
                 legend_use_last_X_labels=None,
                 ax=None):
    """Plot taxonomy.

    Parameters
    ----------
    file_otutable : file
        Path to a biom OTU table
        Alternatively, a pd.DataFrame holding counts.
    metadata : pandas.DataFrame
        metadata
    file_taxonomy : file
        Path to a GreenGenes taxonomy file.
        Alternatively, a pd.Series holding lineage strings.
    reorder_samples : Bool
        True = sort samples in each group according to abundance of most
        abundant taxon
    reorder_taxa : Bool
        Default = True.
        Orders taxa by mean abundance across all samples.
    print_sample_labels : Bool
        True = print sample names on x-axis. Use only for small numbers of
        samples!
    sample_label_column : str
        Default: None
        Use column <sample_label_column> from metadata to print sample labels,
        instead of metadata.index.
    print_meanrelabunances : Bool
        Default: False.
        If True, print mean relative abundance of taxa in legend.
    minreadnr : int
        min number of reads a taxon need to have to be plotted
    plotTopXtaxa : int
        Only plot the X most abundant taxa.
    plottaxa : [str]
        Only plot abundances for taxa IDs provided. If None, all taxa are
        plotted. Default: None
    rank : str
        Set taxonomic level to collapse abundances. Use 'raw' to de-activate
        collapsing.
    fct_aggregate : function
        A numpy function to aggregate over several samples.
    no_top_labels : Bool
        If True, print no labels on top of the bars. Default is False.
    grayscale : Bool
        If True, plot low abundant taxa with gray scale values.
    taxonomy_from_biom : Bool
        Default is False. If true, read taxonomy information from input biom
        file.
    no_sample_numbers : Bool
        Default is False. If true, no n= sample numbers will be reported.
    colors : dict(taxon: (r, g, b))
        Provide a predefined color dictionary to use same colors for several
        plots. Default is an empty dictionary.
        Format: key = taxon name,
        Value: a triple of RGB float values.
    min_abundance_grayscale : float
        Stop drawing gray rectangles for low abundant taxa if their relative
        abundance is below this threshold. Saves time and space.
    legend_use_last_X_labels : int
        Default: None, i.e. leave legend text as is
        If set to e.g. 2, legend will contain lineage strings of the last two
        known labels.
    ax : plt.axis
        Plot on this axis instead of creating a new figure. Only works if
        number of group levels is <= 2.

    Returns
    -------
    fig, rank_counts, graphinfo, vals, color-dict
    """

    NAME_LOW_ABUNDANCE = 'low abundance'
    GRAYS = ['#888888', '#EEEEEE', '#999999', '#DDDDDD', '#AAAAAA',
             '#CCCCCC', '#BBBBBB']
    random.seed(42)

    if metadata.index.value_counts().max() > 1:
        raise ValueError(
            ('The following %i sample(s) occure several times in your '
             'metadata. Please de-replicate and try again:\n\t%s\n') % (
             sum(metadata.index.value_counts() > 1),
             '\n\t'.join(
                set(metadata[metadata.index.value_counts() > 1].index))
             ))

    # Parameter checks: check that grouping fields are in metadata table
    for i, field in enumerate([group_l0, group_l1, group_l2]):
        if field is not None:
            if field not in metadata.columns:
                raise ValueError(('Column "%s" for grouping level %i is not '
                                  'in metadata table!') % (field, i))

    ft = file_taxonomy
    taxonomy = None
    if taxonomy_from_biom:
        ft = None
    if isinstance(file_otutable, pd.DataFrame) and \
       isinstance(file_taxonomy, pd.Series):
        rawcounts, taxonomy = collapseCounts_objects(
            file_otutable, rank, file_taxonomy, out=out)
    else:
        rawcounts, taxonomy = collapseCounts(
            file_otutable, rank, file_taxonomy=ft, verbose=verbose, out=out)


    # restrict to those samples for which we have metadata AND counts
    meta = metadata.loc[[idx
                         for idx in metadata.index
                         if idx in rawcounts.columns], :]
    rank_counts = rawcounts.loc[:, meta.index]
    if (out is not None) and verbose:
        out.write('%i samples left with metadata and counts.\n' %
                  meta.shape[0])

    lowAbundandTaxa = rank_counts[(rank_counts.sum(axis=1) < minreadnr)].index
    highAbundantTaxa = rank_counts[(rank_counts.sum(axis=1) >=
                                    minreadnr)].index

    # normalize to 1 in each sample
    rank_counts /= rank_counts.sum(axis=0)

    # filter low abundant taxa
    if (grayscale is False) & (len(lowAbundandTaxa) > 0):
        lowReadTaxa = rank_counts.loc[lowAbundandTaxa, :].sum(axis=0)
        lowReadTaxa.name = NAME_LOW_ABUNDANCE
        rank_counts = rank_counts.loc[highAbundantTaxa, :]
        #rank_counts = rank_counts.append(lowReadTaxa)  # deprecated
        rank_counts = pd.concat([rank_counts, lowReadTaxa.to_frame().T], axis=0)
        if (out is not None) and verbose:
            out.write('%i taxa left after filtering low abundant.\n' %
                      (rank_counts.shape[0]-1))

    # restrict to those taxa that are asked for in plottaxa
    if plottaxa is not None:
        rank_counts = rank_counts.loc[[t for t in plottaxa if t in rank_counts.index], :]
        if (out is not None) and verbose:
            out.write('%i taxa left after restricting to provided list.\n' %
                      (rank_counts.shape[0]))

    if plotTopXtaxa is not None:
        rank_counts = rank_counts.loc[
            rank_counts.mean(axis=1).sort_values(ascending=False)
            .iloc[:plotTopXtaxa].index, :]
        if (out is not None) and verbose:
            out.write('%i taxa left after restricting to top %i.\n' %
                      (plotTopXtaxa, rank_counts.shape[0]))
    # all for plotting
    # sort taxa according to sum of abundance
    taxaidx = rank_counts.index
    if reorder_taxa:
        taxaidx = list(rank_counts.mean(axis=1).sort_values(ascending=False).index)
    if (grayscale is False) & (len(lowAbundandTaxa) > 0):
        taxaidx = [taxon
                   for taxon in taxaidx
                   if taxon != NAME_LOW_ABUNDANCE] + [NAME_LOW_ABUNDANCE]
    elif grayscale is True:
        taxaidx = [taxon for taxon in taxaidx if taxon in highAbundantTaxa] +\
                  [taxon for taxon in taxaidx if taxon not in highAbundantTaxa]
    rank_counts = rank_counts.loc[taxaidx, :]

    levels = [f for f in [group_l2, group_l1, group_l0] if f is not None]

    # keeping track of correct sample numbers
    num_samples = meta.shape[0]
    if levels != []:
        num_samples = meta.groupby(levels).size().reset_index()

    # aggregate over samples
    if fct_aggregate is not None:
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
            grps1 = g0.groupby(group_l1, sort=False)
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
    availColors = \
        sns.color_palette('Paired', 12) +\
        sns.color_palette('Dark2', 12) +\
        sns.color_palette('Pastel1', 12)
    if colors is None:
        colors = dict()
    colors[NAME_LOW_ABUNDANCE] = 'white'
    for i in range(0, vals.shape[0]):
        taxon = vals.index[i]
        if taxon not in colors:
            colors[taxon] = availColors[len(colors) % len(availColors)]

    # plot the actual thing
    sns.set()
    if (ax is not None):
        if len(grps0) > 1:
            raise Exception('You cannot provide an ax if number of '
                            'grouping levels is > 2!')
        else:
            axarr = ax
            fig = ax
    else:
        fig, axarr = plt.subplots(len(grps0), 1)
    num_saved_boxes = 0
    for ypos, (n0, g0) in enumerate(graphinfo.groupby('group_l0')):
        if group_l0 is None:
            ax = axarr
        else:
            ax = axarr[ypos]
        for i in range(0, vals.shape[0]):
            taxon = vals.index[i]
            color = colors[taxon]
            if taxon in lowAbundandTaxa:
                color = random.choice(GRAYS)
            y_prev = None
            for j, (name, g1_idx) in enumerate(graphinfo.loc[g0.index, :]
                                               .groupby('group_l1')):
                if i == 0:
                    y_prev = [0] * g1_idx.shape[0]
                else:
                    y_prev = vals.loc[:, g1_idx.sort_values(by='xpos').index]\
                        .iloc[i-1, :]
                    if grayscale & (y_prev.min() > 1-min_abundance_grayscale):
                        num_saved_boxes += 1
                        continue
                y_curr = vals.loc[:, g1_idx.sort_values(by='xpos').index]\
                    .iloc[i, :]
                xpos = g1_idx.sort_values(by='xpos')['xpos']

                ax.fill_between(_shiftLeft(_repMiddleValues(xpos)),
                                _repMiddleValues(y_prev),
                                _repMiddleValues(y_curr),
                                color=color)

            if grayscale & \
               (vals.iloc[i, :].min() >= 1-min_abundance_grayscale):
                num_saved_boxes += len(graphinfo.loc[g0.index,
                                                     'group_l1'].unique())
                break

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
                label_value = idx
                if (sample_label_column is not None) and \
                   (sample_label_column in metadata.columns) and \
                   (idx in metadata.index) and \
                   (pd.notnull(meta.loc[idx, sample_label_column])):
                    label_value = meta.loc[idx, sample_label_column]
                if '###' in label_value:
                    label = "%s" % idx.split('###')[-1]
                else:
                    label = label_value
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
        ax.set_facecolor('white')

        if group_l0 is None:
            ax.set_ylabel('relative abundance')
        else:
            label = n0
            if no_sample_numbers is False:
                label = "%s\n(n=%i)" % (label, _get_sample_numbers(
                    num_samples, [group_l0], [n0]))
            ax.set_ylabel(label)

        # print labels on top of the groups
        if not no_top_labels:
            if len(graphinfo.loc[g0.index, 'group_l1'].unique()) > 1:
                ax2 = ax.twiny()
                labels = []
                pos = []
                for n, g in graphinfo.loc[g0.index, :].groupby('group_l1'):
                    pos.append(g['xpos'].mean()+0.5)
                    label = str(n)
                    if no_sample_numbers is False:
                        label += "\n(n=%i)" % _get_sample_numbers(
                            num_samples, [group_l0, group_l1], [n0, n])
                    labels.append(label)
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
                label = str(g.sort_values('xpos').iloc[0, :].loc['group_l2'])
                if no_sample_numbers is False:
                    label += "\n(n=%i)" % _get_sample_numbers(
                        num_samples,
                        [group_l0, group_l1, group_l2],
                        [n0, n[0], n[1]])
                labels.append(label)
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
            l_patches = []
            for tax in vals.index:
                if (tax in highAbundantTaxa) | (tax == NAME_LOW_ABUNDANCE):
                    tax_name = tax
                    if (rank == "raw") & (legend_use_last_X_labels is not None) & isinstance(file_taxonomy, pd.Series):
                        if (tax in taxonomy.index):
                            tax_name = ';'.join([l for l in taxonomy.loc[tax].split(';') if not l.strip().endswith('__')][-1 * legend_use_last_X_labels:])
                        else:
                            raise ValueError("taxon '%s' not included in taxonomy" % tax)
                    label_text = tax_name
                    if print_meanrelabunances:
                        label_text = "%.2f %%: %s" % (
                            rank_counts.loc[tax, :].mean()*100, tax_name)
                    l_patches.append(mpatches.Patch(color=colors[tax],
                                                    label=label_text))
            label_low_abundant = "+%i %s taxa" % (len(lowAbundandTaxa),
                                                  NAME_LOW_ABUNDANCE)
            if grayscale:
                l_patches.append(mpatches.Patch(color='gray',
                                                label=label_low_abundant))
            else:
                if l_patches[-1]._label == NAME_LOW_ABUNDANCE:
                    l_patches[-1]._label = label_low_abundant
            ax.legend(handles=l_patches,
                      loc='upper left',
                      bbox_to_anchor=(1.01, 1.05))
            font0 = FontProperties()
            font0.set_weight('bold')
            title = 'Rank: %s' % rank
            if legend_use_last_X_labels is not None:
                title += ", displayed = last %i known ranks" % legend_use_last_X_labels
            if fct_aggregate is not None:
                title = ('Aggregrated "%s"\n' % fct_aggregate.__name__) + title
            ax.get_legend().set_title(title=title, prop=font0)

    if (out is not None) and verbose:
        out.write("raw counts: %i\n" % rawcounts.shape[1])
        out.write("raw meta: %i\n" % metadata.shape[0])
        out.write("meta with counts: %i samples x %i fields\n" % meta.shape)
        out.write("counts with meta: %i\n" % rank_counts.shape[1])
        if grayscale:
            out.write("saved plotting %i boxes.\n" % num_saved_boxes)

    return fig, rank_counts, graphinfo, vals, colors


def _time_torque2slurm(t_time):
    """Convertes run-time resource string from Torque to Slurm.
    Input format is hh:mm:ss, output is <days>-<hours>:<minutes>

    Parameters
    ----------
    t_time : str
        Input time duration in format hh:mm:ss

    Returns
    -------
    Slurm compatible time duration.
    """
    t_hours, t_minutes, t_seconds = map(int, t_time.split(':'))
    s_minutes = (t_seconds // 60) + t_minutes
    s_hours = (s_minutes // 60) + t_hours
    s_minutes = s_minutes % 60
    s_days = s_hours // 24
    s_hours = s_hours % 24

    # set a minimal run time, if Torque time is < 60 seconds
    if (s_days == 0) and (s_hours == 0) and (s_minutes == 0):
        s_minutes = 1

    return "%i-%i:%i" % (s_days, s_hours, s_minutes)


def _add_timing_cmds(commands, file_timing):
    """Change list of commands, such that system's time is used to trace
       run-time.

    Parameters
    ----------
    commands : [str]
        List of commands.
    file_timing : str
        Filepath to the file into which timing information shall be written

    Returns
    -------
    [str] list of changed commands with timing capability.
    """
    timing_cmds = []
    # report machine name
    timing_cmds.append('uname -a > %s' % file_timing)
    # report commands to be executed (I have problems with quotes)
    # timing_cmds.append('echo `%s` >> ${PBS_JOBNAME}.t${PBS_JOBID}'
    #                    % '; '.join(cmds))
    # add time to every command
    for cmd in commands:
        # cd cannot be timed and any attempt will fail changing the
        # directory
        if cmd.startswith('cd ') or\
           cmd.startswith('module load ') or\
           cmd.startswith('var_') or\
           cmd.startswith('export ') or\
           cmd.startswith('ulimit '):
                timing_cmds.append(cmd)
        elif cmd.startswith('if [ '):
            ifcon, rest = re.findall(
                '(if \[.+?\];\s*then\s*)(.+)', cmd, re.IGNORECASE)[0]
            timing_cmds.append(('%s '
                                '%s '
                                '-v '
                                '-o %s '
                                '-a %s') %
                               (ifcon, settings.EXEC_TIME, file_timing, rest))
        else:
            timing_cmds.append(('%s '
                                '-v '
                                '-o %s '
                                '-a %s') %
                               (settings.EXEC_TIME, file_timing, cmd))
    return timing_cmds


def cluster_run(cmds, jobname, result, environment=None,
                walltime='4:00:00', nodes=1, ppn=10, pmem='8GB',
                gebin=settings.GRIDENGINE_BINDIR, dry=True, wait=False,
                file_qid=None, file_condaenvinfo=None, out=sys.stdout,
                err=sys.stderr, timing=False, file_timing=None, array=1,
                use_grid=settings.USE_GRID,
                force_slurm=False):
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
    gebin : path
        Path to the dir holding SGE binaries.
        Default: /opt/torque-4.2.8/bin
    dry : bool
        Only print command instead of executing it. Good for debugging.
        Default = True
    wait : bool
        Wait for job completion before qsub's return
    file_qid : str
        Default None. Create a file containing the qid of the submitted job.
        This will ease identification of TMP working directories.
    file_condaenvinfo : str
        Default: None.
        If specified, AND environment is not None,
        the result of "conda list --name X" is written to this file.
    out : StringIO
        Buffer onto which messages should be printed. Default is sys.stdout.
    err : StringIO
        Default: sys.stderr.
        Buffer for status reports.
    timing : bool
        If True than add time output to every command and store in cr_*.t*
        file. Default is False.
    file_timing : str
        Default: None
        Define filepath into which timeing information shall be written.
    array : int
        Default: 1
        If > 1 than an array job is submitted. Make sure in- and outputs can
        deal with ${PBS_ARRAY_INDEX}!
        Only available for Torque.
    use_grid : bool
        Defaul: True.
        If False, commands are executed locally instead of submitting them to
        a HPC (= either Torque or Slurm).
    force_slurm : bool
        Default: False.
        If True, cluster_run is enforeced to choose slurm instead of auto
        detection based on machine node name.

    Returns
    -------
    Cluster job ID as str.
    """

    if result is None:
        raise ValueError("You need to specify a result path.")
    parent_res_dir = "/".join(result.split('/')[:-1])
    if not os.access(parent_res_dir, os.W_OK):
        raise ValueError("Parent result directory '%s' is not writable!" %
                         parent_res_dir)
    if file_qid is not None:
        if not os.access('/'.join(file_qid.split('/')[:-1]), os.W_OK):
            raise ValueError("Cannot write qid file '%s'." % file_qid)
    if os.path.exists(result):
        if err:
            err.write("%s already computed\n" % jobname)
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
    if timing:
        if file_timing is None:
            file_timing = '${PBS_JOBNAME}.t${PBS_JOBID}'
        cmds = _add_timing_cmds(cmds, file_timing)

    cmd_list = ""
    cmd_conda = ""
    env_present = None
    if environment is not None:
        if file_condaenvinfo is None:
            file_condaenvinfo = ""
        else:
            file_condaenvinfo = " > %s" % file_condaenvinfo
        # check if environment exists
        with subprocess.Popen("%s/condabin/conda list -n %s %s" % (settings.DIR_CONDA, environment, file_condaenvinfo),
                              shell=True,
                              stdout=subprocess.PIPE) as env_present:
            if (env_present.wait() != 0):
                raise ValueError("Conda environment '%s' not present." %
                                 environment)
        if settings.GRIDNAME == 'JLU':
            # but remember to to create the ~/.bash_profile file and copy and paste conda init script from .bashrc!
            if use_grid is False:
                cmd_conda = "source %s/etc/profile.d/conda.sh; conda activate %s; " % (settings.DIR_CONDA, environment)
            else:
                cmd_conda = "conda activate %s; " % (environment)
        elif settings.GRIDNAME == 'JLU_SLURM':
            cmd_conda = "source %s/etc/profile.d/conda.sh; conda activate %s; " % (settings.DIR_CONDA, environment)
        else:
            cmd_conda = "source %s/etc/profile.d/conda.sh; %s/condabin/conda activate %s; " % (
                settings.DIR_CONDA, settings.DIR_CONDA, environment)

    slurm = False
    if use_grid is False:
        cmd_list += '%s for %s in `seq 1 %i`; do %s; done' % (
            cmd_conda, settings.VARNAME_PBSARRAY, array, " && ".join(cmds))
    else:
        pwd = subprocess.check_output(["pwd"]).decode('ascii').rstrip()

        if (settings.GRIDNAME == 'USF') or (settings.PREFER_SLURM):
            slurm = True
        else:
            slurm = False
        with subprocess.Popen("which srun" if slurm else "which qsub",
                              shell=True, stdout=subprocess.PIPE,
                              executable="bash") as call_x:
            if call_x.wait() != 0:
                msg = ("You don't seem to have access to a grid!")
                if dry:
                    if err is not None:
                        err.write(msg)
                else:
                    raise ValueError(msg)
        if force_slurm:
            slurm = True

        if slurm is False:
            highmem = ''
            if settings.GRIDNAME == 'barnacle':
                if ppn * int(pmem[:-2]) > 250:
                    highmem = ':highmem'
            files_loc = ''
            if file_qid is not None:
                files_loc = ' -o %s/ -e %s/ ' % tuple(
                    ["/".join(file_qid.split('/')[:-1])] * 2)

            flag_array = ''
            if array > 1:
                if settings.GRIDNAME == 'barnacle' or settings.GRIDNAME == 'JLU':
                    flag_array = '-t 1-%i' % array
                elif settings.GRIDNAME == 'HPCHHU':
                    flag_array = '-J 1-%i' % array
            resources = " -l walltime=%s,nodes=%i%s:ppn=%i,mem=%s " % (
                walltime, nodes, highmem, ppn, pmem)
            if settings.GRIDNAME == 'JLU':
                # further differentiate between old and new 18.04 cluster (08.04.2020)
                arg_multislot = " -pe multislot %i " % ppn
                #if settings.GRIDENGINE_BINDIR == '/usr/bin/':
                    # according to Burkhard, the "new cluster" doesn't have multislots yet
                    # UPDATE: 2021-01-06: "das PE ist da, sollte auch funktionieren"
                #    arg_multislot = ""
                pmem_value = pmem
                if pmem is None:
                    pmem_value = '8GB'
                else:
                    pmem_value = pmem[:-1] if pmem.upper().endswith('B') else pmem
                resources = " -l virtual_free=%s %s -S /bin/bash " % (pmem_value, arg_multislot)
            ge_cmd = (
                ("%s/qsub %s %s -V %s -N cr_%s %s %s -r y") %
                (gebin,
                 '-A %s' % settings.GRID_ACCOUNT if settings.GRID_ACCOUNT != "" else "",
                 "-d '%s'" % pwd if settings.GRIDNAME == 'barnqacle' else '',
                 resources,
                 jobname, flag_array, files_loc))
            cmd_list += "echo '%s%s' | %s" % (cmd_conda, " && ".join(cmds), ge_cmd)
        else:
            slurm_script = "#!/bin/bash\n\n"
            slurm_script += '#SBATCH --job-name=cr_%s\n' % jobname
            slurm_script += '#SBATCH --output=%s/slurmlog-%%x-%%A.%%a.log\n' % (pwd if file_qid is None else os.path.abspath(os.path.dirname(file_qid)))
            slurm_script += '#SBATCH --partition=%s\n' % settings.GRID_ACCOUNT
            slurm_script += '#SBATCH --ntasks=1\n'
            slurm_script += '#SBATCH --cpus-per-task=%i\n' % ppn
            slurm_script += '#SBATCH --mem-per-cpu=%s\n' % (pmem.upper() if pmem is not None else '8GB')
            slurm_script += '#SBATCH --time=%s\n' % _time_torque2slurm(
                walltime)
            slurm_script += '#SBATCH --array=1-%i\n' % array
            slurm_script += '#SBATCH --mail-type=END,FAIL\n'
            slurm_script += '#SBATCH --mail-user=sjanssen@ucsd.edu\n\n'
            slurm_script += 'srun uname -a\n'

            for cmd in cmds:
                slurm_script += '%s\n' % (cmd.replace(
                    '${%s}' % settings.VARNAME_PBSARRAY, '${SLURM_ARRAY_TASK_ID}'))
            if file_qid is not None:
                file_script = os.path.dirname(file_qid) + '/slurm_script.sh'
            else:
                _, file_script = mkstemp(suffix='.slurm.sh')
            f = open(file_script, 'w')
            f.write(slurm_script)
            f.close()
            # if on jupyterlab from BCF@JLU, some slurm vars are predefined for the
            # spawner process of the jupyterlab. We need to unset this specific
            # variable to avoid slurm complaining about other resource requests.
            if settings.GRIDNAME == 'JLU_SLURM':
                cmd_list += 'unset SLURM_MEM_PER_NODE && '
            cmd_list += '%s %ssbatch %s' % (cmd_conda, settings.GRIDENGINE_BINDIR, file_script)

    if dry is True:
        if use_grid and slurm:
            out.write('CONTENT OF %s:\n' % file_script)
            out.write(slurm_script + "\n\n")
        out.write(cmd_list + "\n")
        return None
    else:
        if use_grid is True:
            with subprocess.Popen(
                    cmd_list, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable='/bin/bash') as task_qsub:
                err_msg = task_qsub.stderr.read()
                if err_msg != b"":
                    raise ValueError("Error in submitting job via qsub:\n%s" % err_msg.decode('ascii'))
                qid = task_qsub.stdout.read().decode('ascii').rstrip()
                if settings.GRIDNAME == 'JLU':
                    qid = qid.split(" ")[2]
                if slurm:
                    qid = qid.split()[-1]
                    if file_qid is not None:
                        os.remove(file_script)
                if file_qid is not None:
                    f = open(file_qid, 'w')
                    f.write('Cluster job ID is:\n%s\n' % qid)
                    f.close()
                job_ever_seen = False
                if wait:
                    err.write(
                        "\nWaiting for %s-cluster job %s to complete: " % ('slurm' if slurm else 'sge', qid))
                    while True:
                        if slurm:
                            with subprocess.Popen(
                                    ['squeue', '--job', qid],
                                    stdout=subprocess.PIPE) as task_squeue:
                                with subprocess.Popen(
                                        ['wc', '-l'], stdin=task_squeue.stdout,
                                        stdout=subprocess.PIPE) as task_wc:
                                    poll_status = \
                                        int(task_wc.stdout.read().decode(
                                            'ascii').rstrip())
                            # Two ore more if polling gives a table with header
                            # and one status line, i.e. job is still on the
                            # grid. Translate that to 0 of Torque.
                            # If table has only one line, i.e. the header, job
                            # terminated (hopefully successful), translate that
                            # to 1 of Torque
                            if poll_status >= 2:
                                poll_status = 0
                            else:
                                poll_status = 1
                        else:
                            poll_stati = []
                            for i in range(array):
                                p = subprocess.call(
                                    "%s/qstat %s %s" %
                                    (gebin,
                                     ' -j ' if settings.GRIDNAME == 'JLU' else '',
                                     qid.replace('[]', '[%i]' % (i+1))),
                                    shell=True)
                                poll_stati.append(p == 0)
                            if any(poll_stati):
                                poll_status = 0
                            else:
                                poll_status = 127  # some number != 0
                        if (poll_status != 0) and job_ever_seen:
                            err.write(' finished.')
                            break
                        elif (poll_status == 0) and (not job_ever_seen):
                            job_ever_seen = True
                        err.write('.')
                        time.sleep(10)
                else:
                    err.write("Now wait until %s job finishes.\n" % qid)
                return qid
        else:
            #if settings.GRIDNAME == 'JLU':
            #    cmd_list = 'source ~/.profile && ' + cmd_list
            with subprocess.Popen(cmd_list,
                                  shell=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  executable="bash") as call_x:
                if (call_x.wait() != 0):
                    out, err = call_x.communicate()
                    raise ValueError((
                        "SYSTEM CALL FAILED.\n==== STDERR ====\n%s"
                        "\n\n==== STDOUT ====\n%s\n") % (
                            err.decode("utf-8", 'backslashreplace'),
                            out.decode("utf-8", 'backslashreplace')))
                return call_x.pid


def detect_distant_groups_alpha(alpha, groupings,
                                min_group_size=21,
                                fct_test=mannwhitneyu):
    """Given metadata field, test for sig. group differences in alpha
       distances.

    Parameters
    ----------
    alpha : pandas.core.series.Series
        The alpha diversities for the samples
    groupings : pandas.core.series.Series
        A group label per sample.
    min_group_size : int
        A minimal group size to be considered. Smaller group labels will be
        ignored. Default: 21.
    fct_test : function
        Default: mannwhitneyu
        The statistical test that is used to test for differences between
        groups.

    Returns
    -------
    dict with following keys:
        network :          a dict of dicts to list for every pair of group
                           labels its 'p-value' and 'avgdist'
        n_per_group :      a pandas.core.series.Series reporting the remaining
                           number of samples per group
        min_group_size :   passes min_group_size
        num_permutations : None
        metric_name :      passes metric_name
        group_name :       passes the name of the grouping
        fct_name :         string name of test function
    """
    # remove samples whose grouping in NaN
    groupings = groupings.dropna()

    # remove samples for which we don't have alpha div measures
    groupings = groupings.loc[sorted(set(groupings.index) & set(alpha.index))]

    # remove groups with less than minNum samples per group
    groups = sorted([name
                     for name, counts
                     in groupings.value_counts().iteritems()
                     if counts >= min_group_size])

    network = dict()
    for a, b in combinations(groups, 2):
        args = {'a': alpha.loc[groupings[groupings == a].index],
                'b': alpha.loc[groupings[groupings == b].index]}
        if fct_test == mannwhitneyu:
            args['alternative'] = 'two-sided'
            args['x'] = args.pop('a')
            args['y'] = args.pop('b')
        elif fct_test == kruskal:
            args['x'] = list(args.pop('a').values)
            args['y'] = list(args.pop('b').values)

        if a not in network:
            network[a] = dict()

        try:
            if fct_test == kruskal:
                res = fct_test(args['x'], args['y'])
            else:
                res = fct_test(**args)
            network[a][b] = {'p-value': res.pvalue,
                             'test-statistic': res.statistic}
        except ValueError as e:
            if str(e) == 'All numbers are identical in mannwhitneyu':
                network[a][b] = {'p-value': 1,
                                 'test-statistic': 'all numbers are identical'}
            else:
                raise e

    ns = groupings.value_counts()
    return ({'network': network,
             'n_per_group': ns[ns.index.isin(groups)],
             'min_group_size': min_group_size,
             'num_permutations': None,
             'metric_name': alpha.name,
             'group_name': groupings.name,
             'fct_name': fct_test.__name__})


def detect_distant_groups(beta_dm, metric_name, groupings, min_group_size=5,
                          num_permutations=999, err=None,
                          fct_test=permanova):
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
    fct_test : function
        Default: skbio.stats.distance.permanova
        Python function to execute test.
        Valid functions are "permanova" or "anosim" from skbio.stats.distance.

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
        group_name :       passes the name of the grouping
        fct_name :         fct_test.__name__
    """

    # remove samples whose grouping in NaN
    groupings = groupings.dropna()

    # remove samples not in the distance matrix
    groupings = groupings.loc[sorted(set(groupings.index) & set(beta_dm.ids))]

    # remove groups with less than minNum samples per group
    groups = sorted([name
                     for name, counts
                     in groupings.value_counts().items()
                     if counts >= min_group_size])

    network = dict()
    for a, b in combinations(groups, 2):
        if err is not None:
            err.write('%s vs %s\n' % (a, b))
        group = groupings[groupings.isin([a, b])]
        group_dm = beta_dm.filter(group.index)
        # be on the safe side and protect against https://github.com/biocore/scikit-bio/issues/1877
        # however, my tests did not indicate that I have used permanova incorrectly (smj: 2023-09-21)
        res = fct_test(group_dm, group.to_frame(),
                       column=group.name, permutations=num_permutations)

        if a not in network:
            network[a] = dict()
        network[a][b] = {'p-value': res["p-value"],
                         'test-statistic': res["test statistic"],
                         'avgdist':
                         np.mean([group_dm[x, y]
                                  for x in group[group == a].index
                                  for y in group[group == b].index])}

    ns = groupings.value_counts()
    return ({'network': network,
             'n_per_group': ns[ns.index.isin(groups)],
             'min_group_size': min_group_size,
             'num_permutations': num_permutations,
             'metric_name': metric_name,
             'group_name': groupings.name,
             'fct_name': fct_test.__name__})


def _getfirstsigdigit(number):
    """Given a float between < 1, determine the position of first non-zero
       digit.
    """
    if number >= 1:
        return 0
    num_digits = 1
    while ('%f' % number).split('.')[1][num_digits-1] == '0':
        num_digits += 1
    return num_digits


def groups_is_significant(group_infos, pthresh=0.05):
    """Checks if a network has significantly different groups.

    Parameters
    ----------
    group_infos : dict()
        result of a detect_distant_groups() run
    pthresh : float
        The maximal p-value of a group difference to be considered significant.
        It will be corrected for multiple hypothesis testing in a naive way,
        i.e. by dividing with number of all pairwise groups.

    Returns
    -------
    Boolean.
    """
    numComp = len(list(combinations(group_infos['n_per_group'].keys(), 2)))
    for a in group_infos['network'].keys():
        for b in group_infos['network'][a].keys():
            if group_infos['network'][a][b]['p-value'] < pthresh / numComp:
                return True
    return False


def plotDistant_groups(network, n_per_group, min_group_size, num_permutations,
                       metric_name, group_name, fct_name="permanova",
                       pthresh=0.05, _type='beta', draw_edgelabel=False,
                       ax=None, edge_color_sig=None, print_title=True,
                       edgelabel_decimals=2):
    """Plots pairwise beta diversity group relations (obtained by
       'detect_distant_groups')

    Parameters
    ----------
    Most parameters are direct outputs of detect_distant_groups, thus you can
    pass **res = detect_distant_groups(...) here
    network : dict
        a dict of dicts to list for every pair of group labels its 'p-value'
        and 'avgdist'
    n_per_group : pandas.core.series.Series
        reporting the remaining number of samples per group
    min_group_size : int
        The minimal group size that was considered.
    num_permutations : int
        Number of permutations used for permanova test.
    metric_name : str
        The beta diversity metric name used.
    group_name : str
        A label for the grouping criterion.
    fct_name : str
        Default: None
        The name of the statistical test function used.
    pthresh : float
        The maximal p-value of a group difference to be considered significant.
        It will be corrected for multiple hypothesis testing in a naive way,
        i.e. by dividing with number of all pairwise groups.
    _type : str
        Default: 'beta'. Choose from 'beta' or 'alpha'. Determines the type of
        diversity that was considered for testing significant group
        differences.
    draw_edgelabel : boolean
        If true, draw p-values as edge labels.
    edgelabel_decimals : int
        Default: 1
        Number of digits to be printed for p-values.
    ax : plt axis
        If not none, use this axis to plot on.
    edge_color_sig : str
        Default: None
        If not None, define color significant edges should be drawn with.
    edgelabel_decimals : int
        Default: 2
        Number of digits p-values are printed with.
    print_title : bool
        Default: True
        If True, print information about metadata-field, statistical test,
        alpha or beta diversity, permutations, ...

    Returns
    -------
    A matplotlib figure.
    """
    LINEWIDTH_SIG = 2.0
    LINEWIDTH_NONSIG = 0.2
    NODECOLOR = {'alpha': 'lightblue', 'beta': 'lightgreen'}
    EDGE_COLOR_NONSIG = 'gray'

    # initialize empty graph
    G = nx.Graph()
    # add node for every group to the graph
    G.add_nodes_from(list(n_per_group.index))

    numComp = len(list(combinations(n_per_group.keys(), 2)))

    # add edges between all nodes to the graph
    for a in network.keys():
        for b in network[a].keys():
            weight = LINEWIDTH_NONSIG
            color = EDGE_COLOR_NONSIG
            # naive FDR by just dividing p-value by number of groups-pairs
            if network[a][b]['p-value'] < pthresh / numComp:
                weight = LINEWIDTH_SIG
                if edge_color_sig is not None:
                    color = edge_color_sig
            G.add_edge(a, b,
                       pvalue=("%."+str(edgelabel_decimals)+"f") %
                       network[a][b]['p-value'],
                       weight=weight, color=color)

    # ignore warnings of matplotlib due to outdated networkx calls
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                category=matplotlib.cbook.mplDeprecation)
        warnings.filterwarnings("ignore",
                                category=UserWarning,
                                module="matplotlib")

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # use circular graph layout. Spring layout did not really work here.
        pos = nx.circular_layout(G)
        # nodes are randomly assigned to fixed positions, here I re assigned
        # positions by sorted node names to make images determined.
        new_pos = dict()
        l_pos = sorted(list(pos.values()), key=lambda i: i[0] + 1000 * i[1])
        l_nodes = list(sorted(pos.keys()))
        for (key, value) in zip(l_nodes, l_pos):
            new_pos[key] = value
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw(G, with_labels=False, pos=new_pos, width=weights,
                node_color=NODECOLOR[_type],
                edge_color=[G[u][v]['color'] for u, v in G.edges()],
                ax=ax)

        # draw labels for nodes instead of pure names
        for node in G:
            nx.draw_networkx_labels(
                G, new_pos,
                labels={node: "%s\nn=%i" % (node, n_per_group.loc[node])},
                font_color='black', font_weight='bold',
                ax=ax)

        # draw edge labels
        if draw_edgelabel:
            # ensure that edges are addressed in the same way, i.e. (a, b)
            # is not (b, a): tuple(sorted(...))
            edge_labels = \
                dict([(tuple(sorted((a, b,))), data['pvalue'])
                      for a, b, data
                      in G.edges(data=True)
                      if (float(data['pvalue']) < pthresh / numComp) or
                      (len(network.keys()) < 8)])
            nx.draw_networkx_edge_labels(G, new_pos, edge_labels=edge_labels,
                                         ax=ax, label_pos=0.25)
            # , label_pos=0.5, font_size=10, font_color='k',
            # font_family='sans-serif', font_weight='normal', alpha=1.0,
            # bbox=None, ax=None, rotate=True, **kwds)

        # plot title
        if print_title:
            ax.set_title("%s: %s" % (_type, group_name), fontsize=20)
            text = ''
            if _type == 'beta':
                text = 'p-wise %s\n%i perm., %s' % (fct_name, num_permutations,
                                                    metric_name)
            elif _type == 'alpha':
                text = 'p-wise two-sided %s\n%s' % (
                    fct_name.replace('mannwhitneyu', 'Mann-Whitney'),
                    metric_name)
            ax.text(0.5, 0.98, text, transform=ax.transAxes, ha='center',
                    va='top')

        # plot legend
        ax.plot([0], [0], 'gray',
                label=u'p < %0.*f' % (_getfirstsigdigit(pthresh), pthresh),
                linewidth=LINEWIDTH_SIG,
                color=edge_color_sig if edge_color_sig is not None else 'gray')
        ax.plot([0], [0], 'gray',
                label='p >= %0.*f' % (_getfirstsigdigit(pthresh), pthresh),
                linewidth=LINEWIDTH_NONSIG)
        ax.legend(title='FDR corrected')

        # add some space to the axis to not truncate text labels
        factor = 0.15
        ax.set_xlim(
            ax.get_xlim()[0]-(ax.get_xlim()[1]-ax.get_xlim()[0])*factor,
            ax.get_xlim()[1]+(ax.get_xlim()[1]-ax.get_xlim()[0])*factor,
        )
        ax.set_ylim(
            ax.get_ylim()[0]-(ax.get_ylim()[1]-ax.get_ylim()[0])*factor,
            ax.get_ylim()[1]+(ax.get_ylim()[1]-ax.get_ylim()[0])*factor,
        )

    return ax


def plotGroup_histograms(alpha, groupings, min_group_size=21, ax=None):
    """Plots alpha diversity histograms for grouped data.

    Parameters
    ----------
    alpha : pandas.core.series.Series
        The alpha diversities for the samples
    groupings : pandas.core.series.Series
        A group label per sample.
    min_group_size : int
        A minimal group size to be considered. Smaller group labels will be
        ignored. Default: 21.
    ax : plt axis
        The axis to plot on. If none, create a new plt figure and return.

    Returns
    -------
    A plt axis with histograms for each group.
    """
    # remove samples whose grouping in NaN
    groupings = groupings.dropna()

    # remove samples for which we don't have alpha div measures
    groupings = groupings.loc[sorted(set(groupings.index) & set(alpha.index))]

    # remove groups with less than minNum samples per group
    groups = [name
              for name, counts
              in groupings.value_counts().iteritems()
              if counts >= min_group_size]

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for group in groups:
        sns.distplot(alpha.loc[groupings[groupings == group].index],
                     hist=False, label=group, ax=ax, rug=True)

    return ax


def plotGroup_permanovas(beta, groupings,
                         network, n_per_group, min_group_size,
                         num_permutations, metric_name, group_name, fct_name,
                         ax=None, horizontal=False, edgelabel_decimals=2,
                         print_sample_numbers=False, colors_boxplot=None):
    """
    Parameters
    ----------
    horizontal : Bool
        Default: False.
        Plot boxes horizontally. Useful for long group names.
    edgelabel_decimals : int
        Default: 2
        Number of digits p-values are printed with.
    print_sample_numbers : bool
        Default: False.
        If True, adds " (n=xx)" to ticklabels.
    colors_boxplot : dict(metadata_field: RGB)
        Default: None
        Set color schema for boxplots in "sample rel. abundance" plot.
    """
    # remove samples whose grouping in NaN
    groupings = groupings.dropna()

    # remove samples for which we don't have alpha div measures
    groupings = groupings.loc[sorted(set(groupings.index) & set(beta.ids))]

    # remove groups with less than minNum samples per group
    groups = sorted([name
                     for name, counts
                     in groupings.value_counts().iteritems()
                     if counts >= min_group_size])

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if n_per_group.shape[0] < 2:
        ax.text(0.5, 0.5,
                'only %i group:\n"%s"' % (n_per_group.shape[0],
                                          ", ".join(list(n_per_group.index))),
                ha='center', va='center', fontsize=15)
        ax.axis('off')
        return ax, []

    data = []
    name_left = 'left'
    name_right = 'right'
    name_inter = 'between'
    label_left = 'left: '
    label_right = 'right: '
    x_axis, y_axis = 'edge', metric_name
    if horizontal:
        label_left = ''
        label_right = ''
        x_axis, y_axis = metric_name, 'edge'
    for a, b in combinations(groups, 2):
        nw = None
        if a in network:
            if b in network[a]:
                nw = network[a][b]
        if (nw is None) & (b in network):
            if a in network[b]:
                nw = network[b][a]

        # add n=xx information to group labels
        names = dict()
        for name in [a,b]:
            names[name] = name
            if print_sample_numbers:
                names[name] += " (n=%i)" % n_per_group.loc[name]

        edgename = '%s%s\np: %.*f\n%s%s' % (
            label_left,
            names[a],
            max(_getfirstsigdigit(nw['p-value']), edgelabel_decimals),
            nw['p-value'],
            label_right,
            names[b])
        dists = dict()
        # intra group distances
        dists[name_left] = [beta[x, y]
                            for x, y in
                            combinations(groupings[groupings == a].index, 2)]
        dists[name_right] = [beta[x, y]
                             for x, y in
                             combinations(groupings[groupings == b].index, 2)]
        # inter group distances
        dists[name_inter] = [beta[x, y]
                             for x in
                             groupings[groupings == a].index
                             for y in
                             groupings[groupings == b].index]

        for _type in dists.keys():
            grp_name = None
            if _type == 'left':
                grp_name = a
            elif _type == 'right':
                grp_name = b
            else:
                grp_name = 'inter'
            for d in dists[_type]:
                data.append({'edge': edgename, '_type': _type, metric_name: d,
                             'group': grp_name})

    colors = sns.xkcd_palette(["green", "cyan", "lightblue", "dusty purple", "greyish", ])
    if colors_boxplot is not None:
        missing_colors = set([name_left, name_inter, name_right]) - set(colors_boxplot.keys())
        if len(missing_colors) > 0:
            raise ValueError("Not all group conditions have defined colors! %s" % ",".join(missing_colors))
        colors = colors_boxplot
    sns.boxplot(data=pd.DataFrame(data),
                x=x_axis,
                y=y_axis,
                hue='_type',
                hue_order=[name_left, name_inter, name_right],
                ax=ax,
                palette=colors)
    if horizontal:
        ax.legend_.remove()
        ax.yaxis.tick_right()
        ax.yaxis.label.set_visible(False)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1))
        ax.xaxis.label.set_visible(False)

    return ax, data


# definition of network plots
def plotNetworks(field: str, metadata: pd.DataFrame, alpha: pd.DataFrame, beta: dict, b_min_num=5, pthresh=0.05,
                 permutations=999, name=None, minnumalpha=21,
                 fct_beta_test=permanova, fct_alpha_test=mannwhitneyu, summarize=False):
    """Plot a series of alpha- / beta- diversity sig. difference networks.

    Parameters
    ----------
    field : str
        Name of the metdata columns, which shall split samples into groups.
    metadata : pd.DataFrame
        Metadata for samples.
    alpha : pd.DataFrame
        One column per diversity metric.
    beta : dict(str: skbio.DistanceMatrix)
        One key, value pair per diversity metric.
    b_min_num : int
        Default: 5.
        Minimal number of samples per group to be included in beta diversity
        analysis. Lower numbers would have to less power for statistical tests.
    pthresh : float
        Default: 0.05
        Significance niveau.
    permutations : int
        Default: 999.
        Number permutations for PERMANOVA tests.
    name : str
        Default: None
        A title for the returned plot.
    minnumalpha : int
        Default: 21.
        Minimal number of samples per group to be included in alpha diversity
        analysis. Lower numbers would have to less power for statistical tests.
    fct_beta_test : function
        Default: skbio.stats.distance.permanova
        Python function to execute test.
        Valid functions are "permanova" or "anosim" from skbio.stats.distance.
    fct_alpha_test : function
        Default: mannwhitneyu
        Python function to execute test.

    Returns
    -------
    plt.Figure or pd.DataFrame
    """
    def _get_ax(axes, row, col):
        if len(axes.shape) == 2:
            return axes[row][col]
        elif len(axes.shape) == 1:
            return axes[col]
        else:
            raise ValueError("Just one? plot. Should be two!")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        num_rows = 0
        if beta is not None:
            num_rows += len(beta.keys())
        if alpha is not None:
            num_rows += alpha.shape[1]
        if summarize is False:
            f, axarr = plt.subplots(num_rows, 2, figsize=(10, num_rows*5))

        row = 0
        summary = []
        networks = {'alpha': dict(), 'beta': dict()}
        if alpha is not None:
            for a_metric in alpha.columns:
                a = detect_distant_groups_alpha(
                    alpha[a_metric], metadata[field],
                    min_group_size=minnumalpha, fct_test=fct_alpha_test)
                if summarize:
                    networks['alpha'][a_metric] = a
                    for i in a['network'].keys():
                        for j in a['network'][i]:
                            x, y = tuple(sorted([i, j]))
                            summary.append({
                                'group x': x,
                                'group y': y,
                                'p-value': a['network'][i][j]['p-value'],
                                'test-statistic': a['network'][i][j]['test-statistic'],
                                'num samples x': a['n_per_group'].loc[x],
                                'num samples y': a['n_per_group'].loc[y],
                                'metric': a['metric_name'],
                                'group name': a['group_name'],
                                'num_permutations': a['num_permutations'],
                                'test name': a['fct_name'],
                                'min_group_size': a['min_group_size'],
                                'diversity': 'alpha'})
                else:
                    plotDistant_groups(
                        **a, pthresh=pthresh, _type='alpha', draw_edgelabel=True,
                        ax=_get_ax(axarr, row, 0))
                    plotGroup_histograms(
                        alpha[a_metric], metadata[field],
                        ax=_get_ax(axarr, row, 1),
                        min_group_size=minnumalpha)
                # axarr[row][1].set_xlim((0, 20))
                row += 1

        if beta is not None:
            for b_metric in beta.keys():
                b = detect_distant_groups(
                    beta[b_metric], b_metric, metadata[field],
                    min_group_size=b_min_num, num_permutations=permutations,
                    fct_test=fct_beta_test)
                if summarize:
                    networks['beta'][b_metric] = b
                    for i in b['network'].keys():
                        for j in b['network'][i]:
                            x, y = tuple(sorted([i, j]))
                            summary.append({
                                'group x': x,
                                'group y': y,
                                'p-value': b['network'][i][j]['p-value'],
                                'avgdist': b['network'][i][j]['avgdist'],
                                'test-statistic': b['network'][i][j]['test-statistic'],
                                'num samples x': b['n_per_group'].loc[x],
                                'num samples y': b['n_per_group'].loc[y],
                                'metric': b['metric_name'],
                                'group name': b['group_name'],
                                'num_permutations': b['num_permutations'],
                                'test name': b['fct_name'],
                                'min_group_size': b['min_group_size'],
                                'diversity': 'beta'})
                else:
                    plotDistant_groups(
                        **b, pthresh=pthresh, _type='beta', draw_edgelabel=True,
                        ax=_get_ax(axarr, row, 0))
                    plotGroup_permanovas(
                        beta[b_metric], metadata[field], **b,
                        ax=_get_ax(axarr, row, 1),
                        horizontal=True)
                row += 1

        if (summarize is False):
            if (name is not None):
                plt.suptitle(name)
            return f
        else:
            res = pd.DataFrame(summary)
            ordered_cols = [
                'group name', 'diversity', 'metric', 'group x', 'group y',
                'p-value', 'test-statistic', 'num samples x', 'num samples y',
                'avgdist', 'test name', 'min_group_size', 'num_permutations']
            if res.shape[0] > 0:
                res = res[ordered_cols].sort_values(by=['diversity', 'metric'])
            else:
                res = pd.DataFrame(["too few samples"])
            return res, networks


def mutate_sequence(sequence, num_mutations=1,
                    alphabet=['A', 'C', 'G', 'T']):
    """Introduce a number of point mutations to a DNA sequence.

    No position will be mutated more than once.

    Parameters
    ----------
    sequence : str
        The sequence that is going to get mutated.
    num_mutations : int
        Number of mutations that should be made in the sequence.
        Default is 1.
    alphabet : [chars]
        Alphabet of replacement characters for mutations.
        Default is [A,C,G,T], i.e. DNA alphabet. Change to [A,C,G,U] for RNA.

    Returns
    -------
    str : the mutated sequence.

    Raises
    ------
    ValueError:
        a) if number of mutations exceeds available characters in sequence.
        b) if alphabet is so limited that position to be mutated will be the
           same as before.
    """
    if len(sequence) < num_mutations:
        raise ValueError("Sequence not long enough for that many mutations.")
    positions = set(range(0, len(sequence)))
    mutated_positions = set()
    mut_sequence = sequence
    while len(mutated_positions) < num_mutations:
        pos = random.choice(list(positions))
        positions.remove(pos)
        mutated_positions.add(pos)
        cur = mut_sequence[pos].upper()
        replacement_candidates = [c for c in alphabet if c.upper() != cur]
        try:
            mut = random.choice(replacement_candidates)
        except IndexError:
            raise ValueError("Alphabet is too small to find mutation!")
        mut_sequence = mut_sequence[:pos] + mut + mut_sequence[pos+1:]
    return mut_sequence


def cache(func):
    """Decorator: Cache results of a function call to disk.

    Parameters
    ----------
    func : executabale
        A function plus parameters whichs results shall be cached, e.g.
        "fct_example(1,5,3)", where
        @cache
        def fct_test(a, b, c):
            return a + b * c
    cache_filename : str
        Default: None. I.e. caching is deactivated.
        Pathname to cache file, which will hold results of the function call.
        If file exists, results are loaded from it instead of recomputing via
        provided function. Otherwise, function will be executed and results
        stored to this file.
    cache_verbose : bool
        Default: True.
        Report caching status to 'cache_err', which by default is sys.stderr.
    cache_err : StringIO
        Default: sys.stderr.
        Stream onto which status messages shall be printed.
    cache_force_renew : bool
        Default: False.
        Force re-execution of provided function even if cache file exists.

    Returns
    -------
    Results of provided function, either by actually executing the function
    with provided parameters or by loaded results from filename.

    Notes
    -----
    It is the obligation of the user to ensure that arguments for the provided
    function don't change between creation of cache file and loading from cache
    file!
    """
    func_name = func.__name__

    def execute(*args, **kwargs):
        cache_args = {'cache_filename': None,
                      'cache_verbose': True,
                      'cache_err': sys.stderr,
                      'cache_force_renew': False}
        for varname in cache_args.keys():
            if varname in kwargs:
                cache_args[varname] = kwargs[varname]
                del kwargs[varname]

        if cache_args['cache_filename'] is None:
            if cache_args['cache_verbose']:
                cache_args['cache_err'].write(
                    '%s: no caching, since "cache_filename" is None.\n' %
                    func_name)
            return func(*args, **kwargs)

        if os.path.exists(cache_args['cache_filename']) and\
           (os.stat(cache_args['cache_filename']).st_size <= 0):
            if cache_args['cache_verbose']:
                cache_args['cache_err'].write(
                    '%s: removed empty cache.\n' %
                    func_name)
            os.remove(cache_args['cache_filename'])

        if (not os.path.exists(cache_args['cache_filename'])) or\
           cache_args['cache_force_renew']:
            try:
                f = open(cache_args['cache_filename'], 'wb')
                results = func(*args, **kwargs)
                pickle.dump(results, f)
                f.close()
                if cache_args['cache_verbose']:
                    cache_args['cache_err'].write(
                        '%s: stored results in cache "%s".\n' %
                        (func_name, cache_args['cache_filename']))
            except Exception as e:
                raise e
        else:
            f = open(cache_args['cache_filename'], 'rb')
            results = pickle.load(f)
            f.close()
            if cache_args['cache_verbose']:
                cache_args['cache_err'].write(
                    '%s: retrieved results from cache "%s".\n' %
                    (func_name, cache_args['cache_filename']))
        return results
    if func.__doc__ is not None:
        execute.__doc__ = func.__doc__
    else:
        execute.__doc__ = ""
    execute.__doc__ += "\n\n" + cache.__doc__
    # restore wrapped function name
    execute.__name__ = func_name
    return execute


def _map_metadata_calout(metadata, calour_experiment, field):
    """Calour reads a metadata TSV not as dtype=str,
       therefore certain values change :-("""
    _mapped_meta = pd.concat([metadata[field],
                              calour_experiment.sample_metadata[field]],
                             axis=1).dropna()
    _mapped_meta.columns = ['meta', 'calour']
    _map = dict()
    for value_metadata in _mapped_meta['meta'].unique():
        values_calour = _mapped_meta[
            _mapped_meta['meta'] == value_metadata]['calour'].unique()
        if len(values_calour) > 1:
            raise ValueError('More than one value map!!')
        _map[value_metadata] = values_calour[0]
    return _map


def _find_diff_taxa_runpfdr(calour_experiment, metadata, field, diffTaxa=None,
                            out=sys.stdout, method='meandiff', random_seed=None):
    """Finds differentially abundant taxa in a calour experiment for the given
       metadata field.

    Parameters
    ----------
    calour_experiment : calour.experiment
        The calour experiment, holding the OTU table and all metadata
        information.
    metadata : pd.DataFrame
        metadata for samples. Cannot use calour sample_metadata due to internal
        datatype casts, e.g. int might become floats.
    field : str
        The metadata column along which samples should be separated and tested
        for differentially abundant taxa.
    diffTaxa : dict(dict())
        A prefilled return object, for cases where we want to combine evidence.
    out : StringIO
        The strem onto which messages should be written. Default is sys.stdout.
    method : str or function
        Default: 'meandiff'
        the method to use for the t-statistic test. options:
        'meandiff' : mean(A)-mean(B) (binary)
        'mannwhitney' : mann-whitneu u-test (binary)
        'stdmeandiff' : (mean(A)-mean(B))/(std(A)+std(B)) (binary)
        function : use this function to calculate the t-statistic
        (input is data,labels, output is array of float)

    Returns
    -------
        A dict of dict. First level key are the pairs of field values between
        which has been tested, second level is the taxon, value is number of
        times this taxon found to be differentially abundant.
    """

    if diffTaxa is None:
        diffTaxa = dict()

    metadata = metadata.loc[set(calour_experiment.sample_metadata.index) & set(metadata.index), :]

    ns = metadata[field].value_counts()
    e = calour_experiment.filter_ids(metadata.index, axis='s')
    _map_values = _map_metadata_calout(metadata, e, field)
    for (a, b) in combinations(ns.index, 2):
        ediff = e.diff_abundance(field,
                                 _map_values[a],
                                 _map_values[b],
                                 fdr_method='dsfdr', method=method, random_seed=random_seed)
        out.write("  % 4i (of % 4i) taxa different between '%s' (n=%i) vs. '%s' (n=%i)\n"
                  % (ediff.feature_metadata.shape[0], e.feature_metadata.shape[0], a, ns[a], b, ns[b]))
        if ediff.feature_metadata.shape[0] > 0:
            if (a, b) not in diffTaxa:
                diffTaxa[(a, b)] = dict()
            for taxon in ediff.feature_metadata.index:
                if taxon not in diffTaxa[(a, b)]:
                    diffTaxa[(a, b)][taxon] = 0
                diffTaxa[(a, b)][taxon] += 1

    return diffTaxa


def _find_diff_taxa_singlelevel(calour_experiment, metadata,
                                groups, diffTaxa=None,
                                out=sys.stdout,
                                method='meandiff', random_seed=None):
    """Finds differentially abundant taxa in a calour experiment for the given
       metadata group of fields, i.e. samples are controlled for the first :-1
       fields and abundance is checked for the latest field.

    Parameters
    ----------
    calour_experiment : calour.experiment
        The calour experiment, holding the OTU table and all metadata
        information.
    metadata : pd.DataFrame
        metadata for samples. Cannot use calour sample_metadata due to internal
        datatype casts, e.g. int might become floats.
    groups : [str]
        The metadata columns for which samples should be controlled (first n-1)
        and along which samples should be separated and tested for
        differentially abundant taxa (last)
    diffTaxa : dict(dict())
        A prefilled return object, for cases where we want to combine evidence.
    out : StringIO
        The strem onto which messages should be written. Default is sys.stdout.
    method : str or function
        Default: 'meandiff'
        the method to use for the t-statistic test. options:
        'meandiff' : mean(A)-mean(B) (binary)
        'mannwhitney' : mann-whitneu u-test (binary)
        'stdmeandiff' : (mean(A)-mean(B))/(std(A)+std(B)) (binary)
        function : use this function to calculate the t-statistic
        (input is data,labels, output is array of float)

    Returns
    -------
        A dict of dict. First level key are the pairs of field values between
        which has been tested, second level is the taxon, value is number of
        times this taxon found to be differentially abundant.
    """
    if diffTaxa is None:
        diffTaxa = dict()

    metadata = metadata.loc[set(calour_experiment.sample_metadata.index) & set(metadata.index), :]

    if len(groups) > 1:
        e = calour_experiment.filter_ids(
            metadata.index, inplace=False, axis='s')
        for n, g in metadata.groupby(groups[:-1]):
            name = n
            if type(n) != tuple:
                name = [n]
            out.write(", ".join(
                map(lambda x: "%s: %s" % x, zip(groups, name))) + ", ")
            out.write("'%s'" % groups[-1])
            out.write("  (n=%i)\n" % g.shape[0])

            # filter samples for calour
            e_filtered = e
            for (field, value) in zip(groups, name):
                _map_values = _map_metadata_calout(metadata, e, field)
                if value in _map_values:
                    e_filtered = e_filtered.filter_samples(
                        field, [_map_values[value]], inplace=False)
            diffTaxa = _find_diff_taxa_runpfdr(e_filtered,
                                               metadata,
                                               groups[-1],
                                               diffTaxa, method=method,
                                               random_seed=random_seed)
    else:
        out.write("'%s'" % groups[0])
        out.write("  (n=%i)\n" % metadata.shape[0])
        diffTaxa = _find_diff_taxa_runpfdr(
            calour_experiment, metadata, groups[0], diffTaxa,
            random_seed=random_seed)

    return diffTaxa


def find_diff_taxa(calour_experiment, metadata, groups, diffTaxa=None,
                   out=sys.stdout, method='meandiff', random_seed=None):
    # TODO: rephrase docstring
    # TODO: unit tests for all three functions
    # TODO: include calour in requirements
    # TODO: fully drag calour into function and pass counts and metadata instea
    """Finds differentially abundant taxa in a calour experiment for the given
       metadata group of fields, i.e. samples are controlled for the first :-1
       fields and abundance is checked for the latest field.

    Parameters
    ----------
    calour_experiment : calour.experiment
        The calour experiment, holding the OTU table and all metadata
        information.
    metadata : pd.DataFrame
        metadata for samples. Cannot use calour sample_metadata due to internal
        datatype casts, e.g. int might become floats.
    groups : [str]
        The metadata columns for which samples should be controlled (first n-1)
        and along which samples should be separated and tested for
        differentially abundant taxa (last)
    diffTaxa : dict(dict())
        A prefilled return object, for cases where we want to combine evidence.
    out : StringIO
        The strem onto which messages should be written. Default is sys.stdout.
    method : str or function
        Default: 'meandiff'
        the method to use for the t-statistic test. options:
        'meandiff' : mean(A)-mean(B) (binary)
        'mannwhitney' : mann-whitneu u-test (binary)
        'stdmeandiff' : (mean(A)-mean(B))/(std(A)+std(B)) (binary)
        function : use this function to calculate the t-statistic
        (input is data,labels, output is array of float)
    random_seed : set random seed

    Returns
    -------
        A dict of dict. First level key are the pairs of field values between
        which has been tested, second level is the taxon, value is number of
        times this taxon found to be differentially abundant.
    """
    if diffTaxa is None:
        diffTaxa = dict()

    for i in range(len(groups)):
        sub_groups = groups[len(groups)-i-1:]
        diffTaxa = _find_diff_taxa_singlelevel(
            calour_experiment, metadata, sub_groups, diffTaxa, method=method,
            random_seed=random_seed)
        out.write("\n")

    merged_diffTaxa = dict()
    for (a, b) in diffTaxa.keys():
        key = tuple(sorted((a, b)))
        if key not in merged_diffTaxa:
            merged_diffTaxa[key] = dict()
        for feature in diffTaxa[(a, b)].keys():
            if feature not in merged_diffTaxa[key]:
                merged_diffTaxa[key][feature] = 0
            merged_diffTaxa[key][feature] += diffTaxa[(a, b)][feature]

    return merged_diffTaxa


def plot_diff_taxa(counts, metadata_field, diffTaxa, onlyusetaxa=None,
                   taxonomy=None,
                   min_mean_abundance=0.01, max_x_relabundance=None,
                   num_ranks=2, title=None, scale_height=0.7,
                   feature_color_map=None, topXfeatures=None, colors_boxplot=None, color_barplot=None):
    """Plots relative abundance and fold change for taxa.

    Parameters
    ----------
    counts : Pandas.DataFrame
        OTU table with rows for features and columns for samples.
    metadata_field : Pandas.Series
        Group labels for every samples between which differentially abundant
        taxa have been found. I.e. one label per sample.
    diffTaxa : dict of dicts
        First level: keys = pairs of group labels
        Second level: keys = feature, values = some numbers (are not considered
        right now)
    onlyusetaxa : [str]
        Default: None.
        Restrict list of differentially abundant taxa to those provided in the
        list.
    taxonomy : Pandas.Series
        Default: none
        Taxonomy labels for features.
    min_mean_abundance : float
        Default: 0.01.
        Minimal relative mean abundance a feature must have in both groups to
        be plotted.
    max_x_relabundance : float
        Default: None, i.e. max value from data is taken.
        For left plot: maximal x-axis limit, to zoom in if all abundances are
        low.
    num_ranks : int
        Default: 2, i.e. Genus and Species
        How many last ranks shall be displayed on y-axis of right plot.
    title : str
        Default: None
        Something to print as the suptitle
    scale_height : float
        Default: 0.7
        Scaling factor for height of figure.
    feature_color_map : pd.Series
        Colores for tick label plotting of features.
        Black if no value is mentioned.
    topXfeatures : int
        Default: None
        If set to a positive number, only the top X features will be plotted.
    colors_boxplot : dict(metadata_field: RGB)
        Default: None
        Set color schema for boxplots in "sample rel. abundance" plot.
    color_barplot : RGB
        Default: None
        Defines color of barplots for "mean abundance shift" and "mean fold change".

    Returns
    -------
    Matplotlib Figure.
    """
    fig, ax = plt.subplots(len(diffTaxa), 3,
                           figsize=(3*5, 5*len(diffTaxa)))
    meanrealabund = pd.DataFrame()
    counts.index.name = 'feature'
    counts.columns.name = 'sample_name'
    metadata_field.index.name = 'sample_name'
    relabund = counts / counts.sum()
    relabund.index.name = counts.index.name
    relabund.columns.name = counts.columns.name

    comparisons = sorted(map(sorted, diffTaxa.keys()))
    num_drawn_taxa = []
    if colors_boxplot is not None:
        missing_colors = {s for k in diffTaxa.keys() for s in k} - set(colors_boxplot.keys())
        if len(missing_colors) > 0:
            raise ValueError("Not all group conditions have defined colors! %s" % ",".join(missing_colors))
        grp_colors = colors_boxplot
    else:
        grp_colors = {k: sns.color_palette()[i] for i, k in enumerate({s for k in diffTaxa.keys() for s in k})}
    for i, (meta_value_a, meta_value_b) in enumerate(comparisons):
        # only consider taxa given in the diffTaxa object...
        taxa = list(diffTaxa[(meta_value_a, meta_value_b)])
        # ... and further subset if user provides an additional list of taxa
        if onlyusetaxa is not None:
            taxa = [t for t in taxa if t in onlyusetaxa]

        samples_a = metadata_field[metadata_field == meta_value_a].index
        samples_b = metadata_field[metadata_field == meta_value_b].index
        foldchange = np.log(
            (counts.reindex(index=taxa, columns=samples_a)+1).mean(axis=1) /
            (counts.reindex(index=taxa, columns=samples_b)+1).mean(axis=1))

        # only consider current list of taxa, but now also filter out those
        # with too low relative abundance.
        #return relabund, samples_a, taxa
        meanrealabund = pd.concat([
            relabund.reindex(index=taxa, columns=samples_a).mean(axis=1),
            relabund.reindex(index=taxa, columns=samples_b).mean(axis=1)], axis=1, sort=False).rename(columns={0: meta_value_a, 1: meta_value_b})
        meanrealabund = meanrealabund[(meanrealabund >= min_mean_abundance).any(axis=1)]
        taxa = meanrealabund.max(axis=1).sort_values(ascending=False).index
        if topXfeatures is not None:
            taxa = taxa[:topXfeatures]
        # taxa = sorted(list(
        #     set([idx
        #          for idx, meanabund
        #          in relabund.reindex(
        #              index=taxa, columns=samples_a).mean(axis=1).iteritems()
        #          if meanabund >= min_mean_abundance]) |
        #     set([idx
        #          for idx, meanabund
        #          in relabund.reindex(
        #              index=taxa, columns=samples_b).mean(axis=1).iteritems()
        #          if meanabund >= min_mean_abundance])))
        if len(taxa) <= 0:
            print("Warnings: no taxa left!")
        num_drawn_taxa.append(len(taxa))
        relabund_field = []
        for (samples, grpname) in [(samples_a, meta_value_a),
                                   (samples_b, meta_value_b)]:
            r = relabund.reindex(
                index=taxa, columns=samples).stack().reset_index().rename(
                columns={0: 'relative abundance'})
            r['group'] = grpname
            relabund_field.append(r)
        relabund_field = pd.concat(relabund_field)

        curr_ax = ax[0]
        if len(diffTaxa) > 1:
            curr_ax = ax[i][0]
        if len(taxa) > 0:
            grpsby = 'level_1'
            if counts.columns.name is not None:
                grpsby = counts.columns.name
            if relabund_field.groupby('sample_name')['relative abundance'].sum().max() > 1:
                raise ValueError("If we add up all relative abundances, we have more than 100%. This is impossible! Please check if you provided the full count table, or errounously subsetted it to e.g. specific features. If this is the case, correct and use parameter 'onlyusetaxa'!")
            g = sns.boxplot(data=relabund_field,
                            x='relative abundance',
                            y='feature',
                            order=taxa,
                            hue='group',
                            palette=grp_colors,
                            ax=curr_ax,
                            orient='h')
            if max_x_relabundance is None:
                if relabund_field.max() is not None:
                    max_x_relabundance = min(
                        relabund_field['relative abundance'].max() * 1.1, 1.0)
                else:
                    max_x_relabundance = 1.0
            g.set_xlim((0, max_x_relabundance))
            # curr_ax.legend(loc="upper right")
            curr_ax.legend(bbox_to_anchor=(-0.1, 1.15))
            g.set_title("sample rel. abundance")

        # define colors for taxons
        if (feature_color_map is not None) and \
           (feature_color_map.shape[0] > 0):
            availColors = \
                sns.color_palette('Paired', 12) +\
                sns.color_palette('Dark2', 12) +\
                sns.color_palette('Pastel1', 12)
            colors = dict()
            for i, state in enumerate(feature_color_map.unique()):
                if state not in colors:
                    colors[state] = availColors[len(colors) % len(availColors)]
            # color the labels of the Y-axis according to different categories
            # given by feature_color_map
            for tick in curr_ax.get_yticklabels():
                if tick.get_text() in feature_color_map.index:
                    tick.set_color(colors[feature_color_map[tick.get_text()]])

        # MOVED ABUNDANCE
        if len(diffTaxa) > 1:
            curr_ax = ax[i][1]
        else:
            curr_ax = ax[1]
        if len(taxa) > 0:
            #return (meanrealabund[meta_value_a]- meanrealabund[meta_value_b]).to_frame().reset_index()
            relabundshift = (meanrealabund[meta_value_a]- meanrealabund[meta_value_b]).loc[taxa].to_frame().reset_index()
            g = sns.barplot(data=relabundshift,
                            orient='h',
                            y='feature',
                            x=0,
                            ax=curr_ax,
                            color=color_barplot if color_barplot is not None else sns.xkcd_rgb["denim blue"])
            g.yaxis.set_ticklabels([])
            g.set_ylabel('')
            g.set_xlabel('<-- more in %s     |      more in %s -->' %
                         (meta_value_b, meta_value_a))
            g.set_title("mean abundance shift")
            g.set_xlim(-1.05*relabundshift[0].abs().max(),
                       +1.05*relabundshift[0].abs().max())

        # FOLD CHANGE
        if len(diffTaxa) > 1:
            curr_ax = ax[i][2]
        else:
            curr_ax = ax[2]
        if len(taxa) > 0:
            g = sns.barplot(data=foldchange.loc[taxa].to_frame().reset_index(),
                            orient='h',
                            y='feature',
                            x=0,
                            ax=curr_ax,
                            color=color_barplot if color_barplot is not None else sns.xkcd_rgb["denim blue"])
            g.set_ylabel('')
            g.set_title("mean fold change")

            if taxonomy is not None:
                g.yaxis.tick_right()
                g.set(yticklabels=taxonomy.reindex(taxa).fillna('k__').apply(
                    lambda x: " ".join(list(
                        map(str.strip, x.split(';')))[-num_ranks:])))
                # color the labels of the Y-axis according to different
                # categories given by feature_color_map
                if feature_color_map is not None:
                    tickpairs = zip(
                        ax[0].get_yticklabels(),
                        g.yaxis.get_ticklabels())
                    for tick_feature, tick_taxonomy in tickpairs:
                        if tick_feature.get_text() in feature_color_map.index:
                            tick_taxonomy.set_color(
                                colors[
                                    feature_color_map[
                                        tick_feature.get_text()]])
                    # adding a legend to the plot, explaining the font colors
                    g.legend(
                        [Line2D([0], [0], color=colors[category], lw=8)
                         for category
                         in feature_color_map.unique()],
                        [category for category in feature_color_map.unique()])
            else:
                g.yaxis.set_ticklabels([])

            g.set_xlabel('<-- more in %s     |      more in %s -->' %
                         (meta_value_b, meta_value_a))
            g.set_xlim(-1.05*foldchange.loc[taxa].abs().max(),
                       +1.05*foldchange.loc[taxa].abs().max())
        titletext = "%s\nminimal relative abundance: %f" % (
            metadata_field.name, min_mean_abundance)
        if title is not None:
            titletext = title + "\n" + titletext
        fig.suptitle(titletext)

    if sum(num_drawn_taxa) > 0:
        fig.set_size_inches(
            fig.get_size_inches()[0], scale_height*max(num_drawn_taxa)*len(comparisons))

    return fig, meanrealabund, relabund


@cache
def identify_important_features(metadata_group, counts, num_trees=1000,
                                stdout=sys.stdout, test_size=0.25,
                                num_repeats=5, max_features=100, n_jobs=1,
                                min_features=1):
    """Use Random Forests to determine most X important features to predict
       sample labels.

    Parameters
    ----------
    metadata_group : pd.Series
        Labels for samples which shall be predicted of the feature counts.
    counts : pd.DataFrame
        Feature counts. Rows = features, Columns = samples.
    num_trees : int
        Default: 1000.
        Number of decision trees used for random forests.
        Larger number = more precise, but also slower.
    stdout : StringIO
        Default: sys.stdout
        Stream onto which messages to the user are printed.
    test_size : float
        Default: 0.25
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.
    num_repeats : int
        Default: 5.
        Number of repeats of the random forest runs.
    n_jobs : int
        Default: 1.
        Number of CPU cores to use for computation.
    max_features : int
        Default: 100.
        Stop after exploring accuracy for max_features number of features.
    min_features : int
        Default: 1.
        Enforce testing of at least this number of features.
    Returns
    -------
    ?
    """
    idx_samples = sorted(list(set(metadata_group.index) & set(counts.columns)))
    merged_meta = metadata_group.loc[idx_samples]
    # note that matrix is now transposed to comply with sklearn!!
    merged_counts = counts.loc[:, idx_samples].T

    stdout.write("Predicting class labels from counts for:\n")
    stdout.write(str(merged_meta.value_counts()) + "\n")

    # First pass to determine feature importance list
    X_train, X_test, y_train, y_test = train_test_split(
        merged_counts, merged_meta, test_size=test_size, random_state=42)
    best_RF = None
    for i in range(num_repeats):
        clf = RandomForestClassifier(n_estimators=num_trees, n_jobs=n_jobs)
        clf = clf.fit(X_train, y_train)
        clf._has_score = clf.score(X_test, y_test)
        if (best_RF is None) or (best_RF._has_score < clf._has_score):
            best_RF = clf
        print("repeat %i, score %.4f" % (i+1, clf._has_score))
    feature_importance = pd.Series(
        best_RF.feature_importances_,
        index=X_train.columns).sort_values(ascending=False)

    # Second pass to check how many features are necessary for sufficient
    # prediction accuracy
    res = []
    for num_features in range(1, feature_importance.shape[0]):
        if num_features > max_features:
            break

        stdout.write('% 3i features ' % num_features)
        X_train, X_test, y_train, y_test = train_test_split(
            merged_counts.loc[:, feature_importance.iloc[:num_features].index],
            merged_meta, test_size=test_size, random_state=42)
        best_RF = None
        for i in range(num_repeats):
            stdout.write('.')
            clf = RandomForestClassifier(n_estimators=num_trees, n_jobs=n_jobs)
            clf = clf.fit(X_train, y_train)
            clf._has_score = clf.score(X_test, y_test)
            if (best_RF is None) or (best_RF._has_score < clf._has_score):
                best_RF = clf
        stdout.write(' %.4f\n' % best_RF._has_score)
        res.append({'number features': num_features,
                    'R^2 score': best_RF._has_score,
                    'sum of feature importance':
                    feature_importance.iloc[:num_features].sum(),
                    'features': feature_importance.iloc[:num_features].index})
        if (best_RF._has_score >= 1) and (num_features >= min_features):
            break
    res = pd.DataFrame(res)

    return res


def plot_identify_important_features(res):
    # create plot
    fig, axes = plt.subplots(1,1)
    p = plt.scatter(res['number features'], res['sum of feature importance'],
                    s=4, color="blue", label="sum of feature importance")
    p = plt.scatter(res['number features'], res['R^2 score'],
                    s=4, color="green", label="R^2 score")

    p = plt.xlabel("number features")
    p = plt.ylabel("sum of feature importance")

    p = plt.legend(loc=4)

    return fig


# copy and paste from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          verbose=sys.stdout,
                          ax=None,
                          xtickrotation=45):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data
    classes = set(y_true.unique()) | set(y_pred.unique())

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, sorted(classes))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if verbose is not None:
            verbose.write("Normalized confusion matrix\n")
    else:
        if verbose is not None:
            verbose.write('Confusion matrix, without normalization\n')

    if verbose is not None:
        verbose.write(str(cm))

    if ax is None:
        fig, ax = plt.subplots()
        #fig.tight_layout()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=sorted(classes), yticklabels=sorted(classes),#classes,
           title=title,
           ylabel='True %s' % ('label' if y_true.name is None else y_true.name),
           xlabel='Predicted %s' % ('label' if y_true.name is None else y_true.name))

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=xtickrotation, ha="center",
#             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    # deal with an edge case extreme black and white result, thus look at amplitude not absolute values
    thresh = (cm - cm.min()).max() / 2.
    # thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j]-cm.min() > thresh else "black")

    ax.grid(b=False, which='both', axis='both')
    ax.set_ylim((-0.5, cm.shape[1]-0.5))
    ax.set_xlim((-0.5, cm.shape[0]-0.5))

    for loc in ax.spines.keys():
        ax.spines[loc].set_color('0') # ['bottom'].set_color('0')

    return ax


def ganttChart(metadata: pd.DataFrame,
               col_birth: str,
               col_entities: str,
               col_events: str,
               col_death: str = None,
               col_events_title: str = None,
               col_entity_groups: str = None,
               col_entity_colors: str = None,
               col_phases_start: str = None,
               col_phases_end: str = None,
               height_ratio: float = 0.3,
               event_line_width: int = 1,
               colors_events: dict = None,
               colors_entities: dict = None,
               colors_phases: dict = None,
               align_to_event_title: str = None,
               counts: pd.DataFrame = None,
               order_entities: list = None,
               timeresolution: str = 'days',
               ):
    """Generates Gantt chart of chronologic experiment design.

    Parameters
    ----------
    metadata : pd.DataFrame
        Full metadata, one row per sample.
    col_birth : str
        Column name, holding birth date of entities / individuals.
    col_entities : str
        Column name, holding entity names. We will plot one bar per entity.
        Entities may have several events.
    col_events : str
        Column name, holding event dates.
    col_death : str
        Default: None.
        Column name, holding death date of entities.
    col_events_title : str
        Default: None.
        Column name, holding titles for events.
    col_entity_groups : str
        Default: None.
        Column name, holding grouping information for entities,
        e.g. cage number.
    col_entity_colors : str
        Default: None.
        Column name, holding coloring information for entities,
        e.g. sick / healthy.
    col_phases_start : str or [str]
        Default: None.
        Column name(s), holding start date for phase,
        e.g. "exposure to infections"
    col_phases_end : str or [str]
        Default: None.
        Column name(s), holding end date for phase,
        e.g. "antibiotics_treatment_end_timestamp"
    height_ratio : float
        Default: 0.3
        Height for figure per entity.
    event_line_width : int
	Default: 1.
	Line width of vertical lines indicating date of event.
    colors_events : dict(str: str)
        Default: None
        Provide a predefined color dictionary to use same colors for several
        plots. Default is an empty dictionary.
        Format: key = event title,
        Value: a triple of RGB float values.
    colors_entities : dict(str: str)
        Default: None
        Colors for entity bars.
    colors_phases : dict(str: str)
        Default: None
        Colors for entity phases.
    align_to_event_title : str
        Default: None
        Align all dates according to a baseline event, instead of using real
        chronologic distances.
    counts : pd.DataFrame
        Default: None
        Samples might be missue due to rarefaction or other QC procedures.
        If provided, events of missing samples will be drawn dotted,
        instead of with a solid line.
    order_entities : [str]
	    List of entity names to order their plotting vertically.
    timeresolution : str {days, seconds}
        Default: days
        Time resolution for internal coordinate system.
        For short time periods, seconds might be more appropriate to visualize
        time intervals.
    Returns
    -------
    """
    if timeresolution not in ['days', 'seconds', 'weeks']:
        raise ValueError('timeresolution must either be "weeks", "days" or "seconds"!')

    COL_DEATH = '_death'
    COL_GROUP = '_group'
    COL_YPOS = '_ypos'
    COL_ENTITY_COLOR = '_entity_color'
    AVAILCOLORS = \
        sns.color_palette('Paired', 12) +\
        sns.color_palette('Dark2', 12) +\
        sns.color_palette('Pastel1', 12)

    if counts is not None:
        if len(set(counts.columns) & set(metadata.index)) <= 0:
            print((
                'Warning: there is no overlap between sample_names in'
                ' metadata and counts!'), file=sys.stderr)

    def _listify(variable):
        if variable is None:
            return [None]
        if not isinstance(variable, list):
            return [variable]
        return variable
    # convert multi colname arguments into lists, if not already list
    col_phases_start = _listify(col_phases_start)
    col_phases_end = _listify(col_phases_end)

    for col in [COL_DEATH, COL_GROUP]:
        assert(col not in metadata.columns)
    cols_dates = list(set([
        col
        for col
        in [col_birth, col_events, col_death] +
           [col
            for col
            in (col_phases_start + col_phases_end)
            if col is not None]
        if col in metadata.columns]))

    meta = metadata.copy()
    if col_entities is not None:
        meta = meta.dropna(subset=[col_birth])
    for col in cols_dates:
        if col is not None:
            meta[col] = pd.to_datetime(metadata[col])
    # convert dates into internal coordinate system
    date_baseline = meta[cols_dates].stack().min()
    #return date_baseline, cols_dates, meta
    for col in cols_dates:
        date_resolution = timeresolution
        factor = 1
        if (timeresolution == 'weeks'):
            date_resolution = 'days'
            factor = 7
        meta[col] = meta[col].apply(lambda x: getattr((x - date_baseline), date_resolution) / factor)
    # try to find end date for entities
    if col_death is not None:
        meta[COL_DEATH] = meta[col_death]
    # for those entities that don't have a date of death information,
    # use the latest time point available as death
        meta.loc[meta[pd.isnull(meta[COL_DEATH])].index, COL_DEATH] = meta[cols_dates].stack().max()
    else:
        meta[COL_DEATH] = meta[cols_dates].stack().max()

    if align_to_event_title is not None:
        for entity in meta[col_entities].unique():
            offset = meta[
                (meta[col_entities] == entity) &
                (meta[col_events_title] == align_to_event_title)][col_events]\
                    .iloc[0]
            idxs_entity = meta[meta[col_entities] == entity].index
            for col in cols_dates + [COL_DEATH]:
                meta.loc[idxs_entity, col] -= offset

    # group entities according to specific column, if given
    if col_entity_groups is not None:
        meta[COL_GROUP] = meta[col_entity_groups]
    else:
        meta[COL_GROUP] = 1

    # color entities according to specific column, if given
    # define colors for entities
    if colors_entities is None:
        colors_entities = dict()
        if col_entity_colors is not None:
            for entity_category in meta[col_entity_colors].unique():
                colors_entities[entity_category] = AVAILCOLORS[
                    len(colors_entities) % len(AVAILCOLORS)]
        else:
            colors_entities[1] = '#eeeeee'
    legend_entities_entries = []
    if col_entity_colors is not None:
        meta[COL_ENTITY_COLOR] = meta[col_entity_colors]
        for entity_category in meta[col_entity_colors].unique():
            legend_entities_entries.append(
                mpatches.Patch(
                    color=colors_entities[entity_category],
                    label='%s: %s' % (col_entity_colors, entity_category)))
    else:
        meta[COL_ENTITY_COLOR] = 1

    # a DataFrame holding information about entities
    cols = [col_entities, col_birth, COL_DEATH, COL_GROUP, COL_ENTITY_COLOR]
    for col in col_phases_start + col_phases_end:
        if col is not None and col not in cols:
            cols.append(col)
    plot_entities = meta.sort_values(COL_GROUP)[cols].drop_duplicates()
    plot_entities = plot_entities.reset_index().set_index(col_entities)
    if order_entities is not None:
        if set(order_entities) & set(plot_entities.index) == set(plot_entities.index):
            plot_entities = plot_entities.loc[reversed(order_entities),:]#.sort_values(COL_GROUP)
        else:
            raise ValueError("Given order of entities does not match entities in data!")

    # delete old sample_name based index
    del plot_entities[plot_entities.columns[0]]
    plot_entities[COL_YPOS] = range(plot_entities.shape[0])
    groups = list(plot_entities[COL_GROUP].unique())
    if len(groups) > 1:
        for idx in plot_entities.index:
            plot_entities.loc[idx, COL_YPOS] += groups.index(
                plot_entities.loc[idx, COL_GROUP])

    fig, axes = plt.subplots(figsize=(15, plot_entities.shape[0]*height_ratio))

    if colors_phases is None:
        colors_phases = dict()
    # plot phases, i.e. time intervals during something happend to the entities
    for (start, end) in zip(col_phases_start, col_phases_end):
        if start is not None:
            if start not in colors_phases:
                colors_phases[start] = AVAILCOLORS[
                    len(AVAILCOLORS) - 1 - (
                        len(colors_phases) % len(AVAILCOLORS))]
            plt.barh(
                plot_entities[COL_YPOS],
                width=plot_entities[COL_DEATH if end is None else end] -
                plot_entities[start],
                height=1,
                linewidth=0,
                left=plot_entities[start],
                color=colors_phases[start],
            )
            legend_entities_entries.append(
                mpatches.Patch(color=colors_phases[start], label=start))

    plt.barh(
        plot_entities[COL_YPOS],
        width=plot_entities[COL_DEATH] - plot_entities[col_birth],
        height=0.6,
        left=plot_entities[col_birth],
        tick_label=plot_entities.index,
        linewidth=0,
        color=plot_entities[COL_ENTITY_COLOR].apply(
            lambda x: colors_entities.get(x, 'black')),
    )
    plt.xlabel(timeresolution)
    plt.ylabel(col_entities)
    # improve tick frequency, which is not easy!
    # plt.xticks(np.arange(axes.get_xlim()[0], axes.get_xlim()[1], 27))

    # define colors for events
    if colors_events is None:
        colors_events = dict()
    legend_entries = []
    if col_events_title is not None:
        titles = meta.sort_values(col_events)[col_events_title].unique()
        for i, title in enumerate(titles):
            if title not in colors_events:
                colors_events[title] = AVAILCOLORS[
                    len(colors_events) % len(AVAILCOLORS)]
            legend_entries.append(
                mpatches.Patch(color=colors_events[title], label=title))

    def _get_event_color(colors_events, data, col_events_title):
        if col_events_title is None:
            return 'black'
        return colors_events.get(data[col_events_title], 'black')

    for entity in plot_entities.index:
        pos_y = plot_entities.loc[entity, COL_YPOS]
        for idx, row in meta[meta[col_entities] == entity].iterrows():
            linestyle = 'solid'
            if (counts is not None) and (idx not in counts.columns):
                linestyle = 'dotted'
            plt.vlines(x=row[col_events],
                       color=_get_event_color(colors_events,
                                              row, col_events_title),
                       linestyle=linestyle,
                       lw=event_line_width,
                       ymin=pos_y-1/2, ymax=pos_y+1/2)

    legends_left_pos = 1.01
    if len(groups) > 1:
        ax2 = axes.twinx()
        ax2.yaxis.set_ticks_position("right")
        group_labels = (plot_entities.groupby(COL_GROUP)[COL_YPOS].min() + \
            (plot_entities.groupby(COL_GROUP).size()-1)/2).sort_values()
        ax2.set_yticks(group_labels)
        ax2.set_yticklabels(group_labels.index)
        ax2.set_ylabel(col_entity_groups)
        ax2.set_ylim(axes.get_ylim())
        legends_left_pos += 0.05

    if len(legend_entries) > 0:
        legend_events = plt.legend(
            handles=legend_entries, loc='upper left',
            bbox_to_anchor=(legends_left_pos, 1.05), title=col_events_title)
    if len(legend_entities_entries) > 0:
        plt.legend(
            handles=legend_entities_entries, loc='lower left',
            bbox_to_anchor=(legends_left_pos, 0.05))
        if len(legend_entries) > 0:
            plt.gca().add_artist(legend_events)

    return fig, colors_events, plot_entities


def get_dictvalue(_hash, _key, default):
    if _hash is None:
        return default
    return _hash.get(_key, default)


def _get_group_name(group):
    if type(group) == str:
        return group
    else:
        return ' '.join(group)


def plot_timecourse(metadata: pd.DataFrame, data: pd.Series,
                    col_events: str, col_entities: str,
                    cols_groups: [str]=None, colors_groups=None, cis_groups=None, dashes_groups=None,
                    intervals: [(float, float)]=None,
                    ax=None, test_groups=None, alternative: str='two-sided', pthreshold: float=0.05,
                    print_samplesizes=True, print_legend=True):
    idx_samples = set(metadata.index) & set(data.index)
    meta = metadata.loc[idx_samples, :].copy()
    meta['__fakegroup__'] = 'all'
    if cols_groups is None:
        cols_groups = ['__fakegroup__']
    if data.name in meta.columns:
        raise ValueError("Your metadata already contains the column '%s', which is the same name as your numeric data!" % data.name)
    meta_data = meta.merge(data, left_index=True, right_index=True, how='left')

    if intervals is not None:
        for (start, stop) in intervals:
            if start >= stop:
                raise ValueError("Intervals: start need to be smaller than end")
            ax.axvspan(start, stop, facecolor='#fff09d', zorder=0)

    ax.grid(b=True, which='major', color='lightgray', linewidth=1.0, axis='x')

    legend_elements = []
    xtick_labels = []
    group_sizes = []
    if colors_groups is None:
        colors_groups = dict()
    for group in meta.groupby(cols_groups).size().index.to_numpy():
        if group not in colors_groups:
            colors_groups[group] = sns.color_palette()[len(colors_groups)]
        # plot lines
        data_group = meta_data[(meta_data[cols_groups] == group).all(axis=1)]
        sns.lineplot(data=data_group,
                     y=data.name, x=col_events,
                     color=get_dictvalue(colors_groups, group, None),
                     #ci=get_dictvalue(cis_groups, group, 95),
                     errorbar=('ci', get_dictvalue(cis_groups, group, 95)),
                     dashes=[x for x in [get_dictvalue(dashes_groups, group, None)] if x is not None],
                     style=(None if get_dictvalue(dashes_groups, group, None) is None else '__fakegroup__'),
                     ax=ax)

        # fill legend
        legend_elements.append(Line2D([0], [0], color=get_dictvalue(colors_groups, group, None), lw=4, label=_get_group_name(group), linestyle=':' if get_dictvalue(dashes_groups, group, None) is not None else None))

        # collect group size information for ticks
        ns = data_group.groupby(col_events).size()
        ns.name = _get_group_name(group)
        group_sizes.append(ns)

    if print_legend:
        ax.legend(handles=legend_elements)
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # ensure every time point in metadata gets its own tick + grid line
    existing_ticks = set(ax.get_xticks())
    if ax.get_subplotspec().get_geometry()[2:] == (0, 0):
        existing_ticks = set()
    ax.set_xticks(sorted(set(meta_data[col_events].unique()) | existing_ticks))

    group_tests = []
    if test_groups is not None:
        for (group_A, group_B) in test_groups:
            g_res = {}
            for event, g in meta_data.groupby(col_events):
                try:
                    res = mannwhitneyu(
                        g[(g[cols_groups] == group_A).all(axis=1)][data.name].values,
                        g[(g[cols_groups] == group_B).all(axis=1)][data.name].values, alternative=alternative)
                    g_res[event] = res.pvalue if res.statistic > 0 else 1
                except ValueError as e:
                    if str(e) == 'All numbers are identical in mannwhitneyu':
                        g_res[event] = 1
            g_res = pd.Series(g_res)
            g_res.name = '"%s" %s "%s"' % (_get_group_name(group_A), {'two-sided': 'vs', 'greater': '>', 'less': '<'}.get(alternative), _get_group_name(group_B))
            group_tests.append(g_res)
        group_tests = pd.concat(group_tests, axis=1).fillna(1)
    else:
        group_tests = pd.DataFrame()

    # print sample sizes on top x axis
    group_sizes = pd.concat(group_sizes, axis=1, sort=False).fillna(0).astype(int)
    ax_numsamples = ax.twiny()
    ax_numsamples.set_xticks(list(ax.get_xticks()) + [ax.get_xlim()[1]])
    ax_numsamples.set_xlim(ax.get_xlim())

    xlabels = []
    # x tick labels must not always be floats, e.g. generation descriptions
    if group_sizes.reindex(ax.get_xticks()).unstack().dropna().shape[0] > 0:
        group_sizes = group_sizes.reindex(ax.get_xticks())
    if print_samplesizes:
        xlabels.append(group_sizes.applymap(lambda x: "" if pd.isna(x) else str(x)))
    if test_groups:
        xlabels.append(group_tests.applymap(lambda x: 'p=%.2f' % x if x < pthreshold else ''))
    if len(xlabels) > 0:
        xlabels = pd.concat(xlabels, axis=1).apply(lambda x: '\n'.join(x), axis=1)
        xlabels.loc[ax.get_xlim()[1]] = '\n'.join((['\n'.join(map(lambda x: 'n: %s' % x, group_sizes.columns))] if print_samplesizes else []) + \
                                                   ['\n'.join(group_tests.columns)])
        ax_numsamples.set_xticklabels(xlabels)
        ax_numsamples.get_xticklabels()[-1].set_ha("left")


def plot_timecourse_beta(metadata: pd.DataFrame, beta: DistanceMatrix, metric_name: str,
                    col_events: str,
                    col_groups: str=None, colors_groups=None,
                    intervals: [(float, float)]=None,
                    ax=None, pthreshold: float=0.05,
                    print_legend=True, print_samplesizes=True, legend_title=None):
    idx_samples = set(metadata.index) & set(beta.ids)
    metadata = metadata.loc[idx_samples, :]
    beta = beta.filter(idx_samples, strict=False)

    grp_values = sorted(metadata[col_groups].unique())
    if len(grp_values) != 2:
        raise ValueError("Column %s has more or less than 2 states: %s. Cannot visualize more than two states." % (col_groups, grp_values))
    group_names = ['intra: %s' % grp_values[0], 'inter', 'intra: %s' % grp_values[-1]]

    distances = []
    group_tests = []
    # collect group size information for ticks
    group_sizes = metadata.groupby([col_events, col_groups]).size().unstack().fillna(0)
    for event, g in metadata.groupby(col_events):
        for group, g_intra in g.groupby(col_groups):
            for (idx_a, idx_b) in combinations(g_intra.index, 2):
                distances.append({'type': 'intra: %s' % group,
                                  col_events: event,
                                  metric_name: beta[idx_a, idx_b]})

        idx_as = g[g[col_groups] == grp_values[0]].index
        idx_bs = g[g[col_groups] == grp_values[1]].index
        if min(len(idx_as), len(idx_bs)) > 1:
            for idx_a in idx_as:
                for idx_b in idx_bs:
                    distances.append({'type': 'inter',
                                      col_events: event,
                                      metric_name: beta[idx_a, idx_b]})

        x = plotNetworks(col_groups, g, None, {'beta': beta}, summarize=True)
        pvalue = 1
        testname = None
        if 'p-value' in x[0].columns:
            pvalue = x[0]['p-value'].iloc[0]
            testname = '%s: %i' % (x[0]['test name'].iloc[0], x[0]['num_permutations'].iloc[0])
        group_tests.append({col_events: event, 'p-value': pvalue, 'testname': testname})
    group_tests = pd.DataFrame(group_tests).set_index(col_events)
    testname = 'failing'
    if  group_tests['testname'].dropna().shape[0] > 0:
        testname = group_tests['testname'].dropna().iloc[0]
    group_tests = group_tests.rename(columns={'p-value': testname})
    del group_tests['testname']

    distances = pd.DataFrame(distances)

    # ensure every time point in metadata gets its own tick + grid line
    existing_ticks = set(ax.get_xticks())
    if ax.get_subplotspec().get_geometry()[2:] == (0, 0):
        existing_ticks = set()
    xticks = sorted(existing_ticks | set(metadata[col_events].unique()))

    interval_size = min(map(lambda x: abs(x[0]-x[1]), zip(sorted(distances[col_events].unique()), sorted(distances[col_events].unique())[1:])))
    # if False:#ax.get_subplotspec().get_geometry()[2:] != (0, 0):
    #     distances_spikeins = []
    #     for x in range(int(min(xticks)), int(max(xticks))+1, 1):
    #         distances_spikeins.append({'type': 'inter', col_events: x, metric_name: np.nan})
    #     distances = distances.append(pd.DataFrame(distances_spikeins))
    #interval_size = int(min(map(lambda x: abs(x[0]-x[1]), zip(xticks, xticks[1:]))) * 0.6)
    #
    if intervals is not None:
        for (start, stop) in intervals:
            if start >= stop:
                raise ValueError("Intervals: start need to be smaller than end")
            ax.axvspan(start, stop, facecolor='#fff09d', zorder=0)

    xlim = ax.get_xlim()
    colors = {'inter': 'white'}
    if colors_groups is not None:
        for k,v in colors_groups.items():
            colors['intra: %s' % k] = v
    for group in group_names:
        if group not in colors:
            colors[group] = sns.color_palette()[len(colors)]
    for event, g in distances.groupby(col_events):
        box_width = (interval_size / 4)
        for pos, group in enumerate(group_names):
            bp = ax.boxplot(g[g['type'] == group][metric_name], manage_ticks=False, positions=[event + ((pos-1) * (interval_size/4))], widths=box_width, patch_artist=True, sym='d')
            for patch in bp['boxes']:
                patch.set(facecolor=colors[group])
            plt.setp(bp['medians'], color="black")
            plt.setp(bp['fliers'], markerfacecolor='gray')
            plt.setp(bp['fliers'], markeredgecolor='gray')

     # sns.boxplot(data=distances, y=metric_name, x=col_events,
     #             palette=colors,
     #             hue='type',
     #             hue_order=['intra: %s' % grp_values[0], 'inter', 'intra: %s' % grp_values[1]],
     #             ax=ax, manage_ticks=False)
    ax.xaxis.grid(True)
    ax.set_xticks(xticks)
    if ax.get_subplotspec().get_geometry()[2:] != (0, 0):
        ax.set_xlim(xlim)
        ax.set_xticklabels(xticks)

    if print_legend is True:
        legend_elements = [Patch(label=group, facecolor=colors[group], edgecolor='black') for group in group_names]
        ax.legend(handles=legend_elements)
        if legend_title is None:
            ax.get_legend().set_title(col_groups)
        else:
            ax.get_legend().set_title(legend_title)

    ax_numsamples = ax.twiny()
    ax_numsamples.set_xticks(list(ax.get_xticks()) + [ax.get_xlim()[1]])
    ax_numsamples.set_xlim(ax.get_xlim())

    xlabels = pd.concat([group_sizes.reindex(ax.get_xticks()).applymap(lambda x: "" if pd.isnull(x) else str(int(x))) if print_samplesizes else pd.DataFrame(),
                         group_tests.reindex(ax.get_xticks()).applymap(lambda x: 'p=%.2f' % x if x < pthreshold else '')], axis=1).apply(lambda x: '\n'.join(x), axis=1)
    xlabels.loc[ax.get_xlim()[1]] = '\n'.join((['\n'.join(map(lambda x: 'n: %s' % x, group_sizes.columns))] if print_samplesizes else []) + \
                                               ['\n'.join(group_tests.columns)])
    ax_numsamples.set_xticklabels(xlabels)
    ax_numsamples.get_xticklabels()[-1].set_ha("left")
    ax.set_ylabel(metric_name)


def get_empV4region(sequence: str):
    """Very naive method to extract V4 regions from larger sequences,
       by pattern matching with EMP primer sequences. This does not
       respect imperfect primer annealing. PrimerProspector might be
       the better choice for the same operation!!"""
    reference = DNA(sequence)
    primers = {'fwd': DNA("GTGYCAGCMGCCGCGGTAA"),
               'rev': DNA("GGACTACNVGGGTWTCTAAT")}

    # find primer positions in input sequence
    positions = {k: [] for k,v in primers.items()}
    for _type in ['fwd', 'rev']:
        for prm in primers[_type].expand_degenerates():
            if _type == 'rev':
                prm = prm.complement(reverse=True)
            pos = 0
            while True:
                try:
                    pos = reference.index(prm, pos)
                    positions[_type].append(pos)
                except ValueError as e:
                    if 'is not present in' in str(e):
                        break
                pos += 1
        if len(positions['fwd']) <= 0:
            break

    # slice sub-sequences
    slices = [reference[f+len(primers['fwd']):r] for f in positions['fwd'] for r in positions['rev'] if r - f > 0]
    for i, s in enumerate(slices):
        # I learned those parameters from the GG alignment
        if len(s) < 150:
            sys.stderr.write("Warning from get_v4region: the %i. found sub-sequence seems to be too short (< 150).\n" % (i+1))
        if len(s) > 260:
            sys.stderr.write("Warning from get_v4region: the %i. found sub-sequence seems to be too long (> 260).\n" % (i+1))

    return slices


def split_tree_into_clusters(tree: TreeNode, maxdepth: float=0.3, return_fake_taxonomy: bool=True):
    """Traverses given tree and clusters tips according to maximal depth of sub-tree.
       Useful to cluster e.g. sequences with similarity threshold.

       Parameters
       ----------
       tree : skbio.tree.TreeNode
          Input tree.
       maxdepth : float
          Default: 0.3.
          Maximal length of path from root of sub-tree to all of it's tips.
          The smaller, the deeper the cuts, the smaller but more resulting clusters.
       return_fake_taxonomy : bool
          Default: True.

       Returns
       -------
       dict(
        'clusters': dict('cluster_name': [tip names]),
        'taxonomy': pd.Series(tip name: cluster_name),
        'tree': skbio.tree.TreeNode,
       )
       Returned tree is the reduced input tree, where tips are now the found clusters.
    """
    tree_internal = tree.deepcopy()

    for node in tree_internal.postorder():
        node.max_tip_length = None
        if node.length is None:
            node.length = 0

    # round 1: compute maximal distance from node to longest tip
    for node in tree_internal.postorder():
        if node.is_tip():
            node.max_tip_length = 0
        else:
            node.max_tip_length = max([child.length + child.max_tip_length for child in iter(node)])

    # round 2: find clusters
    clusters = dict()
    for node in tree_internal.levelorder():
        if node.max_tip_length <= maxdepth:
            # add all tips to return set
            cluster_name = 'cluster_%i' % (len(clusters)+1)
            clusters[cluster_name] = {subnode.name for subnode in node.preorder() if subnode.is_tip()}

            node.children = []
            node.name = 's__%s' % cluster_name

    taxonomy = None
    if return_fake_taxonomy:
        taxonomy = dict()
        for cluster_name, taxa in clusters.items():
            for taxon in taxa:
                taxonomy[taxon] = 'k__; p__; c__; o__; f__; g__; s__%s' % cluster_name
        taxonomy = pd.Series(taxonomy)

    return {'clusters': clusters, 'taxonomy': taxonomy, 'tree': tree_internal}


def predict_timecourse(counts: pd.DataFrame, meta: pd.DataFrame, col_time: str, col_entities: str, col_prediction: str, train_test_ratio: float=0.5, onlyusetimes=None, err=sys.stderr, iterations=2, usefeatures=None):
    counts = counts.loc[:, set(counts.columns) & set(meta.index)]
    meta = meta.loc[set(counts.columns) & set(meta.index), :]
    if err is not None:
        err.write('Using %i samples: ' % meta.shape[0])

    tps = sorted(meta[col_time].unique())
    if onlyusetimes is not None:
        tps = []
        for tp in sorted(onlyusetimes):
            if meta[meta[col_time] == tp].shape[0] <= 0:
                raise ValueError("Time point %s does not exist in your samples. You have time points [%s]" % (tp, ', '.join(map(str, sorted(meta[col_time].unique())))))
            tps.append(tp)

    if usefeatures is not None:
        iterations = 1
        for feature in usefeatures:
            if feature not in counts.index:
                raise ValueError("Feature '%s' is not in your count table!" % feature)

    data = []
    for hsid, g in meta.groupby(col_entities):
        samples = []
        for tpbin35 in tps:
            if g[g[col_time] == tpbin35].shape[0] > 0:
                x = counts.loc[:, g[g[col_time] == tpbin35].index[0]]
            else:
                x = pd.Series(index=counts.index)
                x.index.name = 'Species'
            x.name = '%s_%s' % (hsid, tpbin35)
            samples.append(x)
        x = pd.concat(samples, axis=1)
        x = x.fillna(0).stack().reset_index().rename(columns={0: 'counts'})
        debug = x
        x['feature'] = x.apply(lambda row: row['Species']+'@'+row['level_1'].split('_')[-1], axis=1)
        x = x.set_index('feature')
        x = x['counts']
        x.name = hsid
        data.append(x)
    timecounts = pd.concat(data, axis=1, sort=False).fillna(0).astype(int)

    results = []
    for i in range(iterations):
        if usefeatures is None:
            coi = random.sample(list(counts.index), random.randint(1, 40))
        else:
            coi = usefeatures

        # 50% train, 50% test
        clf = RandomForestClassifier(n_estimators=1000, n_jobs=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            timecounts.loc[['%s@%s' % (cluster, timepoint) for cluster in coi for timepoint in tps], :].T,
            meta.groupby(col_entities)[col_prediction].unique().apply(lambda x: x[0]),
            test_size=train_test_ratio,
            random_state=42)

        # train the ML tool
        clf = clf.fit(X_train, y_train)
        # make predictions for test samples
        prediction = pd.Series(clf.predict(X_test), index=X_test.index)
        # assess accurracy
        clf.accurracy = clf.score(X_test, y_test)

        results.append({
            'accurracy': clf.accurracy,
            'iteration': i+1,
            '#clusters': len(coi),
            'timepoints': sorted(tps),
            'clusters': sorted(coi),
            'mean_counts': counts.loc[coi, :].sum(axis=0).mean(),
        })

    return [pd.DataFrame(results), timecounts]


def randomForest_phenotype(counts: pd.DataFrame, phenotype: pd.Series, iterations: int=10, train_test_ratio: float=0.5, title=None):
    results = []
    for i in range(iterations):
        if i == 0:
            random_state=42
        else:
            random_state=None
        clf = RandomForestClassifier(n_estimators=1000, n_jobs=1, random_state=random_state)
        idx_samples = sorted(set(phenotype.index) & set(counts.columns))
        X_train, X_test, y_train, y_test = train_test_split(counts.loc[:, idx_samples].T, phenotype.loc[idx_samples], test_size=train_test_ratio, random_state=random_state)
        clf = clf.fit(X_train, y_train)
        prediction = pd.Series(clf.predict(X_test), index=X_test.index)
        results.append({'iteration': i,
                        'accurracy': clf.score(X_test, y_test),
                        'train': y_train,
                        'test': y_test,
                        'prediction': prediction,
                        'clf': clf})
    results = pd.DataFrame(results)

    fig, axes = plt.subplots(1,2)

    ax = axes[0]
    chosen_iteration = results.sort_values(by='accurracy').index[int(results.shape[0]/2)]

    plot_confusion_matrix(results.loc[chosen_iteration, 'test'],
                          results.loc[chosen_iteration, 'prediction'],
                          title='training: %s\ntesting: %s\nmedian accurracy: %.3f' % (', '.join(['n=%i %s' % (n, phenotype) for phenotype, n in results.loc[chosen_iteration, 'train'].value_counts().sort_index().iteritems()]),
                                                                                ', '.join(['n=%i %s' % (n, phenotype) for phenotype, n in results.loc[chosen_iteration, 'test'].value_counts().sort_index().iteritems()]),
                                                                                results.loc[chosen_iteration, 'accurracy']),
                          xtickrotation=0, ax=ax, verbose=None)

    ax = axes[1]
    sns.swarmplot(results['accurracy'], ax=ax, orient='v')
    ax.set_xlabel('%i iterations' % iterations)
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_ticks_position("right")

    if title is not None:
        fig.suptitle(title)

    reported_prediction = results.loc[chosen_iteration, 'prediction']
    reported_prediction.name = 'prediction'
    return fig, results.loc[chosen_iteration, 'clf'], results['accurracy'], {'training': y_train, 'testing': y_test, 'prediction': reported_prediction}


def sync_counts_metadata(featuretable: pd.DataFrame, metadata: pd.DataFrame, verbose=sys.stderr):
    """Subsets samples that appear in feature-table and metdata.

    Parameters
    ----------
    featuretable : pd.DataFrame
        Feature-table. Columns are samples, rows are features.
    metadata : pd.DataFrame
        metadata information for samples.

    Returns
    -------
    (featuretable: pd.DataFrame, metadata: pd.DataFrame)
    """
    sub_featuretable = featuretable.loc[:, [s for s in featuretable.columns if s in metadata.index]]
    sub_metadata = metadata.loc[[s for s in metadata.index if s in featuretable.columns]]

    if sub_featuretable.shape[1] == 0:
        raise ValueError("No samples common in feature-table and metadata!")
    if (sub_featuretable.shape[1] < featuretable.shape[1]) or (sub_metadata.shape[0] < metadata.shape[0]):
        if verbose is not None:
            verbose.write('Reduced to %i samples (feature-table had %i, metadata had %i samples)\n' % (sub_featuretable.shape[1], featuretable.shape[1], metadata.shape[0]))

    return (sub_featuretable, sub_metadata)


def check_column_presents(metadata: pd.DataFrame, column_names: [str]):
    missing_columns = []
    for colname in column_names:
        if colname is None:
            continue
        if colname not in metadata.columns:
            missing_columns.append(colname)
    if len(missing_columns) > 0:
        raise ValueError("The following %i column(s) are not present in the provided table: %s" % (len(missing_columns), ', '.join(missing_columns)))


# https://stackoverflow.com/questions/5300938/calculating-the-position-of-points-in-a-circle
# https://math.stackexchange.com/questions/12166/numbers-of-circles-around-a-circle
def _make_circles(items: [str], center=(0,0), level=1, width=6, squared=False):
    squared_circs_per_row = np.ceil(np.sqrt(len(items)))

    circles = dict()
    if len(items) > 2:
        r = math.sin(math.pi/len(items)) / (1-math.sin(math.pi/len(items)))
        zoom = width/(4*r + 2)
        if squared:
            inner_radius = (width / squared_circs_per_row) / 2
        else:
            inner_radius = r*zoom
        outer_radius = (r+1)*zoom
    else:
        inner_radius = width/2/len(items)
        outer_radius = inner_radius
    if len(items) > 1:
        for i, item in enumerate(reversed(sorted(items))):
            degree = i * (360 / len(items))
            if squared:
                inner_center = (((i %  squared_circs_per_row)+0.5) * (width / squared_circs_per_row) - (width/2),
                                (((i // squared_circs_per_row)+0.5) * (width / squared_circs_per_row) - (width/2)) * -1)
            else:
                inner_center = (
                    center[0] + outer_radius * np.cos(i * (2*math.pi / len(items))),
                    center[1] + outer_radius * np.sin(i * (2*math.pi / len(items))))
            if squared:
                label_position = (
                    inner_center[0],
                    inner_center[1]+inner_radius*0.8)
            else:
                label_position = (
                    center[0] + (outer_radius+inner_radius*1.25) * np.cos(i * (2*math.pi / len(items))),
                    center[1] + (outer_radius+inner_radius*1.25) * np.sin(i * (2*math.pi / len(items))))
            link_angle = random.random()*2*math.pi
            link_position = (
                inner_center[0] + (inner_radius*0.5) * np.cos(link_angle),
                inner_center[1] + (inner_radius*0.5) * np.sin(link_angle))
            if squared:
                label_horizontalalignment = 'center'
            else:
                label_horizontalalignment = 'center' if (degree == 90 or degree == 270) else 'right' if (90 < degree < 270) else 'left'
            circles[item] = {'center': inner_center, 'radius': inner_radius, 'level': level,
                             'label_position': label_position,
                             'link_position': link_position,
                             'label_horizontalalignment': label_horizontalalignment}
    else:
        circles[items[0]] = {'center': center, 'radius': inner_radius, 'level': level,
                             'label_position': center,
                             'link_position': center,
                             'label_horizontalalignment': 'center'}

    return circles


def plot_circles(meta: pd.DataFrame, cols_grps: Dict[str, str]=None, colors: Dict[str, Dict[str, str]]=None, ax: plt.axis=None, width: float=6, links=[], print_labels: bool=True, space_for_missing_entries: bool=False, squared=False):
    """
    Parameters
    ----------
    meta : pd.DataFrame
        Metadata with multi-index.
        That is all that's needed - number of levels decides about number of circle levels.
    cols_grps : dict(str: str)
        For coloring:
        Dict: keys = multi-index level, values = column in meta providing state for color code.
    colors : dict(str: dict(str: str))
        Dict of Dicts. First level key refers to multi-index level, second key to color state and value is RGB color for drawing.
    ax : plt.axis
        Default: None, i.e. new figure will be created.
        Otherwise, axis can be provided to which plot will be generated.
    width : float
        Default: 6.0
        Width of drawing.
    links : [(idx, idx, field]
        List of tuples of indices: Links between objects.
        "field" is optional and is used to color the link. Works only if colors["links"] is provided!
    print_labels : bool
        Default: True
    space_for_missing_entries : bool
        Default: False
    Returns
    -------

    """
    circles = ()
    if type(meta.index) == pd.core.indexes.base.Index:
        level_names = [meta.index.name]
    else:
        level_names = list(map(lambda x: x.name, meta.index.levels))

    meta['__blank'] = False
    if space_for_missing_entries:
        blanks = pd.DataFrame(index=set(pd.MultiIndex.from_product(meta.index.levels)) - set(meta.index))
        blanks['__blank'] = True
        meta = pd.concat([meta, blanks], sort=False)

    for i in range(len(level_names)):
        if i == 0:
            circles = _make_circles(meta.index.get_level_values(i).unique(), level=level_names[i], squared=squared)
        else:
            for idx, g in meta.groupby(level_names[:i]):
                circles.update(_make_circles(list(set(map(lambda x: x[:i+1], g.index))),
                                             center=circles[idx]['center'],
                                             level=level_names[i],
                                             width=circles[idx]['radius']*(2-(0.1*(i+1)))))
    #return circles
    if ax is None:
        fig, ax = plt.subplots()
    # plot circles
    for i, (idx, c) in enumerate(circles.items()):
        color = 'black'
        if (cols_grps is not None) and (c['level'] in cols_grps):
            def _get_uniq_value(data):
                if type(data) == str:
                    return data
                else:
                    return data.iloc[0]
            value = None
            if cols_grps[c['level']] in meta.index.names:
                selection = meta.xs(idx, drop_level=False)
                if type(selection) == pd.core.series.Series:
                    selection = selection.to_frame().T
                    selection.index.names = meta.index.names
                value = _get_uniq_value(selection.reset_index().loc[:, cols_grps[c['level']]])
            else:
                value = _get_uniq_value(meta.loc[idx, cols_grps[c['level']]])
            color = colors.get(c['level'], {None: 'black'}).get(value, 'black')
        if meta.loc[idx, '__blank'].all():
            color="white"
        ax.add_artist(plt.Circle(c['center'], c['radius'], color=color,
                                 fill=c['level'] == level_names[-1]))
        if print_labels:
            if len(level_names) > 1:
                if (c['level'] == level_names[0]):
                    ax.text(*c['label_position'], idx if type(idx) != tuple else idx[-1], horizontalalignment=c['label_horizontalalignment'], verticalalignment='center')
            if (c['level'] == level_names[-1]):
                ax.text(*c['center'], idx if type(idx) != tuple else idx[-1], horizontalalignment='center', verticalalignment='center', fontsize=12*(width/10))

    # plot links
    AVIAL_LINESTYLES = ['--', '-.', ':', '-']
    if len(links) > 0:
        links = pd.DataFrame(links)
        if links.shape[1] == 2:
            links['color'] = 'black'
            links['linestyle'] = AVIAL_LINESTYLES[0]
        else:
            if 'links' in colors:
                links['color'] = links[2].apply(lambda x: colors['links'].get(x), 'black')
                links_ls = dict()
                for i, type_ in enumerate(links[2].unique()):
                    links_ls[type_] = AVIAL_LINESTYLES[i % len(AVIAL_LINESTYLES)]
                links['linestyle'] = links[2].apply(lambda x: links_ls[x])
        for _, g in sorted(links.groupby('color'), key=lambda x: x[1].shape[0], reverse=True):
            for _, link in g.iterrows():
                xa, ya = circles[link.loc[0]]['link_position']
                xb, yb = circles[link.loc[1]]['link_position']
                ax.plot([xa,xb],[ya,yb], ls=link['linestyle'], color=link['color'], lw=width/4)

    ax.set_xlim(-1*width/2*1.3,width/2*1.3)
    ax.set_ylim(-1*width/2*1.3,width/2*1.3)

    legend_elements = []
    if colors is not None:
        for level in colors.keys():
            if level == 'links':
                continue
            for value, color in colors[level].items():
                legend_elements.append(Patch(
                    label=value, facecolor=color))
        if ('links' in colors) and (len(links) > 0):
            for value, color in colors['links'].items():
                legend_elements.append(Line2D([0],[0], label=value, color=color, ls=links_ls.get(value, '-')))
    if len(legend_elements) > 0:
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.41, 1))
    ax.axis('off')

    return circles


def adjust_saturation(color, amount=0.5):
    # copied from https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], amount, amount)

def plot_plate(meta:pd.DataFrame, col_position='well_id', col_label='sample_type', col_texts=None, colors=dict(), highlight_samples=set(), show_legend=True):
    """Draw a 96-well plate layout.

    meta : pd.DataFrame
        The full metadata.
    col_position : str
        The column holding well positions in format G12 or G08
    col_label : str
        The label for the wells
    col_texts : str
        Texts to draw in wells
    highlight_samples : set
        Set of samples that shall be highlighted by increasing the stroke width
    show_legend : bool
        Draw legend.
        Default = True
    """
    ROWS = ['A','B','C','D','E','F','G','H']
    COLS = [1,2,3,4,5,6,7,8,9,10,11,12]

    fig, axes = plt.subplots(1,1,figsize=(12,8))
    for col in range(1,len(COLS)+1):
        for row in range(1,len(ROWS)+1):
            axes.add_patch(plt.Circle((col, row), 0.4, color='lightgray', fill=False))
    axes.set_xlim((0,len(COLS)+1))
    axes.set_ylim((0,len(ROWS)+1))

    axes.set_yticks(range(1,len(ROWS)+1))
    axes.set_yticklabels(ROWS)
    axes.set_xticks(range(1,len(COLS)+1))
    axes.set_xticklabels(COLS)
    axes.invert_yaxis()

    availColors = \
        sns.color_palette('Paired', 12) +\
        sns.color_palette('Dark2', 12) +\
        sns.color_palette('Pastel1', 12)
    for idx, sample in meta.iterrows():
        if not pd.isnull(sample[col_position]):
            row, col = ROWS.index(sample[col_position][0])+1, int(sample[col_position][1:])
            norm_label = sample[col_label]
            if pd.isnull(sample[col_label]):
                norm_label = "nan"
            if norm_label not in colors:
                #print("%s not in colors" % sample[norm_label])
                colors[norm_label] = availColors[len(colors) % len(availColors)]
            color = colors[norm_label]
            axes.add_patch(plt.Circle((col, row), 0.4, color=color, fill=True))
            if idx in highlight_samples:
                axes.add_patch(plt.Circle((col, row), 0.4, fill=False, edgecolor="black", linewidth=5))
            if (col_texts is not None) and pd.notnull(sample[col_texts]):
                axes.text(col, row, sample[col_texts], horizontalalignment='center', verticalalignment='center', fontdict={'fontsize': 'x-small'})
    sorted_keys = []
    if show_legend:
        try:
            sorted_keys = sorted(colors.keys())
        except TypeError:
            sorted_keys = colors.keys()
        axes.legend(handles=[Patch(color=colors[val], label=val) for val in sorted_keys], loc='upper left', bbox_to_anchor=(1.01, 1), title=col_label)
    axes.set_title('Plate Layout by "%s" and "%s"' % (col_position, col_label))

    return fig
