import pandas as pd
from random import seed
import sys
import os.path
from io import StringIO
import string
import numpy as np
import seaborn as sns
from itertools import combinations
from scipy.stats import mannwhitneyu

from skbio import TabularMSA, DNA
from skbio.stats.distance import DistanceMatrix, MissingIDError
from skbio.tree import TreeNode, NoLengthError

from ggmap.snippets import mutate_sequence, biom2pandas, RANKS, cache


def read_otumap(file_otumap):
    """Reads a GreenGenes OTU map.

    Parameters
    ----------
    file_otumap : file
        Filename of GreenGenes OTU map to parse.

    Returns
    -------
    A tuple of (Pandas.Series, Pandas.Series), where the first Series has
    representative IDs as index and lists of non-representative IDs as values.
    It's column is named 'non-representatives'.
    The second Series has every sequence ID as index and its OTU ID as value.

    Raises
    ------
    IOError
        If the file cannot be read.
    """
    try:
        # read OTU map line by line
        otus = dict()
        seqs = dict()
        f = open(file_otumap, 'r')
        for line in f:
            fields = line.rstrip().split("\t")
            reprID = str(fields[1])
            seqids = list(map(str, fields[2:]))
            otus[reprID] = seqids
            for seqid in seqids + [reprID]:
                seqs[seqid] = reprID
        f.close()

        # convert to pd.Series
        otus = pd.Series(otus)
        otus.index.name = 'representative'
        otus.name = 'non-representatives'

        seqs = pd.Series(seqs)
        return (otus, seqs)
    except IOError:
        raise IOError('Cannot read file "%s"' % file_otumap)


@cache
def load_sequences_pynast(file_pynast_alignment, file_otumap,
                          frg_start, frg_stop, frg_expected_length,
                          verbose=True, out=sys.stdout, onlyrepr=False):
    """Extract fragments from pynast alignment, also in OTU map.

    Parameters
    ----------
    file_pynast_alignment : file
        Filename for pynast alignment from GreenGenes.
    file_otumap : file
        Filename for the GreenGenes OTU map to use for determining which IDs to
        use for representative and non-representative sequences.
    frg_start : int
        Column number in pynast alignment where fragment left border is
        located.
    frg_stop : int
        Column number in pynast alignment where fragment right border is
        located.
    frg_expected_length : int
        Expected fragment length (needed because degapped alignment rows do
        not always match correct length)
    verbose : Boolean
        Default: True
        If True, print some info on stdout.
    out : StringIO
        Buffer onto which messages should be written. Default is sys.stdout.
    onlyrepr : bool
        Default: False.
        If True, only fragments from representative sequences are returned.

    Returns
    -------
    A list of dicts with following information
    [{'sequence': the nucleotide sequence
      'seqIDs': GreenGenes sequence IDs that contain this fragment
      'otuIDs': GreenGenes OTU IDs that fragment belongs to
      'num_non-representative-seqs': number of sequences the fragment is
                                     sub-sequence in, which are NOT used to
                                     represent an OTU of GreenGenes 99%
      'only_repr._sequences': True if num_non-representative-seqs==0
      'num_pointmutations': number of artifically introduced point mutations}]
    """
    # load the full pynast GreenGenes alignment with
    # sequences=1261500 and position=7682
    ali = TabularMSA.read(file_pynast_alignment,
                          format='fasta', constructor=DNA)

    # set index of alignment to sequence IDs
    ali.index = [seq.metadata['id'] for seq in ali]

    if verbose:
        out.write("% 8i rows and %i cols in alignment '%s'\n" % (
            ali.shape[0],
            ali.shape[1],
            file_pynast_alignment.split('/')[-1]))

    # load OTU map
    (otumap, seqinfo) = read_otumap(file_otumap)
    # all representative seq IDs
    seqids_to_use = list(otumap.index)
    if onlyrepr is False:
        # all non-representative seq IDs
        seqids_to_use += [seqid for otu in otumap.values for seqid in otu]
    if verbose:
        out.write("% 8i sequences in OTU map '%s'\n" % (
            len(seqids_to_use),
            file_otumap.split('/')[-1]))

    # subset the alignment to those sequences that are selected from OTU map
    ali_otumap = ali.loc[set(seqids_to_use) & set(ali.index)]
    if verbose:
        out.write(("% 8i sequences selected from OTU map and alignment. "
                   "Surprise: %i sequences of OTU map are NOT in "
                   "alignment!\n") % (
            ali_otumap.shape[0],
            len(seqids_to_use) - ali_otumap.shape[0]))
        # To my surprise, not all OTU-IDs of the SEPP reference tree
        # (same with the 99 tree of GreenGenes) are in the pynast alignment.
        # Daniel says: "PyNAST fails on some sequences. The tree is constructed
        # from the ssu-align alignment (based on infernal), but that alignment
        # method is lossy so it is not suitable for extracting variable
        # regions" Therefore, I exclude those 1031 OTU-IDs from further
        # investigation

    # trim alignment down to fragment columns
    ali_fragments = ali_otumap.iloc(axis='position')[frg_start:frg_stop]
    if verbose:
        out.write(("%i -> %i cols: trimming alignment to fragment "
                   "coordinates\n") % (
            ali_otumap.shape[1],
            ali_fragments.shape[1]))

    # ungap alignment rows
    fragments = []
    num_frags_toolong = 0
    for fragment_gapped in ali_fragments:
        fragment = fragment_gapped.degap()
        if len(fragment) >= frg_expected_length:
            if len(fragment) > frg_expected_length:
                num_frags_toolong += 1
            fragments.append({
                'sequence': str(fragment)[:frg_expected_length],
                'seqID': fragment_gapped.metadata['id'],
                'otuID': seqinfo.loc[fragment_gapped.metadata['id']]})
    if verbose:
        out.write(("% 8i fragments with ungapped length >= %int. "
                   "Surprise: %i fragments are too short and %i fragments "
                   "where too long (and have been trimmed)!\n") % (
                  len(fragments),
                  frg_expected_length,
                  ali_fragments.shape[0] - len(fragments), num_frags_toolong))
        # Another surprise is that the trimmed, degapped sequences from pynast
        # alignment do NOT all have length 150nt. Following is a length
        # distribution plot. I checked with Daniel and we decided to omit
        # frgaments smaller than 150nt and timm all other to 150nt.

    # convert fragments into Pandas.DataFrame
    fragments = pd.DataFrame(fragments)
    if verbose:
        out.write('% 8i fragments remaining.\n' % fragments.shape[0])
    # group fragments by sequence and list true OTUids
    unique_fragments = fragments.groupby('sequence').agg(lambda x:
                                                         list(x.values))
    if verbose:
        out.write('% 8i unique fragments after collapsing by sequence.\n' %
                  unique_fragments.shape[0])

    frgs = []
    for i, (sequence, row) in enumerate(unique_fragments.iterrows()):
        frgs.append({'sequence': sequence,
                     'seqIDs': row['seqID'],
                     'otuIDs': sorted(list(set(row['otuID']))),
                     'num_non-representative-seqs':
                     len(set(row['seqID']) - set(row['otuID'])),
                     'only_repr._sequences':
                     len(set(row['seqID']) - set(row['otuID'])) == 0,
                     'num_pointmutations': 0})

    return frgs


@cache
def add_mutations(fragments,
                  max_mutations=10, seednr=42,
                  verbose=True,
                  out=sys.stdout, err=sys.stderr):
    """Add point mutated sequences for all fragments provided.

    Parameters
    ----------
    fragments : [{'sequence', 'seqIDs', 'otuIDs',
                  'num_non-representative-seqs', 'only_repr._sequences',
                  'num_pointmutations'}]
        A list of dicts holding seqID, otuID and sequence.
        I.e. the result of load_sequences_pynast()
    max_mutations : int
        Default 10.
        Maximum number of point mutations introduced to fragment sequences.
    seednr : int
        Default 42.
        Seed for random number generate. Used to ensure mutations are the same
        if run several times.
    verbose : Boolean
        Default: True
        If True, print some info on stdout.
    out : StringIO
        Buffer onto which messages should be written. Default is sys.stdout.
    err : StringIO
        Buffer onto which progress should be written. Default is sys.stderr.

    Returns
    -------
    A list of dicts with following information
    [{'sequence': the nucleotide sequence
      'seqIDs': GreenGenes sequence IDs that contain this fragment
      'otuIDs': GreenGenes OTU IDs that fragment belongs to
      'num_non-representative-seqs': number of sequences the fragment is
                                     sub-sequence in, which are NOT used to
                                     represent an OTU of GreenGenes 99%
      'only_repr._sequences': True if num_non-representative-seqs==0
      'num_pointmutations': number of artifically introduced point mutations}]
    """
    frgs = []

    # add point mutated sequences to the fragment list
    seed(seednr)  # make sure repeated runs produce the same mutated sequences
    frgs = []
    divisor = int(len(fragments)/min(10, len(fragments)))
    for i, fragment in enumerate(fragments):
        frgs.append(fragment)
        for num_mutations in range(1, max_mutations+1):
            mut_fragment = dict()
            for field in fragment.keys():
                if field == 'sequence':
                    mut_fragment[field] = mutate_sequence(fragment[field],
                                                          num_mutations)
                elif field == 'num_pointmutations':
                    mut_fragment[field] = num_mutations
                else:
                    mut_fragment[field] = fragment[field]
            frgs.append(mut_fragment)
        if verbose:
            if i % divisor == 0:
                err.write('.')
    if verbose:
        err.write(' done.\n')
    if verbose:
        out.write(('% 8i fragments generated with 0 to %i '
                   'point mutations.\n') % (len(frgs), max_mutations))

    return frgs


@cache
def _measure_distance_single(seppresults, err=sys.stderr, verbose=True):
    """Computes insertion distance error for given sepp results.

    Parameters
    ----------
    seppresults : analyses.sepp['results']['tree']
        The SEPP insertion tree.
    err : StringIO
        Default: sys.stderr
        Buffer onto which status information shall be written.
    verbose : bool
        Default: True
        If True, status information is written to buffer <err>.

    Returns
    -------
    Pandas.DataFrame:
    distance, num_otus, num_mutations, fragment, num_non-representative-seqs
    If 'lca': Distance is the path length between the inserted fragment
    and the lowest common anchestor of all true OTUs.
    If 'closest': Distance is the path length between the inserted fragment
    and the closest true OTU.
    """
    if verbose:
        err.write('read tree...')
    tree = TreeNode.read(StringIO(seppresults))
    if verbose:
        err.write('OK: ')
    results = []
    treesize = tree.count(tips=True)
    for j, fragment in enumerate(tree.tips()):
        if fragment.name.startswith('seqIDs:'):
            # seqIDs:2789969,2491172,4462991,4456388;
            # otuIDs:4462991;
            # num_pointmutations:6;
            # num_non-representative-seqs:3;
            # only_repr._sequences:False
            seq_data = {}
            for field in fragment.name.split(';'):
                kv = field.split(':')
                if kv[0] in ['seqIDs', 'otuIDs']:
                    seq_data[kv[0]] = list(map(str, kv[1].split(',')))
                else:
                    seq_data[kv[0]] = kv[1]
            trueOTUids = seq_data['otuIDs']

            # metric 'lca':
            node_lca = tree.lca(trueOTUids)
            try:
                dist_lca = node_lca.distance(fragment)
            except NoLengthError:
                dist_lca = np.nan

            # metric 'closest':
            dists = []
            for trueOTU in trueOTUids:
                try:
                    dists.append(tree.find(trueOTU).distance(fragment))
                except NoLengthError:
                    dists.append(np.nan)
            dist_closest = min(dists)

            seq_data['distance_lca'] = dist_lca
            seq_data['distance_closest'] = dist_closest
            results.append(seq_data)
        if verbose:
            if j % max(1, int(treesize/100)) == 0:
                err.write('.')
    if verbose:
        err.write(' done.\n')
    return pd.DataFrame(results)


@cache
def measure_distance(sepp_results,
                     err=sys.stderr, verbose=True):
    """Computes SEPP insertion distance of fragments to true nodes.

    Parameters
    ----------
    sepp_results : [analyses.sepp['results']]
        A list of results of a SEPP run, i.e. resulting tree.
    err : StringIO
        Default: sys.stderr
        Buffer onto which status information shall be written.
    verbose : bool
        Default: True
        If True, status information is written to buffer <err>.

    Returns
    -------
    Pandas.DataFrame:
    distance, num_otus, num_mutations, fragment, num_non-representative-seqs
    """
    d = []
    for i, r in enumerate(sepp_results):
        if verbose:
            err.write('part %i/%i: ' % (i+1, len(sepp_results)))
        d.append(_measure_distance_single(r,
                                          err=err, verbose=verbose))
    return pd.concat(d)


def check_qiita_studies(dir_base):
    """Checks if all files for Qiita studies are consistent.

    Notes
    -----
    a) metadata sample files exist
    b) metadata match study ID
    c) biom files for closedref and deblur exist
    d) feature IDs are either numeric (closedref) or nucleotides (deblur)
    e) all samples from biom files are in metadata files

    Correct naming convention:
    dir_base/
        <QIITA-ID>/
            (metadata)
            qiita<QIITA-ID>_sampleinfo.txt
            (otu table closedref. Ensure to use correct fragment size!)
            prep<PREPNAME>/qiita<QIITA-ID>_prep<PREPNAME>_150nt_closedref.biom
            (otu table deblur all. Unfiltered biom table!)
            prep<PREPNAME>/qiita<QIITA-ID>_prep<PREPNAME>_150nt_deblurall.biom

    Parameters
    ----------
    dir_base : path
        Path to root directory containing QIITA study files.

    Returns
    -------
    True, if everything is fine.

    Raises
    ------
    ValueErrors if inconsitencies are found.
    """
    # collect study names
    studies = [d for d in next(os.walk(dir_base))[1] if not d.startswith('.')]

    for study in studies:
        file_sampleinfo = "%s/%s/qiita%s_sampleinfo.txt" % (dir_base,
                                                            study,
                                                            study)

        # check that sample info file exists
        if not os.path.exists(file_sampleinfo):
            raise ValueError('Missing sample info file for study %s!' % study)

        # checks that sample info file matches study by comparing column
        # qiita_study_id
        metadata = pd.read_csv(file_sampleinfo,
                               sep='\t', dtype=str, index_col=0)
        if metadata['qiita_study_id'].unique() != [study]:
            raise ValueError('Wrong sample info file for study %s!' % study)

        preps = [d
                 for d in next(os.walk(dir_base + '/' + study))[1]
                 if not d.startswith('.')]
        for prep in preps:
            # get all existing biom files
            dir_prep = dir_base + '/' + study + '/' + prep
            files_biom = [d
                          for d in next(os.walk(dir_prep))[2]
                          if d.endswith('.biom')]
            fraglen = set(map(lambda x: x.split('_')[-2].split('n')[0],
                              files_biom))
            if len(fraglen) > 1:
                raise ValueError(('found biom files with differing '
                                  'sequence lengths: "%s"') %
                                 '", "'.join(files_biom))
            fraglen = list(fraglen)[0]
            for _type in ['closedref', 'deblurall']:
                file_biom = "%s/%s/%s/qiita%s_%s_%snt_%s.biom" % (
                    dir_base, study, prep, study, prep, fraglen, _type)

                # check that biom file exists
                if not os.path.exists(file_biom):
                    raise ValueError(
                        'Missing biom "%s" file "%s" for %s in study %s, %s!' %
                        (_type, file_biom, _type, study, prep))

                # check biom contents
                counts = biom2pandas(file_biom)
                obs_alphabet = set([str(c).upper()
                                    for idx in counts.index
                                    for c in set(idx)])
                description = None
                if _type == 'closedref':
                    exp_alphabet = set(map(str, range(0, 10)))
                    description = 'numeric'
                else:
                    exp_alphabet = set(string.ascii_uppercase)
                    description = 'nucleotides'
                if len(obs_alphabet - exp_alphabet) > 0:
                    raise ValueError(('Not all feature IDs are purely %s in '
                                      'study %s, %s: "%s') % (
                                     description, study, prep, file_biom))
                if metadata.loc[counts.columns, :].shape[0] <\
                   len(counts.columns):
                    raise ValueError(("Not all samples of %s of study %s are "
                                      "in the metadata file!") % (prep, study))
    return True


def analyse_2014(dir_study, err=sys.stderr):
    """Replicating figure 1a) of 'Human genetics shape the gut microbiome'.
    Beta distances are piled up comparing pairs of MonoZygotic (MZ) twins,
    DiZygotic (DZ) twins and between individuals NOT from the same family
    (unrelated).
    Using Mann-Whitney to test significance between those three groups.

    Does significance / p-values increase when using deblur+SEPP compared to
    closedref?

    Parameters
    ----------
    dir_study : dir as str
        Filepath to directory of Qiita study 2014 (containing prep1).
    err : StringIO
        Default sys.stderr. Where to report status.

    Returns
    -------
    (fig, stats), where fig is a seaborn facetgrid and stats a Pandas.DataFrame
    with statistics about significance of every test.
    """
    NUMSTEPS = 6
    PREP = 'prep1'

    err.write('Running analysis for study 2014:\n')
    err.write('  step 1/%i: loading metadata ...' % NUMSTEPS)
    metadata = pd.read_csv(dir_study + 'qiita2014_sampleinfo.txt',
                           sep='\t', dtype=str, index_col=0)
    err.write(' done.\n')

    err.write('  step 2/%i: load available beta distance matrices ...'
              % NUMSTEPS)
    files_dm = [dir_study+'/'+PREP+'/'+d
                for d in next(os.walk(dir_study+'/' + PREP + '/'))[2]
                if d.endswith('.dm')]
    betas = dict()
    for file_dm in files_dm:
        technique = file_dm.split('/')[-1].split('.')[0].split('_')[-1]
        if technique not in betas:
            betas[technique] = dict()
        metric = file_dm.split('/')[-1].split('.')[-2]
        betas[technique][metric] = DistanceMatrix.read(file_dm)
    err.write(' done.\n')

    err.write('  step 3/%i: obtain beta distance for specific classes ...'
              % NUMSTEPS)
    dists = []
    for file_dm in files_dm:
        technique = file_dm.split('/')[-1].split('.')[0].split('_')[-1]
        metric = file_dm.split('/')[-1].split('.')[-2]
        for zyg in ['MZ', 'DZ']:
            m_class = metadata[metadata['zygosity'] == zyg]
            for (familyid, age), g in m_class.groupby(['familyid', 'age']):
                if g.shape[0] != 2:
                    continue
                try:
                    dists.append({'technique': technique,
                                  'class': zyg,
                                  'distance':
                                  betas[technique][metric][g.index[0],
                                                           g.index[1]],
                                  'metric': metric})
                except MissingIDError:
                    pass

        pddm = betas[technique][metric].to_data_frame()
        for n, g in metadata.loc[pddm.index, :].groupby('familyid'):
            pddm.loc[g.index, g.index] = np.nan
        for dist in pddm.stack().values:
            dists.append({'technique': technique,
                          'class': 'unrelated',
                          'distance': dist,
                          'metric': metric})
    err.write(' done.\n')

    err.write('  step 4/%i: convert dist array to pandas DataFrame ...'
              % NUMSTEPS)
    distances = pd.DataFrame(dists)
    err.write(' done.\n')

    err.write(('  step 5/%i: generate graphical overview '
               'in terms of boxplots ...')
              % NUMSTEPS)
    fig = sns.FacetGrid(distances, col="metric",
                        sharey=False,
                        col_order=['bray_curtis',
                                   'unweighted_unifrac',
                                   'weighted_unifrac'],
                        hue_order=['closedref', 'deblurall'])
    fig = fig.map(sns.boxplot, "class", "distance", "technique").add_legend()
    err.write(' done.\n')

    # generate statistical summary of class comparisons, i.e. did the
    # significance improve?
    err.write(('  step 6/%i: generate statistical summary of '
               'class comparisons ...') % NUMSTEPS)
    stats = []
    for (metric, technique), g in distances.groupby(['metric', 'technique']):
        for (a, b) in combinations(g['class'].unique(), 2):
            t = mannwhitneyu(g[(g['class'] == a)]['distance'].values,
                             g[(g['class'] == b)]['distance'].values,
                             alternative='two-sided')
            stats.append({'technique': technique,
                          'metric': metric,
                          'comparison': '%s vs %s' % (a, b),
                          'p-value': t.pvalue,
                          'test-statistic': t.statistic})
    stats = pd.DataFrame(stats)
    err.write(' done.\n')

    return (fig, stats)


@cache
def get_taxa_radia(file_tree, err=sys.stderr):
    """Computes phylogenetic radius for each taxon in a reference tree.
       Radius is maximal tip to tip distance.

    Parameters
    ----------
    file_tree : str
        Filename of the phylogenetic tree in Newick format.
    err : StringIO
        Default: stderr. Buffer on which status info is printed.

    Returns
    -------
    Pandas.DataFrame with columns "max_tip_tip_dist" and "rank" for every taxon
    in the given file (which is the index of the DataFrame).
    """
    tree = TreeNode.read(file_tree)

    taxa_radia = dict()
    for rank in RANKS:
        status = rank
        if rank != RANKS[-1]:
            status += ', '
        else:
            status += ' done.\n'
        err.write(status)

        for i, node in enumerate(tree.preorder()):
            if node.name is None:
                continue
            if rank[0].lower()+'__' in node.name:
                taxa_radia[node.name] = {
                    'rank': rank,
                    'max_tip_tip_dist': node.get_max_distance()[0]}
    return pd.DataFrame(taxa_radia).T
