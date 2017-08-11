import pandas as pd
import pickle
from random import seed
import sys
import os.path
import string
import numpy as np
import seaborn as sns
from itertools import combinations
from scipy.stats import mannwhitneyu

from skbio import TabularMSA, DNA
from skbio.stats.distance import DistanceMatrix, MissingIDError

from ggmap.snippets import mutate_sequence, biom2pandas


def read_otumap(file_otumap):
    """Reads a GreenGenes OTU map.

    Parameters
    ----------
    file_otumap : file
        Filename of GreenGenes OTU map to parse.

    Returns
    -------
    Pandas.Series with representative IDs as index and lists of
    non-representative IDs as values.
    Column is named 'non-representatives'.

    Raises
    ------
    IOError
        If the file cannot be read.
    """
    try:
        # read OTU map line by line
        otus = dict()
        f = open(file_otumap, 'r')
        for line in f:
            fields = line.rstrip().split("\t")
            otus[str(fields[1])] = list(map(str, fields[2:]))
        f.close()

        # convert to pd.Series
        otus = pd.Series(otus)
        otus.index.name = 'representative'
        otus.name = 'non-representatives'
        return otus
    except IOError:
        raise IOError('Cannot read file "%s"' % file_otumap)


def load_sequences_pynast(file_pynast_alignment, file_otumap,
                          frg_start, frg_stop, frg_expected_length,
                          file_cache=None,
                          verbose=True, out=sys.stdout):
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
    file_cache : file
        Default is None.
        If not None, resulting fragment are cached to this file and if this
        file already exists, results are re-loaded from this file instead of
        being generated newly.
    verbose : Boolean
        Default: True
        If True, print some info on stdout.
    out : StringIO
        Buffer onto which messages should be written. Default is sys.stdout.

    Returns
    -------
    [{'OTUID': str, 'sequence': str}] A list of dicts holding OTUID and
    sequence.
    Note: sequences might come in duplicates, due to degapping.
    """
    if file_cache is not None:
        if os.path.exists(file_cache):
            f = open(file_cache, 'rb')
            fragments = pickle.load(f)
            f.close()
            if verbose:
                out.write("% 8i fragments loaded from cache '%s'\n" %
                          (len(fragments), file_cache))
            return fragments

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
    otumap = read_otumap(file_otumap)
    # all representative seq IDs
    seqids_to_use = list(otumap.index)
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
            fragments.append({'sequence': str(fragment)[:frg_expected_length],
                              'OTUID': fragment_gapped.metadata['id']})
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

    if file_cache is not None:
        f = open(file_cache, 'wb')
        pickle.dump(fragments, f)
        f.close()
        if verbose:
            out.write("Stored results to cache '%s'\n" % file_cache)

    return fragments


def add_mutations(fragments,
                  max_mutations=10, seednr=42,
                  file_cache=None, verbose=True,
                  out=sys.stdout, err=sys.stderr):
    """Add point mutated sequences for all fragments provided.

    Parameters
    ----------
    fragments : [{'OTUID': str, 'sequence': str}]
        A list of dicts holding OTUID and sequence.
        E.g. result of load_sequences_pynast()
    max_mutations : int
        Default 10.
        Maximum number of point mutations introduced to fragment sequences.
    seednr : int
        Default 42.
        Seed for random number generate. Used to ensure mutations are the same
        if run several times.
    file_cache : file
        Default is None.
        If not None, resulting fragment are cached to this file and if this
        file already exists, results are re-loaded from this file instead of
        being generated newly.
    verbose : Boolean
        Default: True
        If True, print some info on stdout.
    out : StringIO
        Buffer onto which messages should be written. Default is sys.stdout.
    err : StringIO
        Buffer onto which progress should be written. Default is sys.stderr.

    Returns
    -------
    [{'OTUIDs': [str], 'sequence': str, 'num_pointmutations': int}]
    A list of dicts, where every dict holds a fragment which consists of the
    three key-value pairs:
    - 'OTUIDs': a list of OTU IDs this fragment belongs to,
    - 'sequence': the fragment sequence with X point mutations,
                  where X is num_pointmutations
    - 'num_pointmutations': number of introduced point mutations
    """
    frgs = []
    if file_cache is not None:
        if os.path.exists(file_cache):
            f = open(file_cache, 'rb')
            frgs = pickle.load(f)
            f.close()
            if verbose:
                out.write("% 8i mutated fragments loaded from cache '%s'\n" % (
                    len(frgs), file_cache))
            return frgs

    # convert fragments into Pandas.DataFrame
    fragments = pd.DataFrame(fragments)
    if verbose:
        out.write('% 8i fragments to start with\n' % fragments.shape[0])
    # group fragments by sequence and list true OTUids
    unique_fragments = fragments.groupby('sequence').agg(lambda x:
                                                         list(x.values))
    if verbose:
        out.write('% 8i fragments after collapsing by sequence\n' %
                  unique_fragments.shape[0])

    # add point mutated sequences to the fragment list
    seed(seednr)  # make sure repeated runs produce the same mutated sequences
    frgs = []
    divisor = int(unique_fragments.shape[0]/min(10, unique_fragments.shape[0]))
    for i, (sequence, row) in enumerate(unique_fragments.iterrows()):
        for num_mutations in range(0, max_mutations+1):
            frgs.append({'sequence': mutate_sequence(sequence, num_mutations),
                         'OTUIDs': row['OTUID'],
                         'num_pointmutations': num_mutations})
        if i % divisor == 0:
            err.write('.')
    err.write(' done.\n')
    if verbose:
        out.write(('% 8i fragments generated with 0 to %i '
                   'point mutations.\n') % (len(frgs), max_mutations))

    if file_cache is not None:
        f = open(file_cache, 'wb')
        pickle.dump(frgs, f)
        f.close()
        if verbose:
            out.write("Stored results to cache '%s'\n" % file_cache)

    return frgs


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
            if len(fraglen) != 1:
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
                          'p-value': t.pvalue})
    stats = pd.DataFrame(stats)
    err.write(' done.\n')

    return (fig, stats)
