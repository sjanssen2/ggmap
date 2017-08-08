import pandas as pd
import pickle
from random import seed
import sys
import os.path

from skbio import TabularMSA, DNA
from skbio.tree import TreeNode

from ggmap.snippets import mutate_sequence


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


def load_sequences_pynast(file_pynast_alignment, file_backbone_tree,
                          frg_start, frg_stop, frg_expected_length,
                          file_cache=None,
                          verbose=True):
    """Extract fragments from pynast alignment, also in backbone tree.

    Parameters
    ----------
    file_pynast_alignment : file
        Filename for pynast alignment from GreenGenes.
    file_backbone_tree : file
        Filename for SEPPs backbone tree into which fragments shall be
        inserted.
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

    Returns
    -------
    [{'OTUID': str, 'sequence': str}] A list of dicts holding OTUID and
    sequence.
    Note: sequences might come in duplicates, due to degapping.
    """
    if os.path.exists(file_cache):
        f = open(file_cache, 'rb')
        fragments = pickle.load(f)
        f.close()
        if verbose:
            print("% 8i fragments loaded from cache '%s'" %
                  (len(fragments), file_cache))
        return fragments

    # load the full pynast GreenGenes alignment with
    # sequences=1261500 and position=7682
    ali = TabularMSA.read(file_pynast_alignment,
                          format='fasta', constructor=DNA)

    # set index of alignment to sequence IDs
    ali.index = [seq.metadata['id'] for seq in ali]

    if verbose:
        print("% 8i rows and %i cols in alignment '%s'" % (
            ali.shape[0],
            ali.shape[1],
            file_pynast_alignment.split('/')[-1]))

    # load backbone tree and collect all OTU IDs
    otuids_in_backbonetree = [node.name
                              for node
                              in TreeNode.read(file_backbone_tree).tips()]
    if verbose:
        print("% 8i OTUs in backbone tree '%s'" % (
            len(otuids_in_backbonetree),
            file_backbone_tree.split('/')[-1]))

    # subset the alignment to those sequences that are in the backbone tree
    ali_backboneseqs = ali.loc[set(otuids_in_backbonetree) & set(ali.index)]
    if verbose:
        print(("% 8i OTU sequences in backbone tree and alignment. "
               "Surprise: %i OTUs of backbone tree are NOT in alignment!") % (
            ali_backboneseqs.shape[0],
            len(otuids_in_backbonetree) - ali_backboneseqs.shape[0]))
        # To my surprise, not all OTU-IDs of the SEPP reference tree
        # (same with the 99 tree of GreenGenes) are in the pynast alignment.
        # Daniel says: "PyNAST fails on some sequences. The tree is constructed
        # from the ssu-align alignment (based on infernal), but that alignment
        # method is lossy so it is not suitable for extracting variable
        # regions" Therefore, I exclude those 1031 OTU-IDs from further
        # investigation

    # trim alignment down to fragment columns
    ali_fragments = ali_backboneseqs.iloc(axis='position')[frg_start:frg_stop]
    if verbose:
        print("%i -> %i cols: trimming alignment to fragment coordinates" % (
            ali_backboneseqs.shape[1],
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
        print(("% 8i fragments with ungapped length >= %int. "
               "Surprise: %i fragments are too short and %i fragments where "
               "too long (and have been trimmed)!") % (
              len(fragments),
              frg_expected_length,
              ali_fragments.shape[0] - len(fragments), num_frags_toolong))
        # Another surprise is that the trimmed, degapped sequences from pynast
        # alignment do NOT all have length 150nt. Following is a length
        # distribution plot. I checked with Daniel and we decided to omit
        # frgaments smaller than 150nt and timm all other to 150nt.

    f = open(file_cache, 'wb')
    pickle.dump(fragments, f)
    f.close()
    if verbose:
        print("Stored results to cache '%s'" % file_cache)

    # manually deconstruct unused objects to save memory
    # del ali
    # del ali_backboneseqs
    # del ali_fragments
    # del otuids_in_backbonetree

    return fragments


def add_mutations(fragments,
                  max_mutations=10, seednr=42,
                  file_cache=None, verbose=True):
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
    if os.path.exists(file_cache):
        f = open(file_cache, 'rb')
        frgs = pickle.load(f)
        f.close()
        if verbose:
            print("% 8i mutated fragments loaded from cache '%s'" % (
                len(frgs), file_cache))
        return frgs

    # convert fragments into Pandas.DataFrame
    fragments = pd.DataFrame(fragments)
    if verbose:
        print('% 8i fragments to start with' % fragments.shape[0])
    # group fragments by sequence and list true OTUids
    unique_fragments = fragments.groupby('sequence').agg(lambda x:
                                                         list(x.values))
    if verbose:
        print('% 8i fragments after collapsing by sequence' %
              unique_fragments.shape[0])

    # add point mutated sequences to the fragment list
    seed(seednr)  # make sure repeated runs produce the same mutated sequences
    frgs = []
    for i, (sequence, row) in enumerate(unique_fragments.iterrows()):
        for num_mutations in range(0, max_mutations+1):
            frgs.append({'sequence': mutate_sequence(sequence, num_mutations),
                         'OTUIDs': row['OTUID'],
                         'num_pointmutations': num_mutations})
        if i % int(unique_fragments.shape[0]/10) == 0:
            sys.stderr.write('.')
    sys.stderr.write(' done.\n')
    if verbose:
        print('% 8i fragments generated with 0 to %i point mutations.' % (
            len(frgs), max_mutations))

    f = open(file_cache, 'wb')
    pickle.dump(frgs, f)
    f.close()
    if verbose:
        print("Stored results to cache '%s'" % file_cache)

    return frgs
