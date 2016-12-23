import pandas as pd
import os
from os.path import commonprefix
import sys

from ggmap.readwrite import read_metaphlan_profile


def update_taxids(input, updatedTaxids):
    """ Updates a map of sequenceIDs to taxIDs with information from merged.dmp

    Some of NCBI's taxonomy IDs might get merged into others. Older sequences
    can still point to the old taxID. The new taxID must be looked up in the
    merged.dmp file in order to be able to find the right node in the taxonomy
    tree.

    Parameters
    ----------
    input : dict of dicts of sets
        The keys of the outer dicts are the sequence types, e.g. NC, GeneID or
        gi. Keys of the inner dict are OTUs or clades. Values of the inner
        dict are sets of taxIDs.
    updatedTaxids : dict
        Content of merged.dmp in form of a dict where key is current taxID and
        value the new taxID

    Returns
    -------
    The original map, but some taxIDs might have been updated.
    """
    for seqType in input:
        for seqID in input[seqType]:
            cur_taxid = input[seqType][seqID]
            if cur_taxid in updatedTaxids:
                input[seqType][seqID] = updatedTaxids[cur_taxid]
    return input


def convert_profiles(profile_filenames, map_clade_otu, prefix="",
                     out=sys.stderr):
    """
    Converts a list of MetaPhlAn profiles into one GreenGenes OTU table.

    Parameters
    ----------
    profile_filenames : list of str
        A list of filenames of MetaPhlAn profiles that should be converted.
    map_clade_otu : dict clades -> set(OTUs)
        A dict with MetaPhlAn clades as keys and set of GreenGenes OTUs as
        values.
    prefix : str
        Optional. We try to come up with speaking names for the table columns
        by substracting the common pre- and suffix. You might want to add a
        descriptive prefix to the name. Default = "".
    out : filehandle
        Filehandle onto which error messages should be written.

    Returns
    -------
    A pandas DataFrame for the converted OTU table.

    Raises
    ------
    IOError
        If one of the given file does not exist.
    """
    # check if all files exist
    for filename in profile_filenames:
        if not os.path.exists(filename):
            raise IOError('Profile "%s" does not exist.' % filename)

    # for speaking names for columns, we substract the longest common prefix
    # and suffix of each filename.
    common_prefix = os.path.commonprefix(profile_filenames)
    common_suffix = os.path.commonprefix(list(map(lambda n: n[::-1],
                                                  profile_filenames)))[::-1]

    otu_profiles = []
    for filename in profile_filenames:
        name = prefix + filename[len(common_prefix):-len(common_suffix)]
        x = _convert_metaphlan_profile_to_greengenes(
            read_metaphlan_profile(filename),
            map_clade_otu,
            out=out)
        otu_profiles.append(pd.Series(x,
                                      index=x.keys(),
                                      name=name))

    return pd.concat(otu_profiles, axis=1).fillna(0.0)


def _convert_metaphlan_profile_to_greengenes(profile, map_clade_otu,
                                             out=sys.stderr):
    """ Converts one MetaPhlAn profile into a GreenGenes OTU profile.

    Each clade in the profile is mapped to its OTU(s) and the clades relative
    abundance is equally distributed across the set of matching OTUs. Finally,
    the profile is normalized to sum up to 1.0.

    Parameters
    ----------
    profile : dict clade -> relative abundance
        A dict, holding for each clade its realtive abundance (un-normalized).
    map_clade_otu : dict clades -> set(OTUs)
        A dict with MetaPhlAn clades as keys and set of GreenGenes OTUs as
        values.
    out : filehandle
        Filehandle onto which error messages should be written.

    Returns
    -------
    A dict where keys are OTUs and values are relative abundances.
    Normalized to 1.
    """
    result = {}
    missedClades = {}
    for lineage in profile:
        clade = None
        for c in reversed(lineage.split("|")):
            if c in map_clade_otu:
                clade = c
                break
        if clade in map_clade_otu:
            otus = map_clade_otu[clade]
            for otu in otus:
                if otu not in result:
                    result[otu] = 0
                result[otu] += profile[lineage] / len(otus)
        else:
            missedClades[lineage] = profile[lineage]

    # renormalize to 1
    s = sum(result.values())
    for otu in result:
        result[otu] /= s

    if len(missedClades) > 0:
        out.write(("Due to %i unmatched MetaPhlAn lineages, we missed %.2f of "
                   "the relative abundance! Those clades are:\n") %
                  (len(missedClades),
                   sum(missedClades.values()) / sum(profile.values())))
        for clade in sorted(missedClades):
            out.write("\t%s\t%f\n" % (clade, missedClades[clade]))

    return result


# WHAT MUST BE DONE TO CREATE A MAP FROM MP TO GG!
# def generate_clade2otus_map():
#     nodes = read_ncbi_nodes('/home/sjanssen///GreenGenes/NCBItaxonomy/nodes.dmp')
#     merged = read_ncbi_merged('/home/sjanssen///GreenGenes/NCBItaxonomy/merged.dmp')
#     tax = build_ncbi_tree(nodes, True)
#
#     taxids_metaphlan = read_taxid_list('/home/sjanssen///GreenGenes/Cache/taxids_metaphlan.txt')
#     taxids_metaphlan = update_taxids(taxids_metaphlan, merged)
#     clades_metaphlan = read_metaphlan_markers_info('/home/sjanssen///GreenGenes/Metaphlan/markers_info.txt')
#
#     gg_ids_accessions = read_gg_accessions('/home/sjanssen///GreenGenes/gg_13_5_accessions.txt')
#     gg_taxids = read_taxid_list('/home/sjanssen///GreenGenes/Cache/taxids_greengenes13_5.txt')
#     gg_taxids = update_taxids(gg_taxids, merged)
#     gg_otumap_97_orig = read_gg_otu_map('/home/sjanssen///GreenGenes/gg_13_5_otus/otus/97_otu_map.txt', gg_ids_accessions)
#
#     tree_gg = map_onto_ncbi(tax, gg_otumap_97_orig, gg_taxids, 'otus', verbose=True)
#     tree_mp = map_onto_ncbi(tax, clades_metaphlan, taxids_metaphlan, 'mp_clades', verbose=True)
#
#     resmap = match_metaphlan_greengenes(clades_metaphlan.keys(), tree_mp, 'mp_clades', tree_gg, 'otus')
#
#     return resmap
