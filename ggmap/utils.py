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
