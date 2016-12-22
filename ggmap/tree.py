import sys

from skbio.tree import TreeNode


def get_lineage(taxid, nodes):
    """ Obtain whole lineage for a given taxID.

    Parameters
    ----------
    taxid : int
        taxID of the node for which the lineage should be obtained.
    nodes : dict(ID: parentID)
        Dictionary containing the whole taxonomy. Key is node taxID, value is
        parent taxID.

    Returns
    -------
    A list of taxIDs ending with the given taxid.
    """
    lineage = [taxid]
    if taxid not in nodes:
        raise ValueError('%s not in nodes' % taxid)
    else:
        while nodes[lineage[-1]] != lineage[-1]:
            lineage.append(nodes[lineage[-1]])
        return list(reversed(lineage))


def build_ncbi_tree(nodes, verbose=False, out=sys.stdout):
    """ Build a TreeNode from a dict of nodes.

    Parameters
    ----------
    nodes : dict(ID: parentID)
        Dictionary containing the whole taxonomy. Key is node taxID, value is
        parent taxID.
    verbose : Boolean
        Print verbose status information while executing. Default = False
    out : file handle
        File handle into verbosity information should be printed.

    Returns
    -------
    A skbio.tree.TreeNode object, holding the whole taxonomy.
    """

    parents = set(nodes.values())
    tips = set(nodes) - parents

    if verbose:
        out.write("build ncbi tree for %i tips: " % len(list(tips)))
    ls = {}
    for c, tip in enumerate(list(tips)):
        if verbose and (int(c % (len(tips) / 100)) == 0):
            out.write(".")
        try:
            ls[tip] = get_lineage(tip, nodes)[:-1]
        except KeyError:
            raise KeyError("Cannot obtain lineage for taxid %s" % tip)
    tree = TreeNode.from_taxonomy(ls.items())
    if verbose:
        out.write(" done.\n")

    return tree
