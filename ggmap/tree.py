import sys

from skbio.tree import TreeNode, MissingNodeError


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


def map_onto_ncbi(taxonomy, clusters, cluster_taxids, attribute_name,
                  verbose=False, out=sys.stdout):
    """Subsets a given NCBI taxonomy to those taxIDs that are used by clusters.

    Clusters might be either OTUs from GreenGenes or Clades from MetaPhlAn.

    Parameters
    ----------
    taxonomy : TreeNode
        The NCBI taxonomy as TreeNode.
    clusters : Dict of dicts of sets
        cluster name: cluster type: accession.
    cluster_taxids : Dict of dicts of taxIDs
        Cluster type: accession: NCBI taxonomy ID
    attribute_name: str
        Name of the attribute which are added to the tree nodes.
    verbose : Boolean
        Print verbose status information while executing. Default = False
    out : file handle
        File handle into verbosity information should be printed.
        Default = sys.stderr

    Returns
    -------
    A subtree of taxonomy, in which nodes are decorated with either MetaPhlAn
    clades or GreenGenes OTUs that match to those taxids.
    """
    if verbose:
        out.write("Starting deep copy (might take 40 seconds): ...")
    tree = taxonomy.deepcopy()
    if verbose:
        out.write(" done.\n")

    for cluster in clusters:
        for ctype in clusters[cluster]:
            taxids = set(map(lambda accession:
                             cluster_taxids[ctype][accession],
                             clusters[cluster][ctype]))
            for taxid in taxids:
                try:
                    node = tree.find(taxid)
                    if not hasattr(node, attribute_name):
                        setattr(node, attribute_name, set())
                        node.isUsed = True
                        for n in node.ancestors():
                            n.isUsed = True
                    attr_set = getattr(node, attribute_name)
                    attr_set.add(cluster)
                    setattr(node, attribute_name, attr_set)
                except MissingNodeError:
                    out.write(("Cannot find taxid %s in taxonomy for "
                               "%s '%s'\n") % (taxid, attribute_name, cluster))

    tree.remove_deleted(lambda node: not hasattr(node, 'isUsed'))

    return tree


def match_metaphlan_greengenes(metaphlan_clades, tree_metaphlan,
                               attr_metaphlan, tree_greengenes,
                               attr_greengenes, out=sys.stderr):
    """ Match all MetaPhlAn clades to GreenGenes OTUs.

    Parameters
    ----------
    metaphlan_clades : list of str
        The MetaPhlAn clades for which GreenGenes OTUs shall be found.
    tree_metaphlan : TreeNode
        The subtree of the NCBI taxonmy tree with annotated MetaPhlAn clades.
        Annotation attribute name must match attr_metaphlan.
    attr_metaphlan : str
        Name of the nodes attribute that holds the set of MetaPhlAn clade
        annotations.
    tree_greengenes : TreeNode
        The subtree of the NCBI taxonmy tree with annotated GreenGenes OTUs.
        Annotation attribute name must match attr_greengenes.
    attr_greengenes : str
        Name of the nodes attribute that holds the set of GreenGenes OTUs
        annotations.
    out : filehandle
        File handle into verbosity information should be printed.
        Default = sys.stderr

    Returns
    -------
    A dict with MetaPhlAn clades as keys and set of GreenGenes OTUs as values.
    """
    clade_to_otu = {}
    for clade in metaphlan_clades:
        try:
            clade_to_otu[clade] = _get_otus_from_clade(clade, tree_metaphlan,
                                                       attr_metaphlan,
                                                       tree_greengenes,
                                                       attr_greengenes)
        except ValueError:
            out.write(("Clade '%s' omitted, since it is not in "
                       "tree_metaphlan.\n") % clade)

    return clade_to_otu


def _get_otus_from_clade(metaphlan_clade, tree_metaphlan, attr_metaphlan,
                         tree_greengenes, attr_greengenes, out=sys.stderr):
    """ Find all GreenGenes OTUs that 'match' to one MetaPhlAn clade.

    The connection between GreenGenes and MetaPhlAn is NCBI taxonomy IDs.
    All OTU clusters of GreenGenes must be translated into according NCBI
    taxonomy IDs and mapped onto the full NCBI taxonomy tree (tree_greengenes).
    Note that one OTU can map to many taxIDs, since it is a cluster of several
    sequences.
    Similarily, all MetaPhlAn clades must be translated and mapped onto the
    same NCBI taxonomy (tree_greengenes). For computational reasons, we are not
    using one tree, but two significantly smaller ones, because only a small
    fraction of NCBI taxIDs are used by GreenGenes and MetaPhlAn.

    To obtain the right set of OTUs for a given clade, we first collect all
    nodes in tree_metaphlan that are annotated with that clade (mp_nodes_clade)
    . Next we find the lowest common ancestor (mp_lca) for this set of nodes.
    We than aim to find a node with the same name as mp_lca in the GreenGenes
    tree. If it exists, we collect all annotated OTUs of this subtree.
    Sometimes the lca from MetaPhlAn does not exists in GreenGenes or the
    according subtree does not contain OTUs. In those cases we acend one level
    from mp_lca and search again. We iterate until we end in the root node.

    Parameters
    ----------
    metaphlan_clade : str
        The MetaPhlAn clade for which GreenGenes OTUs shall be found.
    tree_metaphlan : TreeNode
        The subtree of the NCBI taxonmy tree with annotated MetaPhlAn clades.
        Annotation attribute name must match attr_metaphlan.
    attr_metaphlan : str
        Name of the nodes attribute that holds the set of MetaPhlAn clade
        annotations.
    tree_greengenes : TreeNode
        The subtree of the NCBI taxonmy tree with annotated GreenGenes OTUs.
        Annotation attribute name must match attr_greengenes.
    attr_greengenes : str
        Name of the nodes attribute that holds the set of GreenGenes OTUs
        annotations.
    out : filehandle
        File handle into verbosity information should be printed.
        Default = sys.stderr

    Returns
    -------
    The (maybe empty) set of OTUs matching the given clade.

    Raises
    ------
    ValueError
        If metaphlan_clade is not in the according tree_metaphlan at all. Thus,
        a mapping to GreenGenes OTUs cannot be done.
    """
    def _has_matching_clade(node):
        return hasattr(node, attr_metaphlan) and \
               (metaphlan_clade in getattr(node, attr_metaphlan))

    def _hasOTUs(node):
        return hasattr(node, attr_greengenes)

    mp_nodes_clade = list(tree_metaphlan.find_by_func(_has_matching_clade))
    if len(mp_nodes_clade) > 0:
        mp_clade_names = list(map(lambda node: node.name, mp_nodes_clade))
        mp_lca = tree_metaphlan.lowest_common_ancestor(mp_clade_names)

        otus = []
        if 131567 in map(lambda node: node.name, mp_lca.ancestors()):
            c = mp_lca
            while len(otus) <= 0:
                otus = []
                try:
                    gg_lca_match = tree_greengenes.find(c.name)
                    for node in gg_lca_match.find_by_func(_hasOTUs):
                        otus.extend(getattr(node, attr_greengenes))
                    otus = set(otus)
                except MissingNodeError:
                    c = c.ancestors()[0]

            return set(otus)
        else:
            out.write("'%s' not a cellular organism.\n" % metaphlan_clade)
            return set()
    else:
        raise ValueError("Clade '%s' is not in MetaPhlAn tree." %
                         metaphlan_clade)


def distance_seppinsertion(tree_orig, tree_changed, nodename):
    node_orig = tree_orig.find(nodename)
    node_changed = tree_changed.find(nodename)

    # the newly created internal node (by SEPP) does not come with a name.
    # Thus, we need to use its grandparent node as a common reference in both
    # trees.
    gparent_orig = node_orig.parent.parent
    gparent_changed = node_changed.parent.parent

    dist_pp = gparent_orig.distance(tree_orig.find(gparent_changed.name))
    dist_sub_orig = gparent_orig.distance(node_orig)
    dist_sub_changed = gparent_changed.distance(node_changed)

    def get_nodenames(nodes):
        return [node.name for node in nodes]

    if set(get_nodenames(node_orig.siblings())) == \
       set(get_nodenames(node_changed.siblings())):
        dist_sub_changed *= -1

    return dist_pp + dist_sub_orig + dist_sub_changed
