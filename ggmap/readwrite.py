
def read_ncbi_nodes(filename):
    """ Reads NCBI's nodes.dmp file and returns a dict of nodes and parents.

    Parameters
    ----------
    filename : str
        Path to the filename 'nodes.dmp' of NCBI's taxonomy.

    Returns
    -------
    A dict, where keys are node IDs and their values are parent node IDs.

    Raises
    ------
    IOError
        If the file cannot be read.
    ValueError
        If IDs of nodes or parent nodes cannot be converted into int.
    """
    nodes = {}
    try:
        file = open(filename, 'r')
        for line in file:
            fields = line.split('\t|\t')
            try:
                nodes[int(fields[0])] = int(fields[1])
            except ValueError:
                file.close()
                raise ValueError("cannot convert node IDs (%s, %s) to int."
                                 % (fields[0], fields[1]))

        file.close()
        return nodes

    except IOError:
        raise IOError('Cannot read file "%s"' % filename)


def read_metaphlan_markers_info(filename):
    """ Reads the MetaPhlAn markers_info.txt file.

    MetaPhlAn's OTU analogous are 'clades'. Currently, they have around 8900.
    A 'clade' is composed of one or many (sub)sequences of specific marker
    genes. Those marker genes come from three sources: 1) genbank: "^gi|",
    2) gene: "^GeneID:", and 3) NCBI nr: "^NC_".

    Parameters
    ----------
    filename : str
        Path to the filename 'markers_info' of MetaPhlAn.

    Returns
    -------
    A dict with an entry for each 'clade'. Their values are dicts themselves,
    with keys that refer to one of the three sequence sources. And their values
    are sets of marker gene IDs. For example:
    's__Escherichia_phage_vB_EcoP_G7C': {'GeneID': {'11117645', '11117646'}}

    Raises
    ------
    IOError
        If the file cannot be read.
    """
    clades = {}
    try:
        file = open(filename, 'r')
        for line in file:
            if line.startswith('gi|'):
                type_ids = 'gi'
                accession = (line.split('\t')[0]).split('|')[1]
            elif line.startswith('GeneID:'):
                type_ids = 'GeneID'
                accession = (line.split('\t')[0]).split(':')[1]
            elif line.startswith('NC_'):
                type_ids = 'NC'
                accession = line.split('\t')[0]
            else:
                type_ids = None
                accession = None

            if (type_ids is not None) and (accession is not None):
                clade = line.split("clade': '")[1].split("'")[0]
                if clade not in clades:
                    clades[clade] = {}
                if type_ids not in clades[clade]:
                    clades[clade][type_ids] = {}
                clades[clade][type_ids][accession] = True

        for clade in clades:
            for type_id in clades[clade]:
                clades[clade][type_id] = set(clades[clade][type_id].keys())

        file.close()
        return clades

    except IOError:
        raise IOError('Cannot read file "%s"' % filename)


def read_taxid_list(filename, dict={}):
    """ Read a taxID list file.

    A taxID list file consists of three tab separated columns: 1. ID type,
    2. ID of sequence, 3. NCBI taxonomy ID for the sequence. It is headed by
    one line starting with the '#' char.

    Parameters
    ----------
    filename : str
        Path to the file containing the taxID list.
    dict : dict
        Optional. Provide an existing dictionary into which parsed results
        should be added. Useful if the taxID list consists of several files.

    Returns
    -------
    A dict of dict. First dict's keys are the sequence types, e.g. "gi",
    "GeneID", "NC". Second level keys are the sequence IDs and their values are
    the according NCBI taxonomy IDs, or taxIDs for short.

    Raises
    ------
    IOError
        If the file cannot be read.
    ValueError
        If a line does not contain of exactly three tab delimited fields.
    """
    try:
        f = open(filename, 'r')
        f.readline()  # header
        for line in f:
            try:
                type, accession, taxid = line.rstrip().split("\t")
                if type not in dict:
                    dict[type] = {}
                dict[type][accession] = taxid
            except ValueError:
                f.close()
                raise ValueError("Error parsing line '%s' of file '%s'" %
                                 (line, filename))

        f.close()
        return dict
    except IOError:
        raise IOError('Cannot read file "%s"' % filename)
