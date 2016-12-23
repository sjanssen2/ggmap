def _read_ncbitaxonomy_file(filename):
    """ A generic function to read an NCBI taxonomy file, which is delimited
    by '\t|\t?'.

    Parameters
    ----------
    filename : str
        Path to a file from an NCBI taxonomy dump.

    Returns
    -------
    A dict. Key = first field of file, Value = second field of file.

    Raises
    ------
    IOError
        If the file cannot be read.
    ValueError
        If IDs of entries cannot be converted into int.
    """
    entries = {}
    try:
        file = open(filename, 'r')
        for line in file:
            fields = list(map(str.strip, line.split('|')))
            try:
                entries[int(fields[0])] = int(fields[1])
            except ValueError:
                file.close()
                raise ValueError("cannot convert entry IDs (%s, %s) to int."
                                 % (fields[0], fields[1]))

        file.close()
        return entries

    except IOError:
        raise IOError('Cannot read file "%s"' % filename)


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
    return _read_ncbitaxonomy_file(filename)


def read_ncbi_merged(filename):
    """ Reads NCBI's merged.dmp file and returns a dict of old and merged IDs.

    Parameters
    ----------
    filename : str
        Path to the filename 'merged.dmp' of NCBI's taxonomy.

    Returns
    -------
    A dict, where keys are old IDs and their values are the new merged IDs.

    Raises
    ------
    IOError
        If the file cannot be read.
    ValueError
        If IDs of old or merged nodes cannot be converted into int.
    """
    return _read_ncbitaxonomy_file(filename)


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


def read_taxid_list(filename, dict=None):
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
    if dict is None:
        dict = {}
    try:
        f = open(filename, 'r')
        f.readline()  # header
        for line in f:
            try:
                type, accession, taxid = line.rstrip().split("\t")
                if type not in dict:
                    dict[type] = {}
                dict[type][accession] = int(taxid)
            except ValueError:
                f.close()
                raise ValueError("Error parsing line '%s' of file '%s'" %
                                 (line, filename))

        f.close()
        return dict
    except IOError:
        raise IOError('Cannot read file "%s"' % filename)


def read_gg_accessions(filename):
    """ Reads a GreenGenes accession list.

    Parameters
    ----------
    filename: str
        Path to the file containing GreenGenes accessions.

    Returns
    -------
    A dict that holds all accessions, split into accession types e.g. Genbank,
    IMG

    Raises
    ------
    IOError
        If the file cannot be read.
    """
    accessions = {}
    try:
        file = open(filename, 'r')
        file.readline()  # header
        for line in file:
            try:
                gg_id, accession_type, accession = line.rstrip().split("\t")
                if accession_type not in accessions.keys():
                    accessions[accession_type] = {}
                accessions[accession_type][int(gg_id)] = accession
            except ValueError:
                file.close()
                raise ValueError("Wrong number of tab seperated columns.")
        file.close()
        return accessions
    except IOError:
        raise IOError('Cannot read file "%s"' % filename)


def read_gg_otu_map(filename, accessions):
    """ Reads a GreenGenes OTU map.

    Parameters
    ----------
    filename : str
        Path to the file containing GreenGenes OTU map.
    accessions :
        GreenGenes accession list

    Returns
    -------
    ...

    Raises
    ------
    IOError
        If the file cannot be read.
    """
    otus = {}
    try:
        file = open(filename, 'r')
        for line in file:
            fields = line.rstrip().split("\t")
            otu_repr = int(fields[1])
            otus[otu_repr] = {}
            otu_members = list(map(int, fields[1:]))
            for otu in otu_members:
                for ctype in accessions:
                    if otu in accessions[ctype]:
                        if ctype not in otus[otu_repr]:
                            otus[otu_repr][ctype] = set()
                        otus[otu_repr][ctype].add(accessions[ctype][otu])
        file.close()
        return otus
    except IOError:
        raise IOError('Cannot read file "%s"' % filename)
    except ValueError:
        file.close()
        raise ValueError("wrong file format.")


def write_clade2otus_map(filename, map_clade2otu):
    """ Write MetaPhlAn clades to GreenGenes OTUs map to a file.

    Parameters
    ----------
    filename : str
        Path to the file that shall be created.
    map_clade2otu : dict
        The dict holding the information which MetaPhlAn clade maps to which
        set of GreenGenes OTUs.

    Raises
    ------
    IOError
        If the file cannot be written.
    """
    try:
        fh = open(filename, 'w')
        fh.write('#MetaPhlAn clade\tmatching GreenGenes OTUs\n')
        for clade in sorted(map_clade2otu):
            fh.write('\t'.join([clade] + sorted(map(str,
                                                    map_clade2otu[clade]))))
            fh.write("\n")
        fh.close()
    except IOError:
        raise IOError('Cannot write to file "%s"' % filename)


def read_clade2otus_map(filename):
    """ Read a MetaPhlAn clades to GreenGenes OTUs map file.

    Parameters
    ----------
    filename : str
        Path to the file from which should be read.

    Returns
    -------
    The dict holding the information which MetaPhlAn clade maps to which set of
    GreenGenes OTUs.

    Raises
    ------
    IOError
        If the file cannot be read.
    """
    try:
        map_clade2otu = {}
        fh = open(filename, 'r')
        for line in fh.readlines():
            if not line.startswith('#'):
                fields = line.rstrip().split('\t')
                map_clade2otu[fields[0]] = set(map(int, fields[1:]))
        fh.close()
        return map_clade2otu
    except IOError:
        raise IOError('Cannot read file "%s"' % filename)
