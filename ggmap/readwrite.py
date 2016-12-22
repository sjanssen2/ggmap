
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
