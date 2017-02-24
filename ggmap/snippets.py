import pandas as pd
import biom
from biom.util import biom_open


def biom2pandas(file_biom, withTaxonomy=False, astype=int):
    """ Converts a biom file into a Pandas.DataFrame

    Parameters
    ----------
    file_biom : str
        The path to the biom file.
    withTaxonomy : bool
        If TRUE, returns a second Pandas.DataFrame with lineage information for
        each feature, e.g. OTU or deblur-sequence. Default: FALSE
    astype : type
        datatype into each value of the biom table is casted. Default: int.
        Use e.g. float if biom table contains relative abundances instead of
        raw reads.

    Returns
    -------
    A Pandas.DataFrame holding holding numerical values from the biom file.
    If withTaxonomy is TRUE then a second Pandas.DataFrame is returned, holding
    lineage information about each feature.

    Raises
    ------
    IOError
        If file_biom cannot be read.
    ValueError
        If withTaxonomy=TRUE but biom file does not hold taxonomy information.
    """
    try:
        table = biom.load_table(file_biom)
        counts = pd.DataFrame(table.matrix_data.T.todense().astype(astype),
                              index=table.ids(axis='sample'),
                              columns=table.ids(axis='observation')).T
        if withTaxonomy:
            otu_ids = table.ids(axis='observation')
            if table.metadata(otu_ids[0], axis='observation') is not None:
                if 'taxonomy' in table.metadata(otu_ids[0],
                                                axis='observation'):
                    mapping = {i: table.metadata(id=i, axis='observation')
                               ['taxonomy']
                               for i in otu_ids}
                    taxonomy = pd.DataFrame(mapping,
                                            index=['kingdom', 'phylum',
                                                   'class', 'order', 'family',
                                                   'genus', 'species']).T
                    return counts, taxonomy
            raise ValueError('No taxonomy information found in biom file.')
        else:
            return counts
    except IOError:
        raise IOError('Cannot read file "%s"' % file_biom)


def pandas2biom(file_biom, table):
    """ Writes a Pandas.DataFrame into a biom file.

    Parameters
    ----------
    file_biom: str
        The filename of the BIOM file to be created.
    table: a Pandas.DataFrame
        The table that should be written as BIOM.

    Raises
    ------
    IOError
        If file_biom cannot be written.

    TODO
    ----
        1) also store taxonomy information
    """
    try:
        bt = biom.Table(table.values,
                        observation_ids=table.index,
                        sample_ids=table.columns)
        with biom_open(file_biom, 'w') as f:
            bt.to_hdf5(f, "example")
    except IOError:
        raise IOError('Cannot write to file "%s"' % file_biom)


def parse_splitlibrarieslog(filename):
    """ Parse the log of a QIIME split_libraries_xxx.py run.

    Especially deal with multiple input files, i.e. several sections in log.

    Parameters
    ----------
    filename : str
        The filename of the log to parse.

    Returns
    -------
    A Pandas.DataFrame containing two column with 'counts' and sample name for
    each sample in the log file.
    (We might see duplicate sample names from multiple input files, thus we
     cannot make the sample name the index.)

    Raises
    ------
    IOError
        If filename cannot be read.
    """
    try:
        counts = []
        f = open(filename, 'r')
        endOfFile = False
        while not endOfFile:
            # find begin of count table
            while True:
                line = f.readline()
                if 'Median sequence length:' in line:
                    break
            # collect counts
            while True:
                line = f.readline()
                if line == '\n':
                    break
                samplename, count = line.split()
                counts.append({'sample': samplename, 'counts': count})
            # check if file contains more blocks
            while True:
                line = f.readline()
                if 'Input file paths' in line:
                    break
                if line == '':
                    endOfFile = True
                    break
        return pd.DataFrame(sorted(counts,
                                   key=lambda x: int(x['counts']),
                                   reverse=True), dtype=int)
    except IOError:
        raise IOError('Cannot read file "%s"' % filename)
