import urllib.request
import tempfile
import glob
import sys
import threading
import time


LOGFILE_PREFIX = 'log_taxid_'
LOGFILE_SUFFIX = '.txt'
exitFlag = 0


class thread_fetch(threading.Thread):
    def __init__(self, threadID, name, accessions, chunk_size):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.accessions = accessions
        self.chunk_size = chunk_size

    def run(self):
        print("Starting %s with %i accessions." %
              (self.name, len(self.accessions)))
        fetchTaxids(self.accessions, chunk_size=self.chunk_size)
        print("Exiting " + self.name)


def parse_gg_accessions(filename, abort_after_lines=20):
    """ Reads the GreenGenes accession list.

    Parameters
    ----------
    filename: str
        Path to the file containing GreenGenes accessions.

    Returns
    -------
    A dict that holds all accessions, split into accession types e.g. Genbank,
    IMG
    """
    accessions = {}
    try:
        file = open(filename, 'r')
        file.readline()  # header
        readlines = 0
        for line in file:
            readlines += 1
            gg_id, accession_type, accession = line.rstrip().split("\t")
            if accession_type not in accessions.keys():
                accessions[accession_type] = {}
            accessions[accession_type][gg_id] = accession
            if (abort_after_lines is not None) and (readlines >= abort_after_lines):
                break
        file.close()
        return accessions
    except IOError:
        print('Cannot read file')


def write_accession_taxids(dict, filehandle=None, verbose=True):
    if filehandle is None:
        filehandle = tempfile.NamedTemporaryFile(dir="./",
                                                 delete=False,
                                                 prefix=LOGFILE_PREFIX,
                                                 suffix=LOGFILE_SUFFIX,
                                                 mode='w')
        filehandle.write("#" + "\t".join(['Accession', 'NCBI-taxid']) + "\n")
    if verbose:
        print("  logged %i accessions to file '%s'" %
              (len(dict), filehandle.name), file=sys.stderr)
    for accession, taxid in dict.items():
        filehandle.write("\t".join([accession, str(taxid)]) + "\n")
    filehandle.flush()

    return filehandle


def read_accesion_taxids(filename, dict={}):
    try:
        f = open(filename, 'r')
        f.readline()  # header
        for line in f:
            accession, taxid = line.rstrip().split("\t")
            dict[accession] = taxid
        f.close()
        return dict
    except IOError:
        print('Cannot read file')


def _get_taxids_cache(accessions, verbose=True):
    if len(accessions) > 0:
        cache = {}
        if verbose:
            print('searching cache: ', file=sys.stderr, end="")
        for logfile in glob.glob("./%s*%s" % (LOGFILE_PREFIX, LOGFILE_SUFFIX)):
            read_accesion_taxids(logfile, cache)
        results = {}
        for id in accessions:
            if id in cache:
                results[id] = cache[id]
        if verbose:
            print('found %i of %i accessions.' % (
                    len(results), len(accessions)),
                  file=sys.stderr)
        return results
    else:
        return {}


def _get_taxids_http(accessions, verbose=True, log_results=True, chunk_size=100):
    if len(accessions) > 0:
        if verbose:
            print("fetching %i accessions from EBI:" % len(accessions),
                  file=sys.stderr)
        chunks = [accessions[i:i+chunk_size]
                  for i in range(0, len(accessions), chunk_size)]

        results = {}
        log = None
        for chunk_accessions in chunks:
            chunk_results = {}
            url = ('http://www.ebi.ac.uk/ena/data/view/%s&display=xml&header=true'
                   % ','.join(chunk_accessions))
            if verbose:
                print('  chunk %i: requesting %i accessions: ...' % (
                        chunks.index(chunk_accessions),
                        len(chunk_accessions)),
                      file=sys.stderr, end="")
            response = urllib.request.urlopen(url)

            for line in response.read().decode('utf-8').split('\n'):
                if line.startswith('<entry accession="'):
                    for field in line.split(" "):
                        if field.startswith('accession'):
                            accession = (field.split('=')[1])[1:-1]
                        elif field.startswith('version'):
                            accession += "." + (field.split('=')[1])[1:-1]
                elif 'taxId="' in line:
                    for field in line.split(" "):
                        if field.startswith('taxId'):
                            taxid = (field.split('=')[1])[1:-2]
                            chunk_results[accession] = int(taxid)
            if verbose:
                print(' got %i.' % (len(chunk_results)), file=sys.stderr)
                #print('missing accession: %s' % ",".join([ accession for accession in chunk_accessions if accession not in chunk_results]))

            if log_results and len(chunk_results) > 0:
                log = write_accession_taxids(chunk_results, filehandle=log)
            results = {**results, **chunk_results}
        return results
    else:
        return {}


def fetchTaxids(accessions, verbose=True, log_results=True, chunk_size=100):
    """ Fetches NCBI taxonomy IDs via EBI for a list of accessions.
    """

    # first, check if we have not already information about the accession
    cached_results = _get_taxids_cache(accessions, verbose)

    # second, for the remaining accessions, start a REST request to EBI
    http_accessions = list(set(accessions) - set(cached_results.keys()))
    http_results = _get_taxids_http(http_accessions, verbose, log_results,
                                    chunk_size=chunk_size)

    return {**cached_results, **http_results}


def slice_it(li, cols=2):
    start = 0
    for i in range(cols):
        stop = start + len(li[i::cols])
        yield li[start:stop]
        start = stop


if __name__ == "__main__":
    abort_after_lines = None
    num_threads = 1
    inner_chunk_size = 100
    file_input = '/home/sjanssen/GreenGenes/gg_13_5_accessions.txt'

    r = parse_gg_accessions(file_input, abort_after_lines=abort_after_lines)
    genbank_ids = [v for k, v in r['Genbank'].items()]
    chunks = list(slice_it(genbank_ids, num_threads))
    threads = []
    for chunk in chunks:
        threads.append(thread_fetch(
            chunks.index(chunk),
            "Thread-%i" % chunks.index(chunk),
            chunk,
            inner_chunk_size
        ))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print("Exiting Main Thread")
