import tempfile
import shutil
import subprocess
import sys

import pandas as pd

from ggmap.snippets import (pandas2biom, cluster_run)


FILE_REFERENCE_TREE = '/projects/emp/03-otus/reference/97_otus.tree'


def _parse_alpha(num_iterations, workdir, rarefaction_depth):
    coll = dict()
    for iteration in range(num_iterations):
        x = pd.read_csv('%s/alpha_rarefaction_%i_%i.txt' % (
            workdir,
            rarefaction_depth,
            iteration), sep='\t', index_col=0)
        if iteration == 0:
            for metric in x.columns:
                coll[metric] = pd.DataFrame(data=x[metric])
                coll[metric].columns = [iteration]
        else:
            for metric in x.columns:
                coll[metric][iteration] = x[metric]

    result = pd.DataFrame(index=list(coll.values())[0].index)
    for metric in coll.keys():
        result[metric] = coll[metric].mean(axis=1)

    return result


def alpha_diversity(counts, metrics, rarefaction_depth,
                    num_threads=10, num_iterations=10, dry=True,
                    use_grid=True):
    """ Computes alpha diversity values for given BIOM table.

    Paramaters
    ----------
    counts : Pandas.DataFrame
        OTU counts
    metrics : [str]
        Alpha diversity metrics to be computed.
    rarefaction_depth : int
        Rarefaction depth that must be applied to counts.
    num_threads : int
        Number of parallel threads. Default: 10.
    num_iterations : int
        Number of iterations to rarefy the input table.
    dry : boolean
        Do NOT run clusterjobs, just print commands. Default: True
    use_grid : boolean
        Use grid engine instead of local execution. Default: True

    Returns
    -------
    Pandas.DataFrame: alpha diversity values for each sample (rows) for every
    chosen metric (columns)."""

    # create a temporary working directory
    workdir = tempfile.mkdtemp(prefix='ana_alpha_')

    # store counts as a biom file
    pandas2biom(workdir+'/input.biom', counts)

    # create a mock metadata file
    metadata = pd.DataFrame(index=counts.columns)
    metadata.index.name = '#SampleID'
    metadata['mock'] = 'foo'
    metadata.to_csv(workdir+'/metadata.tsv', sep='\t')

    commands = []
    commands.append(('parallel_multiple_rarefactions.py '
                     '-T '                       # direct polling
                     '-i %s '                    # input biom file
                     '-m %i '                    # min rarefaction depth
                     '-x %i '                    # max rarefaction depth
                     '-s 1 '                     # depth steps
                     '-o %s '                    # output directory
                     '-n %i '                    # number iterations per depth
                     '--jobs_to_start %i') % (   # number parallel jobs
        workdir+'/input.biom',
        rarefaction_depth,
        rarefaction_depth,
        workdir+'/rarefactions',
        num_iterations,
        num_threads))

    commands.append(('parallel_alpha_diversity.py '
                     '-T '                      # direct polling
                     '-i %s '                   # dir to rarefied tables
                     '-o %s '                   # output directory
                     '--metrics %s '            # list of alpha div metrics
                     '-t %s '                   # tree reference file
                     '--jobs_to_start %i') % (  # number parallel jobs
        workdir+'/rarefactions',
        workdir+'/alpha_div/',
        ",".join(metrics),
        FILE_REFERENCE_TREE,
        num_threads))

    sys.stderr.write("Working directory is '%s'. " % workdir)

    if not use_grid:
        if dry:
            sys.stderr.write("\n\n".join(commands))
            return None
        with subprocess.Popen("source activate qiime_env && %s" %
                              " && ".join(commands),
                              shell=True,
                              stdout=subprocess.PIPE) as call_x:
            if (call_x.wait() != 0):
                raise ValueError("something went wrong")
    else:
        cluster_run(commands, 'ana_alpha', workdir+'mock', 'qiime_env',
                    ppn=num_threads, wait=True, dry=dry)

    results = _parse_alpha(num_iterations, workdir+'/alpha_div/',
                           rarefaction_depth)

    if results is not None:
        results.index.name = counts.index.name
        shutil.rmtree(workdir)
        sys.stderr.write("Was removed.\n")

    return results
