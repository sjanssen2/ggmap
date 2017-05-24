import tempfile
import shutil
import subprocess
import sys

import pandas as pd
from skbio.stats.distance import DistanceMatrix

from ggmap.snippets import (pandas2biom, cluster_run)


FILE_REFERENCE_TREE = None
QIIME_ENV = 'qiime_env'


def _get_ref_phylogeny():
    """Use QIIME config to infer location of reference tree."""
    global FILE_REFERENCE_TREE
    if FILE_REFERENCE_TREE is None:
        cmd = "source activate qiime_env"
        print("1:", subprocess.check_output(cmd, shell=True, executable="bash"))
        cmd = "source activate qiime_env"
        print("1:", subprocess.check_output(cmd, shell=True, executable="bash"))
        cmd = "source activate qiime_env && print_qiime_config.py"
        print("2:", subprocess.check_output(cmd, shell=True, executable="bash"))
        cmd = "source activate qiime_env && print_qiime_config.py | grep 'pick_otus_reference_seqs_fp:'"
        print("3:", subprocess.check_output(cmd, shell=True, executable="bash"))

        with subprocess.Popen(("source activate %s && "
                               "print_qiime_config.py "
                               "| grep 'pick_otus_reference_seqs_fp:'" %
                               QIIME_ENV),
                              shell=True,
                              stdout=subprocess.PIPE,
                              executable="bash") as call_x:
            out, err = call_x.communicate()
            if (call_x.wait() != 0):
                print('stderr: %s' % err)
                print('stdout: %s' % out)
                print('status: %s' % call_x.wait())
                raise ValueError("_get_ref_phylogeny(): something went wrong")

            # convert from b'' to string
            out = out.decode()
            # split key:\tvalue
            out = out.split('\t')[1]
            # remove trailing \n
            out = out.rstrip()
            # chop '/rep_set/97_otus.fasta' from found path
            out = '/'.join(out.split('/')[:-2])
            FILE_REFERENCE_TREE = out + '/trees/97_otus.tree'
    return FILE_REFERENCE_TREE


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
    """Computes alpha diversity values for given BIOM table.

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
        _get_ref_phylogeny(),
        num_threads))

    sys.stderr.write("Working directory is '%s'. " % workdir)

    if not use_grid:
        if dry:
            sys.stderr.write("\n\n".join(commands))
            return None
        with subprocess.Popen("source activate %s && %s" %
                              (QIIME_ENV, " && ".join(commands)),
                              shell=True,
                              stdout=subprocess.PIPE,
                              executable="bash") as call_x:
            if (call_x.wait() != 0):
                raise ValueError("something went wrong")
    else:
        cluster_run(commands, 'ana_alpha', workdir+'mock', QIIME_ENV,
                    ppn=num_threads, wait=True, dry=dry)

    results = _parse_alpha(num_iterations, workdir+'/alpha_div/',
                           rarefaction_depth)

    if results is not None:
        results.index.name = counts.index.name
        shutil.rmtree(workdir)
        sys.stderr.write("Was removed.\n")

    return results


def beta_diversity(counts, metrics, dry=True, use_grid=True):
    """Computes beta diversity values for given BIOM table.

    Parameters
    ----------
    counts : Pandas.DataFrame
        OTU counts
    metrics : [str]
        Beta diversity metrics to be computed.
    dry : boolean
        Do NOT run clusterjobs, just print commands. Default: True
    use_grid : boolean
        Use grid engine instead of local execution. Default: True

    Returns
    -------
    Dict of Pandas.DataFrame, one per metric."""

    # create a temporary working directory
    workdir = tempfile.mkdtemp(prefix='ana_beta_')

    # store counts as a biom file
    pandas2biom(workdir+'/input.biom', counts)

    commands = []
    commands.append(('beta_diversity.py '
                     '-i %s '                   # input biom file
                     '-m %s '                   # list of beta div metrics
                     '-t %s '                   # tree reference file
                     '-o %s ') % (
        workdir+'/input.biom',
        ",".join(metrics),
        _get_ref_phylogeny(),
        workdir+'/beta'))

    sys.stderr.write("Working directory is '%s'. " % workdir)

    if not use_grid:
        if dry:
            sys.stderr.write("\n\n".join(commands))
            return None
        with subprocess.Popen("source activate %s && %s" %
                              (QIIME_ENV, " && ".join(commands)),
                              shell=True,
                              stdout=subprocess.PIPE,
                              executable="bash") as call_x:
            if (call_x.wait() != 0):
                raise ValueError("something went wrong")
    else:
        cluster_run(commands, 'ana_beta', workdir+'mock', QIIME_ENV,
                    ppn=1, wait=True, dry=dry)

    results = dict()
    for metric in metrics:
        results[metric] = DistanceMatrix.read('%s/%s_input.txt' % (
            workdir+'/beta',
            metric))

    if results is not None:
        shutil.rmtree(workdir)
        sys.stderr.write("Was removed.\n")

    return results
