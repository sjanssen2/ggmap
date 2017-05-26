# TODO: squb scheint nicht zu warten
# TODO: zu viel Rand bei rare plots

import tempfile
import shutil
import subprocess
import sys
import operator
import hashlib
import os
import pickle
import io
import collections

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

from skbio.stats.distance import DistanceMatrix
import seaborn as sns

from ggmap.snippets import (pandas2biom, cluster_run)


plt.switch_backend('Agg')
plt.rc('font', family='DejaVu Sans')

FILE_REFERENCE_TREE = None
QIIME_ENV = 'qiime_env'


def _zoom(pos, factor):
    """ Zooms in or out of a plt figure. """
    x0 = pos.x0 + pos.width * (1-factor)
    y0 = pos.y0 + pos.height * (1-factor)
    x1 = pos.x1 - pos.width * (1-factor)
    y1 = pos.y1 - pos.height * (1-factor)
    width = x1 - x0
    height = y1 - y0
    return [x0, y0, width, height]


def _get_ref_phylogeny():
    """Use QIIME config to infer location of reference tree."""
    global FILE_REFERENCE_TREE
    if FILE_REFERENCE_TREE is None:
        with subprocess.Popen(("source activate %s && "
                               "print_qiime_config.py "
                               "| grep 'pick_otus_reference_seqs_fp:'" %
                               QIIME_ENV),
                              shell=True,
                              stdout=subprocess.PIPE,
                              executable="bash") as call_x:
            out, err = call_x.communicate()
            if (call_x.wait() != 0):
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


def _plot_collateRarefaction(workdir, metrics, counts, metadata):
    size = 10

    fig = plt.figure(figsize=(metadata.shape[1] * size,
                              (len(metrics)+1) * size))
    gs = gridspec.GridSpec(len(metrics)+1, metadata.shape[1],
                           wspace=0, hspace=0)
    _plot_loosing_curve(counts, plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]))

    # compose one huge chart out of all individual rarefaction plots
    for row, metric in enumerate(metrics):
        for col, field in enumerate(metadata.columns):
            ax = plt.subplot(gs[row+1, col])
            img = mpimg.imread(
                '%s/rare/alpha_rarefaction_plots/average_plots/%s%s.png' %
                (workdir, metric, field))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(img, aspect='auto')

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0)
    return buf


def _plot_loosing_curve(counts, ax1, ax2):
    # compute number of lost / remained samples
    reads_per_sample = counts.sum()
    depths = sorted(reads_per_sample.unique())
    x = pd.DataFrame(index=depths,
                     data=[sum(reads_per_sample >= depth) for depth in depths],
                     columns=['remaining'])
    x['lost'] = counts.shape[1] - x['remaining']
    x.index.name = 'readcounts'

    # loosing samples
    ax1.set_position(_zoom(ax1.get_position(), 0.9))
    plt.sca(ax1)
    plt.plot(x.index, x['remaining'], label='remaining')
    plt.plot(x.index, x['lost'], label='lost')
    ax1.set_xlabel("rarefaction depth")
    ax1.set_ylabel("# samples")
    ax1.set_title('How many of the %i samples do we loose?' % counts.shape[1])
    ax1.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ',')))
    lostHalf = abs(x['remaining'] - x['lost'])
    lostHalf = lostHalf[lostHalf == lostHalf.min()].index[0]
    ax1.set_xlim(0, lostHalf * 1.1)
    #p = ax.set_xscale("log", nonposx='clip')

    # read count histogram
    ax2.set_position(_zoom(ax2.get_position(), 0.9))
    plt.sca(ax2)
    sns.distplot(reads_per_sample, kde=False)
    ax2.set_title('Read count distribution across samples')
    ax2.set_xlabel("read counts")
    ax2.set_ylabel("# samples")
    #p = ax.set_xscale("log", nonposx='clip')
    ax2.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ',')))


def _display_image_in_actual_size(filename):
    # from https://stackoverflow.com/questions/28816046/displaying-different-
    #    images-with-actual-size-in-matplotlib-subplot
    dpi = 80
    im_data = plt.imread(filename)
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full
    # figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    return fig


def rarefaction_curves(counts, metadata, metrics, num_steps=20, num_threads=10,
                       dry=True, use_grid=True, nocache=False):
    def pre_execute(workdir, args):
        # store counts as a biom file
        pandas2biom(workdir+'/input.biom', args['counts'])

        # determine those 5 fields in the metadata information that have highes
        # variability
        var = dict()
        for field in args['metadata'].columns:
            var[field] = len(args['metadata'][field].unique())
        top_var_fields = [field[0]
                          for field
                          in sorted(var.items(), key=operator.itemgetter(1),
                                    reverse=True)[:5]]
        # store top 5 variable fields of metadata as csv file
        metatop = args['metadata'].loc[:, top_var_fields]
        metatop.to_csv(workdir+'/input.meta.tsv',
                       sep='\t', index_label='#SampleID')

        # create parameter file
        f = open(workdir+'params.txt', 'w')
        f.write('alpha_diversity:metrics %s\n' % ",".join(args['metrics']))
        f.close()

        return {'metatop': metatop}

    def commands(workdir, ppn, args):
        commands = []
        commands.append(('xvfb-run alpha_rarefaction.py '
                         '-i %s '                # input biom file
                         '-m %s '                # input metadata file
                         '-o %s '                # output directory
                         '-p %s '                # input parameter file
                         '-a '                   # run in parallel
                         '-t %s '                # tree reference file
                         '-O %i '                # number parallel jobs
                         '--min_rare_depth=%i '  # minimal rarefaction depth
                         '--max_rare_depth=%i '  # maximal rarefaction depth
                         '--num_steps=%i') % (   # number steps between min max
            workdir+'/input.biom',
            workdir+'/input.meta.tsv',
            workdir+'/rare/',
            workdir+'params.txt',
            _get_ref_phylogeny(),
            ppn,
            args['counts'].sum().min(),
            args['counts'].sum().describe()['75%'],
            args['num_steps']))
        return commands

    def post_execute(workdir, args, pre_data):
        return _plot_collateRarefaction(workdir,
                                        args['metrics'],
                                        args['counts'],
                                        pre_data['metatop'])

    imagebuffer = _executor('rare',
                            {'counts': counts,
                             'metadata': metadata,
                             'metrics': metrics,
                             'num_steps': num_steps},
                            pre_execute,
                            commands,
                            post_execute,
                            dry=dry,
                            use_grid=use_grid,
                            ppn=num_threads,
                            nocache=nocache)

    tmp_imagename = 'tmp.png'
    imagebuffer.seek(0)
    f = open(tmp_imagename, 'wb')
    f.write(imagebuffer.read())
    f.close()
    res = _display_image_in_actual_size(tmp_imagename)
    os.remove(tmp_imagename)
    return res


def alpha_diversity(counts, metrics, rarefaction_depth,
                    num_threads=10, num_iterations=10, dry=True,
                    use_grid=True, nocache=False):
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

    def pre_execute(workdir, args):
        # store counts as a biom file
        pandas2biom(workdir+'/input.biom', args['counts'])

        # create a mock metadata file
        metadata = pd.DataFrame(index=args['counts'].columns)
        metadata.index.name = '#SampleID'
        metadata['mock'] = 'foo'
        metadata.to_csv(workdir+'/metadata.tsv', sep='\t')

    def commands(workdir, ppn, args):
        commands = []
        commands.append(('parallel_multiple_rarefactions.py '
                         '-T '                       # direct polling
                         '-i %s '                    # input biom file
                         '-m %i '                    # min rarefaction depth
                         '-x %i '                    # max rarefaction depth
                         '-s 1 '                     # depth steps
                         '-o %s '                    # output directory
                         '-n %i '                 # number iterations per depth
                         '--jobs_to_start %i') % (   # number parallel jobs
            workdir+'/input.biom',
            args['rarefaction_depth'],
            args['rarefaction_depth'],
            workdir+'/rarefactions',
            args['num_iterations'],
            ppn))

        commands.append(('parallel_alpha_diversity.py '
                         '-T '                      # direct polling
                         '-i %s '                   # dir to rarefied tables
                         '-o %s '                   # output directory
                         '--metrics %s '            # list of alpha div metrics
                         '-t %s '                   # tree reference file
                         '--jobs_to_start %i') % (  # number parallel jobs
            workdir+'/rarefactions',
            workdir+'/alpha_div/',
            ",".join(args['metrics']),
            _get_ref_phylogeny(),
            ppn))
        return commands

    def post_execute(workdir, args, pre_data):
        res = _parse_alpha(args['num_iterations'],
                           workdir+'/alpha_div/',
                           args['rarefaction_depth'])
        if res is not None:
            res.index.name = args['counts'].index.name
        return res

    return _executor('adiv',
                     {'counts': counts,
                      'metrics': metrics,
                      'rarefaction_depth': rarefaction_depth,
                      'num_iterations': num_iterations},
                     pre_execute,
                     commands,
                     post_execute,
                     dry=dry,
                     use_grid=use_grid,
                     ppn=num_threads,
                     nocache=nocache)


def beta_diversity(counts, metrics, dry=True, use_grid=True, nocache=False):
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

    def pre_execute(workdir, args):
        # store counts as a biom file
        pandas2biom(workdir+'/input.biom', args['counts'])

    def commands(workdir, ppn, args):
        commands = []
        commands.append(('beta_diversity.py '
                         '-i %s '                   # input biom file
                         '-m %s '                   # list of beta div metrics
                         '-t %s '                   # tree reference file
                         '-o %s ') % (
            workdir+'/input.biom',
            ",".join(args['metrics']),
            _get_ref_phylogeny(),
            workdir+'/beta'))
        return commands

    def post_execute(workdir, args, pre_data):
        results = dict()
        for metric in args['metrics']:
            results[metric] = DistanceMatrix.read('%s/%s_input.txt' % (
                workdir+'/beta',
                metric))
        return results

    return _executor('bdiv',
                     {'counts': counts,
                      'metrics': metrics},
                     pre_execute,
                     commands,
                     post_execute,
                     dry=dry,
                     use_grid=use_grid,
                     ppn=1,
                     nocache=nocache)


def _executor(jobname, cache_arguments, pre_execute, commands, post_execute,
              dry=True, use_grid=True, ppn=10, nocache=False):
    # caching
    _input = collections.OrderedDict(sorted(cache_arguments.items()))
    file_cache = ".anacache/%s.%s" % (hashlib.md5(
        str(_input).encode()).hexdigest(), jobname)

    if os.path.exists(file_cache) and (nocache is not True):
        sys.stderr.write("Using existing results from '%s'. " % file_cache)
        f = open(file_cache, 'rb')
        results = pickle.load(f)
        f.close()
        return results

    # create a temporary working directory
    prefix = 'ana_%s_' % jobname
    workdir = None
    if use_grid:
        workdir = tempfile.mkdtemp(prefix=prefix, dir='/home/sjanssen/TMP/')
    else:
        workdir = tempfile.mkdtemp(prefix=prefix)
    sys.stderr.write("Working directory is '%s'. " % workdir)

    pre_data = pre_execute(workdir, cache_arguments)

    lst_commands = commands(workdir, ppn, cache_arguments)
    if not use_grid:
        if dry:
            sys.stderr.write("\n\n".join(lst_commands))
            return None
        with subprocess.Popen("source activate %s && %s" %
                              (QIIME_ENV, " && ".join(lst_commands)),
                              shell=True,
                              stdout=subprocess.PIPE,
                              executable="bash") as call_x:
            if (call_x.wait() != 0):
                raise ValueError("something went wrong")
    else:
        cluster_run(lst_commands, 'ana_%s' % jobname, workdir+'mock',
                    QIIME_ENV, ppn=ppn, wait=True, dry=dry)
        if dry:
            return None

    results = post_execute(workdir, cache_arguments, pre_data)

    if results is not None:
        shutil.rmtree(workdir)
        sys.stderr.write("Was removed.\n")

    os.makedirs(os.path.dirname(file_cache), exist_ok=True)
    f = open(file_cache, 'wb')
    pickle.dump(results, f)
    f.close()

    return results
