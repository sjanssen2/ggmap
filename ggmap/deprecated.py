import io

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

import seaborn as sns

from ggmap.analyses import _getremaining

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
    x = _getremaining(reads_per_sample)
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
    # p = ax.set_xscale("log", nonposx='clip')

    # read count histogram
    ax2.set_position(_zoom(ax2.get_position(), 0.9))
    plt.sca(ax2)
    sns.distplot(reads_per_sample, kde=False)
    ax2.set_title('Read count distribution across samples')
    ax2.set_xlabel("read counts")
    ax2.set_ylabel("# samples")
    # p = ax.set_xscale("log", nonposx='clip')
    ax2.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ',')))


def _display_image_in_actual_size(filename):
    # from https://stackoverflow.com/questions/28816046/displaying-different-
    #    images-with-actual-size-in-matplotlib-subplot
    dpi = 70
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


def sepp_old(counts, chunksize=10000, reference=None, stopdecomposition=None,
             ppn=20, pmem='50GB', walltime='12:00:00',
             **executor_args):
    """Tip insertion of deblur sequences into GreenGenes backbone tree.

    Parameters
    ----------
    counts : Pandas.DataFrame | Pandas.Series
        a) OTU counts in form of a Pandas.DataFrame.
        b) If providing a Pandas.Series, we expect the index to be a fasta
           headers and the colum the fasta sequences.
    reference : str
        Default: None.
        Valid values are ['pynast']. Use a different alignment file for SEPP.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose
    chunksize: int
        Default: 30000
        SEPP jobs seem to fail if too many sequences are submitted per job.
        Therefore, we can split your sequences in chunks of chunksize.

    Returns
    -------
    ???"""
    def pre_execute(workdir, args):
        chunks = range(0, seqs.shape[0], args['chunksize'])
        for chunk, i in enumerate(chunks):
            # write all deblur sequences into one file per chunk
            file_fragments = workdir + '/sequences%s.mfa' % (chunk + 1)
            f = open(file_fragments, 'w')
            chunk_seqs = seqs.iloc[i:i + args['chunksize']]
            for header, sequence in chunk_seqs.items():
                f.write('>%s\n%s\n' % (header, sequence))
            f.close()

    def commands(workdir, ppn, args):
        commands = []
        commands.append('cd %s' % workdir)
        ref = ''
        if args['reference'] is not None:
            ref = ' -r %s' % args['reference']
        sdcomp = ''
        if 'stopdecomposition' in args:
            sdcomp = ' -M %f ' % args['stopdecomposition']
        fp_sepp = '$CONDA_PREFIX/'
        commands.append('%sbin/run-sepp.sh "%s/sequences%s.mfa" res${%s} -x %i %s %s -r %sshare/sepp/ref/RAxML_info-reference-gg-raxml-bl.info -b 1' % (
            fp_sepp,
            workdir,
            '${%s}' % settings.VARNAME_PBSARRAY if len(range(0, seqs.shape[0], chunksize)) > 1 else '1',
            settings.VARNAME_PBSARRAY,
            ppn,
            ref,
            sdcomp,
            fp_sepp))
        return commands

    def post_execute(workdir, args):
        files_placement = sorted(
            [workdir + '/' + file_placement
             for file_placement in next(os.walk(workdir))[2]
             if file_placement.endswith('_placement.json')])
        if len(files_placement) > 1:
            file_mergedplacements = workdir + '/merged_placements.json'
            if not os.path.exists(file_mergedplacements):
                sys.stderr.write("step 1) merging placement files: ")
                fout = open(file_mergedplacements, 'w')
                for i, file_placement in enumerate(files_placement):
                    sys.stderr.write('.')
                    fin = open(file_placement, 'r')
                    write = i == 0
                    for line in fin.readlines():
                        if '"placements": [{' in line:
                            write = True
                            if i != 0:
                                continue
                        if '}],' in line:
                            write = i+1 == len(files_placement)
                        if write is True:
                            fout.write(line)
                    fin.close()
                    if i+1 != len(files_placement):
                        fout.write('    },\n')
                        fout.write('    {\n')
                fout.close()
                sys.stderr.write(' done.\n')

            sys.stderr.write("step 2) placing fragments into tree: ...")
            # guppy ran for: and consumed 45 GB of memory for 2M, chunked 10k
            # sepp benchmark:
            # real	37m39.772s
            # user	31m3.906s
            # sys	3m49.602s
            file_merged_tree = file_mergedplacements[:-5] +\
                '.tog.relabelled.tre'
            cluster_run([
                'cd %s' % workdir,
                '$CONDA_PREFIX/bin/guppy tog %s' %
                file_mergedplacements,
                'cat %s | python %s > %s' % (
                    file_mergedplacements.replace('.json', '.tog.tre'),
                    files_placement[0].replace('placement.json',
                                               'rename-json.py'),
                    file_merged_tree)],
                environment='sepp',
                jobname='guppy_rename',
                result=file_merged_tree,
                ppn=1, pmem='100GB', walltime='1:00:00', dry=False,
                wait=True)
            sys.stderr.write(' done.\n')
        else:
            file_merged_tree = files_placement[0].replace(
                '.json', '.tog.relabelled.tre')

        sys.stderr.write("step 3) reading skbio tree: ...")
        tree = TreeNode.read(file_merged_tree, format='newick')
        sys.stderr.write(' done.\n')

        sys.stderr.write("step 4) use the phylogeny to det"
                         "ermine tips lineage: ")
        lineages = []
        features = []
        divisor = int(tree.count(tips=True) / min(10, tree.count(tips=True)))
        for i, tip in enumerate(tree.tips()):
            if i % divisor == 0:
                sys.stderr.write('.')
            if tip.name.isdigit():
                continue

            lineage = []
            for ancestor in tip.ancestors():
                try:
                    float(ancestor.name)
                except TypeError:
                    pass
                except ValueError:
                    lineage.append(ancestor.name)

            lineages.append("; ".join(reversed(lineage)))
            features.append(tip.name)
        sys.stderr.write(' done.\n')

        # storing tree as newick string is necessary since large trees would
        # result in too many recursions for the python heap :-/
        newick = StringIO()
        tree.write(newick)
        return {'taxonomy': pd.DataFrame(data=lineages,
                                         index=features,
                                         columns=['taxonomy']),
                'tree': newick.getvalue(),
                'reference': args['reference']}

    inp = sorted(counts.index)
    if type(counts) == pd.Series:
        # typically, the input is an OTU table with index holding sequences.
        # However, if provided a Pandas.Series, we expect index are sequence
        # headers and single column holds sequences.
        inp = counts.sort_index()

    def post_cache(cache_results):
        newicks = []
        for tree in cache_results['trees']:
            newicks.append(TreeNode.read(StringIO(tree), format='newick'))
        cache_results['trees'] = newicks
        return cache_results

    seqs = inp
    if type(inp) != pd.Series:
        seqs = pd.Series(inp, index=inp).sort_index()
    args = {'seqs': seqs,
            'reference': reference,
            'chunksize': chunksize}
    if stopdecomposition is not None:
        args['stopdecomposition'] = stopdecomposition
    return _executor('sepp',
                     args,
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=ppn, pmem=pmem, walltime=walltime,
                     array=len(range(0, seqs.shape[0], chunksize)),
                     **executor_args)


def sepp_stepbystep(counts, reference=None,
                    stopdecomposition=None,
                    ppn=20, pmem='8GB', walltime='12:00:00',
                    **executor_args):
    """Step by Step version of SEPP to track memory consumption more closely.
       Tip insertion of deblur sequences into GreenGenes backbone tree.

    Parameters
    ----------
    counts : Pandas.DataFrame | Pandas.Series
        a) OTU counts in form of a Pandas.DataFrame.
        b) If providing a Pandas.Series, we expect the index to be a fasta
           headers and the colum the fasta sequences.
    reference : str
        Default: None.
        Valid values are ['pynast']. Use a different alignment file for SEPP.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose
    chunksize: int
        Default: 30000
        SEPP jobs seem to fail if too many sequences are submitted per job.
        Therefore, we can split your sequences in chunks of chunksize.

    Returns
    -------
    ???"""
    def pre_execute(workdir, args):
        file_fragments = workdir + '/sequences.mfa'
        f = open(file_fragments, 'w')
        for header, sequence in seqs.items():
            f.write('>%s\n%s\n' % (header, sequence))
        f.close()
        os.makedirs(workdir + '/sepp-tempssd/', exist_ok=True)

    def commands(workdir, ppn, args):
        commands = []
        name = 'seppstepbysteprun'
        dir_base = ('/home/sjanssen/miniconda3/envs/seppGG_py3/'
                    'src/sepp-package/')
        dir_tmp = workdir + '/sepp-tempssd/'

        commands.append('cd %s' % workdir)

        commands.append(
            ('python %s -P %i -A %s -t %s -a %s -r %s -f %s -cp '
             '%s/chpoint-%s -o %s -d %s -p %s '
             '1>>%s/sepp-%s-out.log 2>%s/sepp-%s-err.log') % (
                ('%ssepp/run_sepp.py' % dir_base),  # python script of SEPP
                5000,  # problem size for tree
                1000,  # problem size for alignment
                # reference tree file
                ('%sref/reference-gg-raxml-bl-rooted-relabelled.tre' %
                    dir_base),
                # reference alignment file
                ('%sref/gg_13_5_ssu_align_99_pfiltered.fasta' % dir_base),
                # reference info file
                ('%sref/RAxML_info-reference-gg-raxml-bl.info' % dir_base),
                workdir + '/sequences.mfa',  # sequence input file
                dir_tmp,  # tmpdir
                name,
                name,
                workdir,
                dir_tmp,
                workdir,
                name,
                workdir,
                name))

        commands.append(('%s/sepp/tools/bundled/Linux/guppy-64 tog %s/%s_plac'
                         'ement.json') % (dir_base, workdir, name))
        commands.append(('python %s/%s_rename-json.py < %s/%s_placement.tog.t'
                         're > %s/%s_placement.tog.relabelled.tre') %
                        (workdir, name, workdir, name, workdir, name))
        commands.append(('%s/sepp/tools/bundled/Linux/guppy-64 tog --xml %s/%'
                         's_placement.json') % (dir_base, workdir, name))
        commands.append(('python %s/%s_rename-json.py < %s/%s_placement.tog.x'
                         'ml > %s/%s_placement.tog.relabelled.xml') %
                        (workdir, name, workdir, name, workdir, name))

        return commands

    def post_execute(workdir, args):
        file_merged_tree = workdir +\
            '/seppstepbysteprun_placement.tog.relabelled.tre'
        sys.stderr.write("step 1/2) reading skbio tree: ...")
        tree = TreeNode.read(file_merged_tree, format='newick')
        sys.stderr.write(' done.\n')

        sys.stderr.write("step 2/2) use the phylogeny to det"
                         "ermine tips lineage: ")
        lineages = []
        features = []
        divisor = int(tree.count(tips=True) / min(10, tree.count(tips=True)))
        for i, tip in enumerate(tree.tips()):
            if i % divisor == 0:
                sys.stderr.write('.')
            if tip.name.isdigit():
                continue

            lineage = []
            for ancestor in tip.ancestors():
                try:
                    float(ancestor.name)
                except TypeError:
                    pass
                except ValueError:
                    lineage.append(ancestor.name)

            lineages.append("; ".join(reversed(lineage)))
            features.append(tip.name)
        sys.stderr.write(' done.\n')

        # storing tree as newick string is necessary since large trees would
        # result in too many recursions for the python heap :-/
        newick = StringIO()
        tree.write(newick)
        return {'taxonomy': pd.DataFrame(data=lineages,
                                         index=features,
                                         columns=['taxonomy']),
                'tree': newick.getvalue(),
                'reference': args['reference']}

    inp = sorted(counts.index)
    if type(counts) == pd.Series:
        # typically, the input is an OTU table with index holding sequences.
        # However, if provided a Pandas.Series, we expect index are sequence
        # headers and single column holds sequences.
        inp = counts.sort_index()

    seqs = inp
    if type(inp) != pd.Series:
        seqs = pd.Series(inp, index=inp).sort_index()
    return _executor('seppstep',
                     {'seqs': seqs,
                      'reference': reference},
                     pre_execute,
                     commands,
                     post_execute,
                     ppn=ppn, pmem=pmem, walltime=walltime,
                     **executor_args)


def sepp_git(counts,
             ppn=20, pmem='8GB', walltime='12:00:00',
             **executor_args):
    """Latest git version of SEPP.
       Tip insertion of deblur sequences into GreenGenes backbone tree.

    Parameters
    ----------
    counts : Pandas.DataFrame | Pandas.Series
        a) OTU counts in form of a Pandas.DataFrame.
        b) If providing a Pandas.Series, we expect the index to be a fasta
           headers and the colum the fasta sequences.
    executor_args:
        dry, use_grid, nocache, wait, walltime, ppn, pmem, timing, verbose

    Returns
    -------
    ???"""
    def pre_execute(workdir, args):
        file_fragments = workdir + '/sequences.mfa'
        f = open(file_fragments, 'w')
        for header, sequence in seqs.iter():
            f.write('>%s\n%s\n' % (header, sequence))
        f.close()
        os.makedirs(workdir + '/sepp-tempssd/', exist_ok=True)

    def commands(workdir, ppn, args):
        commands = []
        commands.append('cd %s' % workdir)
        commands.append('%srun-sepp.sh "%s" res -x %i' % (
            ('/home/sjanssen/Benchmark_insertiontree/'
             'Software/sepp/sepp-package/'),
            workdir+'/sequences.mfa',
            ppn))
        return commands

    def post_execute(workdir, args):
        file_merged_tree = workdir +\
            '/res_placement.tog.relabelled.tre'
        sys.stderr.write("step 1/2) reading skbio tree: ...")
        tree = TreeNode.read(file_merged_tree, format='newick')
        sys.stderr.write(' done.\n')

        sys.stderr.write("step 2/2) use the phylogeny to det"
                         "ermine tips lineage: ")
        lineages = []
        features = []
        divisor = int(tree.count(tips=True) / min(10, tree.count(tips=True)))
        for i, tip in enumerate(tree.tips()):
            if i % divisor == 0:
                sys.stderr.write('.')
            if tip.name.isdigit():
                continue

            lineage = []
            for ancestor in tip.ancestors():
                try:
                    float(ancestor.name)
                except TypeError:
                    pass
                except ValueError:
                    lineage.append(ancestor.name)

            lineages.append("; ".join(reversed(lineage)))
            features.append(tip.name)
        sys.stderr.write(' done.\n')

        # storing tree as newick string is necessary since large trees would
        # result in too many recursions for the python heap :-/
        newick = StringIO()
        tree.write(newick)
        return {'taxonomy': pd.DataFrame(data=lineages,
                                         index=features,
                                         columns=['taxonomy']),
                'tree': newick.getvalue()}

    inp = sorted(counts.index)
    if type(counts) == pd.Series:
        # typically, the input is an OTU table with index holding sequences.
        # However, if provided a Pandas.Series, we expect index are sequence
        # headers and single column holds sequences.
        inp = counts.sort_index()

    seqs = inp
    if type(inp) != pd.Series:
        seqs = pd.Series(inp, index=inp).sort_index()
    return _executor('seppgit',
                     {'seqs': seqs},
                     pre_execute,
                     commands,
                     post_execute,
                     environment='sepp_git',
                     ppn=ppn, pmem=pmem, walltime=walltime,
                     **executor_args)
