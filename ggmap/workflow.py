from os.path import exists
import sys
import requests

from skbio.tree import TreeNode
import pandas as pd
from tempfile import mkstemp
from glob import glob

from ggmap.snippets import *
from ggmap.analyses import *
from ggmap.correlations import *


JLU_PROXY = 'http://proxy.computational.bio.uni-giessen.de:3128'

def init_project(pi, name, prj_data, project_dir_prefix='/vol/jlab/MicrobiomeAnalyses/Projects/', verbose=sys.stderr, force=False):
    if (pi is None) or (pi.strip() == ""):
        raise ValueError("Abort: No Principle Investigator last name given!")
    if (name is None) or (name.strip() == ""):
        raise ValueError("Abort: No project name is given!")

    for text in [(pi, 'pi'), (name, 'name')]:
        if text[0][0].upper() != text[0][0]:
            raise ValueError("Abort: Please let '%s' start with a capital letter." % text[1])

    fp_path_project = os.path.join(project_dir_prefix, '%s_%s' % (pi[0].upper() + pi[1:], name[0].upper() + name[1:]))
    prj_data['paths'] = dict()
    if force is False:
        if os.path.exists(fp_path_project):
            raise ValueError("Abort: Project directory '%s' already exists! If you are sure to continue with this directory switch parameter 'force' to True!" % fp_path_project)

        if verbose:
            print('Creating project directory structure:', file=verbose)
    for subdir in ['', 'Incoming', 'Outgoing', 'FromQiita', 'tmp_workdir', os.path.join('Generated', 'Emperor'), 'Figures']:
        fp_subdir = os.path.join(fp_path_project, subdir)
        prj_data['paths']['root' if subdir == '' else subdir] = os.path.abspath(fp_subdir)
        if force is False:
            os.makedirs(fp_subdir, exist_ok=True)
            print('  %s' % fp_subdir, file=verbose)

    # set proxy to reach internet within slurm@lummerland cluster
    for protocol in ['ftp','http','https']:
        os.environ["%s_proxy" % protocol] = JLU_PROXY

    prj_data['git_name'] = 'microbiome_%s_%s' % (name.lower(), pi.lower())
    if not os.path.exists(os.path.join(prj_data['paths']['root'], '.git')):
        # initiate git repository
        with subprocess.Popen(
            ["cd %s && git init --shared=group" % fp_path_project], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable='/bin/bash') as task:
            res = task.stdout.read().decode('ascii')
        # add remote to git repo
        with subprocess.Popen(
            ["cd %s && git remote add origin 'https://github.com/jlab/%s.git'" % (fp_path_project, prj_data['git_name'])], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable='/bin/bash') as task:
            res = task.stdout.read().decode('ascii')
    else:
        if force is False:
            raise ValueError("Git repository already initialized. If you want to continue, switch 'force' to True.")
    # check if project repo already exists in jlab
    r = requests.get('https://github.com/jlab/%s' % prj_data['git_name'], proxies={'https': JLU_PROXY})
    if force is False:
        if r.status_code != 404:
            raise ValueError("Git repo '%s' already present in github.com/jlab. You should use another project name! If you want to continue with this repo, switch 'force' to True" % prj_data['git_name'])
        print("Initiated Git repo and added github.com/jlab as remote 'origin'. Ask Stefan to actually create repo 'github.com/jlab%s'!" % prj_data['git_name'])

    return prj_data

def project_demux(fp_illuminadata, fp_demuxsheet, prj_data, force=False, ppn=10, verbose=sys.stderr):
    # peek into demux sheet
    with open(fp_demuxsheet, 'r') as f:
        firstlines = f.readlines()[:2]
        if not force:
            if '[Header]' not in firstlines[0] or 'IEMFileVersion' not in firstlines[1]:
                raise ValueError("Your file '%s' does not look like a typical Illumina demultiplexing sheet. Please double check. If you are certain it is correct, switch parameter 'force' to True." % fp_demuxsheet)

    # peek into Illumina data directory
    if not force:
        if not os.path.exists(os.path.join(fp_illuminadata, 'RunInfo.xml')) or not os.path.exists(os.path.join(fp_illuminadata, 'Data', 'Intensities', 'BaseCalls')):
            raise ValueError("Your path '%s' does not look like a typical Illumina raw data directory. Please double check. If you are certain it is correct, switch parameter 'force' to True.")

    prj_data['paths']['demux'] = os.path.join(prj_data['paths']['tmp_workdir'], 'demultiplex')
    os.makedirs(prj_data['paths']['demux'], exist_ok=True)
    prj_data['paths']['illumina_rawdata'] = os.path.abspath(fp_illuminadata)
    prj_data['paths']['illumina_demuxsheet'] = os.path.abspath(fp_demuxsheet)

    if verbose:
        _, fp_tmp = tempfile.mkstemp()
        cluster_run(['bcl2fastq --version > %s 2>&1' % fp_tmp], 'info', '/dev/null/kurt', environment='spike', dry=False, use_grid=False)
        with open(fp_tmp, 'r') as f:
            print('using version: %s' % f.readlines()[1].strip())

    print('Since bcl2fastq opens a LOT of file handles, I recommand you copy and paste the below command and ssh into machine `hp-dl980.internal.computational.bio.uni-giessen.de` where you execute the command:\n')
    cluster_run(["bcl2fastq --runfolder-dir %s --output-dir %s --ignore-missing-bcls --sample-sheet %s --loading-threads %i --processing-threads %i --writing-threads %i" % (
        prj_data['paths']['illumina_rawdata'],
        prj_data['paths']['demux'],
        prj_data['paths']['illumina_demuxsheet'],
        ppn, ppn, ppn)], "demux", prj_data['paths']['demux']+"/Undetermined_S0_L001_R1_001.fastq.gz", environment="spike", ppn=ppn,
        use_grid=False, dry=True)

    return prj_data

def project_trimprimers(primerseq_fwd, primerseq_rev, prj_data, force=False, verbose=sys.stderr):
    knownprimer = {
        'GTGCCAGCMGCCGCGGTAA': {
            'gene': '16s',
            'region': 'V4',
            'orientation': 'fwd',
            'position': '515f',
            'reference': 'Caporaso et al.',
            'doi': '10.1073/pnas.1000080107'},
        'GGACTACHVGGGTWTCTAAT': {
            'gene': '16s',
            'region': 'V4',
            'orientation': 'rev',
            'position': '806r',
            'reference': 'Caporaso et al.',
            'doi': '10.1073/pnas.1000080107'},

        'GTGYCAGCMGCCGCGGTAA': {
            'gene': '16s',
            'region': 'V4',
            'orientation': 'rev',
            'position': '806r',
            'reference': 'Parada et al.',
            'doi': '10.1111/1462-2920.13023 '},
        'GGACTACNVGGGTWTCTAAT': {
            'gene': '16s',
            'region': 'V4',
            'orientation': 'rev',
            'position': '806r',
            'reference': 'Apprill et al.',
            'doi': '10.3354/ame01753'},
    }

    if primerseq_fwd.upper() not in knownprimer.keys():
        print("Forward primer sequence unknown.")
    if primerseq_rev.upper() not in knownprimer.keys():
        print("Reverse primer sequence unknown.")

    prj_data['paths']['trimmed'] = os.path.join(prj_data['paths']['tmp_workdir'], 'trimmed')
    os.makedirs(prj_data['paths']['trimmed'], exist_ok=True)

    if verbose:
        _, fp_tmp = tempfile.mkstemp()
        cluster_run(['cutadapt --version > %s 2>&1' % fp_tmp], 'info', '/dev/null/kurt', environment='spike', dry=False, use_grid=False)
        with open(fp_tmp, 'r') as f:
            print('using cutadapt version: %s' % f.readlines()[0].strip())

    for fp_fastq in glob(os.path.join(prj_data['paths']['demux'],"*_R1_001.fastq.gz")):
        if "Undetermined_S0_" in fp_fastq:
            continue
        fp_in_r1 = os.path.abspath(fp_fastq)
        fp_out_r1 = os.path.join(prj_data['paths']['trimmed'], os.path.basename(fp_in_r1))
        fp_out_r2 = fp_out_r1.replace("_R1_", "_R2_")
        cluster_run(["cutadapt -g %s -G %s -n 2 -o %s -p %s %s %s" % (primerseq_fwd, primerseq_rev,
             fp_out_r1, fp_out_r2, fp_in_r1, fp_in_r1)], "trimming", fp_out_r1, environment="spike", ppn=1, dry=False, use_grid=True)

    return prj_data

def project_deblur(prj_data, trimlength=150, ppn=4):
    prj_data['paths']['deblur'] = os.path.join(prj_data['paths']['tmp_workdir'], 'deblur')

    _, fp_tmp = tempfile.mkstemp()
    cluster_run(['deblur --version > %s 2>&1' % fp_tmp], 'info', '/dev/null/kurt', environment=settings.QIIME2_ENV, dry=False, use_grid=False)
    with open(fp_tmp, 'r') as f:
        print('using version: %s' % f.readlines()[-1].strip())

    # create temp dir
    os.makedirs('%s/inputs' % prj_data['paths']['deblur'], exist_ok=True)
    os.makedirs('%s/deblur_res' % prj_data['paths']['deblur'], exist_ok=True)

    cmds = []

    # link input fastq files, but only fwd
    cmds.append('for f in `find %s -type f -name "*_R1_001.fastq.gz"`; do ln -s $f %s/inputs/; done' % (prj_data['paths']['trimmed'], prj_data['paths']['deblur']))

    # deblur
    cmds.append('deblur workflow --seqs-fp %s/inputs --output-dir %s/deblur_res --trim-length %i --jobs-to-start %i --keep-tmp-files --overwrite ' % (
        prj_data['paths']['deblur'],
        prj_data['paths']['deblur'],
        trimlength,
        ppn,
    ))
    prj_data['paths']['deblur_table'] = os.path.join(prj_data['paths']['deblur'], 'deblur_res', 'reference-hit.biom')
    cluster_run(cmds, 'deblur', prj_data['paths']['deblur_table'], environment=settings.QIIME2_ENV, dry=False, ppn=ppn)

    return prj_data

def project_sepp(prj_data, ppn=8, verbose=sys.stderr):
    # execute fragment insertion
    res_sepp = sepp(biom2pandas(prj_data['paths']['deblur_table']), ppn=ppn, dry=False, environment=settings.QIIME2_ENV)
    print('SEPP version number: %s' % ', '.join([line.split()[1] for line in res_sepp['conda_list'] if line.startswith('sepp') or line.startswith('q2-fragment-insertion')]))
    fp_tree = os.path.join(prj_data['paths']['tmp_workdir'], 'sepp_uncorrected_tree.tmp')
    if not os.path.exists(fp_tree):
        with open(fp_tree, 'w') as f:
            f.write(res_sepp['results']['tree'])

    # correct zero branch length
    prj_data['paths']['insertion_tree'] = os.path.join(prj_data['paths']['tmp_workdir'], 'insertion_tree.nwk')
    if not os.path.exists(prj_data['paths']['insertion_tree']):
        writeReferenceTree(fp_tree, prj_data['paths']['tmp_workdir'], fix_zero_len_branches=True)
        os.rename(os.path.join(prj_data['paths']['tmp_workdir'], 'reference.tree'), prj_data['paths']['insertion_tree'])

    # load the tree as skbio.TreeNode object into memory: might take ~8 minutes
    if ('insertion_tree' not in prj_data) or (prj_data['insertion_tree'] is None):
        print('loading tree. Be patient. Can take up around 8 minutes.', file=sys.stderr)
        prj_data['insertion_tree'] = TreeNode.read(prj_data['paths']['insertion_tree'])

    return prj_data

# TODO: find a way to define PPN, but use default if None, e.g. to limit ppn for beta div computation, due to known error if ppn > sample number
def process_study(metadata: pd.DataFrame,
                  control_samples: {str},
                  fp_deblur_biom: str,
                  fp_insertiontree: str,
                  fp_closedref_biom: str=None,
                  rarefaction_depth=None,
                  rarefaction_max_depth=None,
                  rarefaction_sample_grouping:pd.Series=None,
                  fp_taxonomy_trained_classifier: str='/home/sjanssen/GreenGenes/gg-13-8-99-515-806-nb-classifier.qza',
                  tree_insert: TreeNode=None,
                  verbose=sys.stderr,
                  is_v4_region: bool=True,
                  fix_zero_len_branches: bool=False,
                  dry: bool=True,
                  use_grid: bool=True,
                  ppn: int=8,
                  pmem=None,
                  emperor_infix: str="",
                  emperor_fp: str=None,
                  alpha_metrics=["PD_whole_tree", "shannon", "observed_features"],
                  beta_metrics=["unweighted_unifrac", "weighted_unifrac", "bray_curtis"],
                  deblur_remove_features_lessthanXreads: int=10):
    """
    parameters
    ----------
    fp_closedref_biom : str
        Default: None, i.e. no metagenomic predictions will be computed.
        Filepath to a feature-table, produced by closed reference picking.
        Used for PICRUSt1, Bugbase (future work) predictions.
    deblur_remove_features_lessthanXreads : int
        For Deblur only: As a first step, filter out features with less than X
        reads in all samples combined.
    """
    dupl_meta_cols = pd.Series(metadata.columns).value_counts()
    if dupl_meta_cols.max() > 1:
        raise ValueError("Your metadata contains duplicate columns:\n%s" % dupl_meta_cols[dupl_meta_cols > 1])

    for (_type, fp) in [('Deblur table', fp_deblur_biom),
                        ('Insertion tree', fp_insertiontree),
                        ('ClosedRef table', fp_closedref_biom),
                        ('Naive bayes classifier', fp_taxonomy_trained_classifier)]:
        if (fp is not None) and (not exists(fp)):
            raise ValueError('The given file path "%s" for the %s does not exist!' % (fp, _type))

    # load deblur biom table
    counts = biom2pandas(fp_deblur_biom).fillna(0)
    if pd.Series(counts.columns).value_counts().max() > 1:
        raise ValueError("Your Deblur biom table has at least one sample duplicate!")

    if deblur_remove_features_lessthanXreads > 0:
        num_features_original = counts.shape[0]
        counts = counts[counts.sum(axis=1) >= deblur_remove_features_lessthanXreads]
        if num_features_original - counts.shape[0] > 0:
            verbose.write('Information: %i of %i features have been removed from your Deblur table, since they have less than %i read counts in all samples combined.\n' % (num_features_original - counts.shape[0], num_features_original, deblur_remove_features_lessthanXreads))
        else:
            verbose.write('Information: no features had less than %i reads.\n' % deblur_remove_features_lessthanXreads)

    # check if most deblur features are actually starting with TACG, as expected from V4 regions
    if is_v4_region and (pd.Series(list(map(lambda x: x[:4], counts.index))).value_counts().idxmax() != 'TACG'):
        primer_forward="GTGYCAGCMGCCGCGGTAA"
        primer_reverse="GGACTACNVGGGTWTCTAAT"
        text = 'are defaults'
        if ('pcr_primers' in metadata) and (metadata['pcr_primers'].dropna().unique().shape[0] == 1):
            for ppart in metadata['pcr_primers'].dropna().unique()[0].split('; '):
                if ppart.upper().startswith('FWD:'):
                    primer_forward = ppart.split(':')[-1].strip()
                elif ppart.upper().startswith('REV:'):
                    primer_reverse = ppart.split(':')[-1].strip()
            text = 'are read from metadata[\'pcr_primers\']'
        verbose.write((
            'Warning: most abundant prefix of features is NOT "TACG".\n'
            'If you are targetting EMP V4 region, this might point to primer removal issues!\n'
            'You might want to trim your raw reads prior to Qiita upload via:\n'
            '"cutadapt -g %s -G %s -n 2 -o {output.forward} -p {output.reverse} {input.forward} {input.forward}"\n'
            '(p.s. primer suggestions %s)\n') % (primer_forward, primer_reverse, text))

    # compute taxonomic lineages for feature sequences
    res_taxonomy = taxonomy_RDP(counts, fp_taxonomy_trained_classifier, dry=dry, wait=True, use_grid=use_grid, ppn=ppn)
    idx_chloroplast_mitochondria = res_taxonomy['results'][res_taxonomy['results']['Taxon'].apply(lambda lineage: 'c__Chloroplast' in lineage or 'f__mitochondria' in lineage)]['Taxon'].index

    if type(control_samples) != set:
        raise ValueError('control samples need to be provided as a SET, not as %s.' % type(control_samples))
    plant_ratio = counts.loc[set(counts.index) - set(idx_chloroplast_mitochondria), set(counts.columns) - control_samples].sum(axis=0) / counts.loc[:, set(counts.columns) - control_samples].sum(axis=0)
    if plant_ratio.min() < 0.95:
        verbose.write('Information: You are loosing a significant amount of reads due to filtration of plant material!\n%s\n' % (1-plant_ratio).sort_values(ascending=False).iloc[:10])

    if (tree_insert is None) and (fp_insertiontree is not None):
        tree_insert = TreeNode.read(fp_insertiontree)
    # collect tips actually inserted into tree
    if (tree_insert is not None):
        features_inserted = {node.name for node in tree_insert.tips()}
    else:
        # default to all features of the counts table, if no insertion tree has been provided
        features_inserted = set(counts.index)

    # remove features assigned taxonomy to chloroplasts / mitochondria,
    # report min, max removal
    # remove features not inserted into tree
    results = dict()
    results['counts_plantsStillIn'] = counts
    counts = counts.loc[sorted((set(counts.index) - set(idx_chloroplast_mitochondria)) & features_inserted), sorted(counts.columns)]

    results['taxonomy'] = {'RDP': res_taxonomy}
    results['counts_plantsremoved'] = counts

    #return results
    # run: rarefaction curves
    results['rarefaction_curves'] = rarefaction_curves(counts, reference_tree=fp_insertiontree, control_sample_names=control_samples, sample_grouping=rarefaction_sample_grouping, max_depth=rarefaction_max_depth, dry=dry, wait=False, use_grid=use_grid, fix_zero_len_branches=fix_zero_len_branches,
        pmem=pmem, metrics=alpha_metrics)
    if rarefaction_depth is None:
        return results

    # run: rarefy counts 1x
    results['rarefaction'] = rarefy(counts, rarefaction_depth=rarefaction_depth, dry=dry, wait=True, use_grid=use_grid)

    # run: alpha diversity
    results['alpha_diversity'] = alpha_diversity(counts, rarefaction_depth=rarefaction_depth, reference_tree=fp_insertiontree, dry=dry, metrics=alpha_metrics, wait=False, use_grid=use_grid, fix_zero_len_branches=fix_zero_len_branches)

    # run: beta diversity
    if results['rarefaction']['results'] is not None:
        results['beta_diversity'] = beta_diversity(results['rarefaction']['results'].fillna(0), reference_tree=fp_insertiontree, dry=dry, wait=False, use_grid=use_grid, fix_zero_len_branches=fix_zero_len_branches, metrics=beta_metrics)
    else:
        raise ValueError("Be patient and wait/poll for rarefaction results!")

    # run: emperor plot
    if results['beta_diversity']['results'] is not None:
        results['emperor'] = emperor(metadata, results['beta_diversity']['results'], './' if emperor_fp is None else emperor_fp, infix=emperor_infix, dry=dry, wait=False, use_grid=use_grid, walltime='00:20:00', pmem='8GB', run_tsne_umap=True)
    else:
        raise ValueError("Be patient and wait/poll for beta diversity results!")

    # run: picrust
    if fp_closedref_biom is not None:
        counts_closedref = biom2pandas(fp_closedref_biom)
        if pd.Series(counts_closedref.columns).value_counts().max() > 1:
            raise ValueError("Your ClosedRef biom table has at least one sample duplicate!")
        results['picrust'] = dict()
        results['picrust']['counts'] = picrust(counts_closedref, dry=dry, wait=False, use_grid=use_grid)
        if results['picrust']['counts']['results'] is not None:
            results['picrust']['diversity'] = dict()
            for db in results['picrust']['counts']['results'].keys():
                results['picrust']['diversity'][db] = dict()
                for level in results['picrust']['counts']['results'][db].keys():
                    results['picrust']['diversity'][db][level] = dict()
                    results['picrust']['diversity'][db][level]['alpha'] = alpha_diversity(results['picrust']['counts']['results'][db][level], rarefaction_depth=None, metrics=['shannon', 'observed_features'], dry=dry, wait=False, use_grid=use_grid)
                    results['picrust']['diversity'][db][level]['beta'] = beta_diversity(results['picrust']['counts']['results'][db][level], metrics=['bray_curtis'], dry=dry, wait=False, use_grid=use_grid)
        else:
            raise ValueError("Be patient and wait/poll for picrust results!")

        results['bugbase'] = dict()
        results['bugbase']['counts'] = bugbase(counts_closedref, dry=False, wait=False, use_grid=use_grid)
    return results
