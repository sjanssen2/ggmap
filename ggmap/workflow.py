from os.path import exists
import sys

from skbio.tree import TreeNode
import pandas as pd

from ggmap.snippets import biom2pandas
from ggmap.analyses import *


# TODO: find a way to define PPN, but use default if None, e.g. to limit ppn for beta div computation, due to known error if ppn > sample number
def process_study(metadata: pd.DataFrame,
                  control_samples: {str},
                  fp_deblur_biom: str,
                  fp_insertiontree: str,
                  fp_closedref_biom: str=None,
                  rarefaction_depth=None,
                  fp_taxonomy_trained_classifier: str='/home/sjanssen/GreenGenes/gg-13-8-99-515-806-nb-classifier.qza',
                  tree_insert: TreeNode=None,
                  verbose=sys.stderr,
                  is_v4_region: bool=True,
                  fix_zero_len_branches: bool=False,
                  dry: bool=True,
                  use_grid: bool=True,
                  ppn: int=20,
                  emperor_infix: str="",
                  emperor_fp: str=None,
                  beta_metrics=["unweighted_unifrac", "weighted_unifrac", "bray_curtis"]):
    """
    parameters
    ----------
    fp_closedref_biom : str
        Default: None, i.e. no metagenomic predictions will be computed.
        Filepath to a feature-table, produced by closed reference picking.
        Used for PICRUSt1, Bugbase (future work) predictions.
    """

    for (_type, fp) in [('Deblur table', fp_deblur_biom),
                        ('Insertion tree', fp_insertiontree),
                        ('ClosedRef table', fp_closedref_biom),
                        ('Naive bayes classifier', fp_taxonomy_trained_classifier)]:
        if (fp is not None) and (not exists(fp)):
            raise ValueError('The given file path "%s" for the %s does not exist!' % (fp, _type))

    # load deblur biom table
    counts = biom2pandas(fp_deblur_biom)

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
    res_taxonomy = taxonomy_RDP(counts, fp_taxonomy_trained_classifier, dry=dry, wait=True, use_grid=use_grid)
    idx_chloroplast_mitochondria = res_taxonomy['results'][res_taxonomy['results']['Taxon'].apply(lambda lineage: 'c__Chloroplast' in lineage or 'f__mitochondria' in lineage)]['Taxon'].index

    if type(control_samples) != set:
        raise ValueError('control samples need to be provided as a SET, not as %s.' % type(control_samples))
    plant_ratio = counts.loc[set(counts.index) - set(idx_chloroplast_mitochondria), set(counts.columns) - control_samples].sum(axis=0) / counts.loc[:, set(counts.columns) - control_samples].sum(axis=0)
    if plant_ratio.min() < 0.95:
        verbose.write('Information: You are loosing a significant amount of reads due to filtration of plant material!\n')

    if tree_insert is None:
        tree_insert = TreeNode.read(fp_insertiontree)
    # collect tips actually inserted into tree
    features_inserted = {node.name for node in tree_insert.tips()}

    # remove features assigned taxonomy to chloroplasts / mitochondria,
    # report min, max removal
    # remove features not inserted into tree
    counts = counts.loc[sorted((set(counts.index) - set(idx_chloroplast_mitochondria)) & features_inserted), sorted(counts.columns)]

    results = dict()
    results['taxonomy'] = {'RDP': res_taxonomy}
    results['counts_plantsremoved'] = counts

    #return results
    # run: rarefaction curves
    results['rarefaction_curves'] = rarefaction_curves(counts, reference_tree=fp_insertiontree, control_sample_names=control_samples, dry=dry, wait=False, use_grid=use_grid, fix_zero_len_branches=fix_zero_len_branches)
    if rarefaction_depth is None:
        return results

    # run: rarefy counts 1x
    results['rarefaction'] = rarefy(counts, rarefaction_depth=rarefaction_depth, dry=dry, wait=True, use_grid=use_grid)

    # run: alpha diversity
    results['alpha_diversity'] = alpha_diversity(counts, rarefaction_depth=rarefaction_depth, reference_tree=fp_insertiontree, dry=dry, wait=False, use_grid=use_grid, fix_zero_len_branches=fix_zero_len_branches)

    # run: beta diversity
    if results['rarefaction']['results'] is not None:
        results['beta_diversity'] = beta_diversity(results['rarefaction']['results'].fillna(0), reference_tree=fp_insertiontree, dry=dry, wait=False, use_grid=use_grid, fix_zero_len_branches=fix_zero_len_branches, metrics=beta_metrics)
    else:
        raise ValueError("Be patient and wait/poll for rarefaction results!")

    # run: emperor plot
    if results['beta_diversity']['results'] is not None:
        results['emperor'] = emperor(metadata, results['beta_diversity']['results'], './' if emperor_fp is None else emperor_fp, infix=emperor_infix, dry=dry, wait=False, use_grid=use_grid, walltime='00:20:00', pmem='8GB')
    else:
        raise ValueError("Be patient and wait/poll for beta diversity results!")

    # run: picrust
    if fp_closedref_biom is not None:
        results['picrust'] = dict()
        results['picrust']['counts'] = picrust(biom2pandas(fp_closedref_biom), dry=dry, wait=False, use_grid=use_grid)
        if results['picrust']['counts']['results'] is not None:
            results['picrust']['diversity'] = dict()
            for db in results['picrust']['counts']['results'].keys():
                results['picrust']['diversity'][db] = dict()
                for level in results['picrust']['counts']['results'][db].keys():
                    results['picrust']['diversity'][db][level] = dict()
                    results['picrust']['diversity'][db][level]['alpha'] = alpha_diversity(results['picrust']['counts']['results'][db][level], rarefaction_depth=None, metrics=['shannon', 'observed_otus'], dry=dry, wait=False, use_grid=use_grid)
                    results['picrust']['diversity'][db][level]['beta'] = beta_diversity(results['picrust']['counts']['results'][db][level], metrics=['bray_curtis'], dry=dry, wait=False, use_grid=use_grid)
        else:
            raise ValueError("Be patient and wait/poll for picrust results!")

    #res_bugbase = bugbase(biom2pandas('FromQiita/67822.otu_table.biom'), dry=False, wait=True, use_grid=False)
    return results
