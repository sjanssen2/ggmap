from unittest import TestCase, main
import warnings
import matplotlib.pyplot as plt
import tempfile
import random
import os
from os import remove
from io import StringIO
import sys
import numpy as np
from tempfile import mkstemp

from skbio.util import get_data_path
import pandas as pd

from ggmap.snippets import plotTaxonomy, pandas2biom
from ggmap.imgdiff import compare_images

plt.switch_backend('Agg')
plt.rc('font', family='DejaVu Sans')


def generate_plots(biomfile, metadata, taxonomy, outdir=None, extension='.png',
                   list_existing=False):
    if outdir is None:
        outdir = tempfile.mkdtemp(prefix='taxplot_') + "/"
        print("Temdir is %s" % outdir)

    random.seed(1634)
    configs = dict()
    configs['tp_default'] = {
        'description': 'A plain plot with only default behaviour',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'file_taxonomy': taxonomy},
        'threshold': 900}
    configs['tp_reorder'] = {
        'description': ('Check if samples are re-ordered according to most '
                        'abundant taxa.'),
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'reorder_samples': True,
                   'file_taxonomy': taxonomy},
        'threshold': 900}
    configs['tp_samplenames'] = {
        'description': 'Can we plot sample names on the X-axis?',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': True,
                   'file_taxonomy': taxonomy},
        'threshold': 13388}
    configs['tp_reorder_samplenames'] = {
        'description': ('Get the sample names on the X-axis re-ordered '
                        'properly?'),
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': True,
                   'reorder_samples': True,
                   'file_taxonomy': taxonomy},
        'threshold': 14168}
    configs['tp_groupl1'] = {
        'description': 'Four l1 groups',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l1': 'diet_brief',
                   'file_taxonomy': taxonomy},
        'threshold': 900}
    configs['tp_groupl1_sampellables'] = {
        'description': 'Do we have sample labels, if group l1 is given?',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': True,
                   'reorder_samples': True,
                   'group_l1': 'diet_brief',
                   'file_taxonomy': taxonomy},
        'threshold': 13060}
    configs['tp_groupl2'] = {
        'description': 'Can we further subdivide g1 into g2?',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'file_taxonomy': taxonomy},
        'threshold': 1079}
    configs['tp_groupl2_samplelabels'] = {
        'description': ('Do we get the clash with sample labels if l2 group '
                        'is present?'),
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': True,
                   'reorder_samples': True,
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'file_taxonomy': taxonomy},
        'threshold': 12421}
    configs['tp_groupl0'] = {
        'description': 'Also sub-group into Q2 geography.',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'group_l0': 'Q2',
                   'file_taxonomy': taxonomy},
        'threshold': 2366}
    configs['tp_species'] = {
        'description': 'Collapse on Species instead of Phylum',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'group_l0': 'Q2',
                   'rank': 'Species',
                   'file_taxonomy': taxonomy},
        'threshold': 2318}
    configs['tp_minreads'] = {
        'description': 'Stricter read threshold',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'group_l0': 'Q2',
                   'rank': 'Species',
                   'minreadnr': 10000,
                   'file_taxonomy': taxonomy},
        'threshold': 2314}
    configs['tp_onlyg1'] = {
        'description': ('Can we have vertical lines for l2, even if no l1 is'
                        ' defined?'),
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l2': 'genspec',
                   'file_taxonomy': taxonomy},
        'threshold': 900}
    configs['tp_onlyg0'] = {
        'description': 'Can we have rows if no l1 or l2 is defined?',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l0': 'Q2',
                   'file_taxonomy': taxonomy},
        'threshold': 900}
    configs['tp_nog1'] = {
        'description': 'Several rows and vertical lines, but no l1 defined.',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l2': 'genspec',
                   'group_l0': 'Q2',
                   'file_taxonomy': taxonomy},
        'threshold': 1684}
    configs['tp_minAbundance'] = {
        'description': 'There is one more row than expected.',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata[pd.notnull(metadata.Q2)],
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'reorder_samples': True,
                   'print_sample_labels': False,
                   'verbose': False,
                   'file_taxonomy': taxonomy,
                   'minreadnr': 5000},
        'threshold': 900}
    configs['tp_None'] = {
        'description': 'What if nan values are in metadata?',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'reorder_samples': True,
                   'print_sample_labels': False,
                   'verbose': False,
                   'file_taxonomy': taxonomy,
                   'minreadnr': 5000},
        'threshold': 1078}
    configs['tp_taxalist'] = {
        'description': 'Only plot a subset of taxa.',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'group_l1': 'diet_brief',
                   'group_l0': 'Q2',
                   'reorder_samples': True,
                   'print_sample_labels': False,
                   'verbose': False,
                   'file_taxonomy': taxonomy,
                   'plottaxa': ['p__Tenericutes', 'p__Deferribacteres']},
        'threshold': 1568}
    configs['tp_agg_all3'] = {
        'description': 'Aggregate plot on all three groupings',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'group_l0': 'Q2',
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'reorder_samples': True,
                   'print_sample_labels': True,
                   'verbose': False,
                   'file_taxonomy': taxonomy,
                   'fct_aggregate': np.mean},
        'threshold': 2062}
    configs['tp_agg_no1'] = {
        'description': 'Aggregate plot when group l1 is not set',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'group_l0': 'Q2',
                   'group_l2': 'genspec',
                   'reorder_samples': True,
                   'print_sample_labels': True,
                   'verbose': False,
                   'file_taxonomy': taxonomy,
                   'fct_aggregate': np.mean},
        'threshold': 1512}
    configs['tp_agg_no0'] = {
        'description': 'Aggregate plot when group l0 is not set',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'reorder_samples': True,
                   'print_sample_labels': True,
                   'verbose': False,
                   'file_taxonomy': taxonomy,
                   'fct_aggregate': np.mean},
        'threshold': 1443}
    configs['tp_agg_no2'] = {
        'description': 'Aggregate plot when group l2 is not set',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'group_l0': 'Q2',
                   'group_l1': 'diet_brief',
                   'reorder_samples': True,
                   'print_sample_labels': True,
                   'verbose': False,
                   'file_taxonomy': taxonomy,
                   'fct_aggregate': np.mean},
        'threshold': 1869}
    configs['tp_agg_only0'] = {
        'description': 'Aggregate plot when only group l0 is set',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'group_l0': 'Q2',
                   'reorder_samples': True,
                   'print_sample_labels': True,
                   'verbose': False,
                   'file_taxonomy': taxonomy,
                   'fct_aggregate': np.mean},
        'threshold': 900}
    configs['tp_notop_1'] = {
        'description': 'Check that labels above bars are printed; agg all 3.',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'group_l0': 'AGE',
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'no_top_labels': True,
                   'verbose': False,
                   'file_taxonomy': taxonomy,
                   'fct_aggregate': np.mean},
        'threshold': 1191}
    configs['tp_notop_2'] = {
        'description': 'Check that labels above bars are printed; agg 0, 2',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'group_l0': 'AGE',
                   'group_l2': 'genspec',
                   'no_top_labels': True,
                   'verbose': False,
                   'file_taxonomy': taxonomy,
                   'fct_aggregate': np.mean},
        'threshold': 900}
    configs['tp_notop_3'] = {
        'description': 'Check that labels above bars are printed, all 3',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'group_l0': 'AGE',
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'no_top_labels': True,
                   'verbose': False,
                   'file_taxonomy': taxonomy},
        'threshold': 2525}
    configs['tp_rawcounts'] = {
        'description': 'plot raw taxa',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'file_taxonomy': taxonomy,
                   'rank': 'raw',
                   'minreadnr': 5000},
        'threshold': 1700}
    configs['tp_gray_minreads'] = {
        'description': 'Stricter read threshold',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'group_l0': 'Q2',
                   'rank': 'Species',
                   'minreadnr': 10000,
                   'file_taxonomy': taxonomy,
                   'grayscale': True},
        'threshold': 2314}
    configs['tp_gray_rawcounts'] = {
        'description': 'plot raw taxa',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'file_taxonomy': taxonomy,
                   'rank': 'raw',
                   'minreadnr': 5000,
                   'grayscale': True},
        'threshold': 1700}
    configs['tp_taxfrombiom'] = {
        'description': 'use taxonomy from metadata biom table',
        'params': {'file_otutable': get_data_path('taxannot_withtax.biom'),
                   'metadata':
                   pd.read_csv(get_data_path('skin_metadata.tsv'),
                               sep='\t', index_col=0),
                   'verbose': False,
                   'rank': 'Phylum',
                   'file_taxonomy': taxonomy,
                   'taxonomy_from_biom': True},
        'threshold': 310}
    configs['tp_no-n_3groups'] = {
        'description': 'Do not show n=X sample number information',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'rank': 'Phylum',
                   'file_taxonomy': taxonomy,
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'group_l0': 'AGE',
                   'no_sample_numbers': True},
        'threshold': 1471}
    configs['tp_no-n_3groups_agg'] = {
        'description': ('Do not show n=X sample number information, '
                        'for aggregations'),
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'rank': 'Phylum',
                   'file_taxonomy': taxonomy,
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'group_l0': 'AGE',
                   'no_sample_numbers': True,
                   'fct_aggregate': np.mean},
        'threshold': 735}

    # following is stuff to produce taxonomy plots as in Amina's skin study.
    def get_depth(bodysite):
        minreads = 70000
        if bodysite == 'Arm':
            minreads = 65000
        elif bodysite == 'Armpit':
            minreads = 20000
        elif bodysite == 'Face':
            minreads = 42000
        elif bodysite == 'Foot':
            minreads = 26800
        return minreads
    exp_diffs = {'amina_gray_Foot': 8380,
                 'amina_nogray_Foot': 8380,
                 'amina_nogray_Armpit': 6988,
                 'amina_gray_Armpit': 6988,
                 'amina_gray_Arm': 7707,
                 'amina_nogray_Arm': 7707,
                 'amina_gray_Face': 7226,
                 'amina_nogray_Face': 7226}
    meta_amina = pd.read_csv(get_data_path('amina.meta.tsv'),
                             index_col=0, sep='\t')
    for bodysite in sorted(meta_amina['sample_site'].unique()):
        configs['amina_nogray_%s' % bodysite] = {
            'description': ('no low abundant grayscale for %s' % bodysite),
            'params': {'file_otutable': get_data_path('amina.sub10k.biom'),
                       'metadata':
                       meta_amina[meta_amina['sample_site'] == bodysite],
                       'verbose': False,
                       'rank': 'Family',
                       'file_taxonomy': get_data_path('amina.taxonomy.cr.tsv'),
                       'group_l1': 'phase',
                       'group_l0': 'hsid',
                       'no_sample_numbers': True,
                       'fct_aggregate': np.mean,
                       'grayscale': False,
                       'minreadnr': get_depth(bodysite)},
            'threshold': exp_diffs['amina_nogray_%s' % bodysite]}
        configs['amina_gray_%s' % bodysite] = {
            'description': ('with low abundant grayscale for %s' % bodysite),
            'params': {'file_otutable': get_data_path('amina.sub10k.biom'),
                       'metadata':
                       meta_amina[meta_amina['sample_site'] == bodysite],
                       'verbose': False,
                       'rank': 'Family',
                       'file_taxonomy': get_data_path('amina.taxonomy.cr.tsv'),
                       'group_l1': 'phase',
                       'group_l0': 'hsid',
                       'no_sample_numbers': True,
                       'fct_aggregate': np.mean,
                       'min_abundance_grayscale': 0.1,
                       'grayscale': True,
                       'minreadnr': get_depth(bodysite)},
            'threshold':  exp_diffs['amina_gray_%s' % bodysite]}

        # plot mock taxonomy for testing grayscale
        metadata = pd.read_csv(get_data_path('tax_mock_meta.tsv'),
                               index_col=0, sep='\t')
        configs['mock_nogray'] = {
            'description': ('Plotting mock data without gray taxa'),
            'params': {'file_otutable': get_data_path('tax_mock_counts.biom'),
                       'metadata': metadata,
                       'verbose': False,
                       'rank': 'Family',
                       'file_taxonomy': get_data_path('tax_mock_taxonomy.txt'),
                       'group_l1': 'phase',
                       'group_l0': 'hsid',
                       'fct_aggregate': np.mean,
                       'minreadnr': 10000,
                       'grayscale': False},
            'threshold': 2116}
        configs['mock_gray0.2'] = {
            'description': ('Now plotting gray taxa and leave a 20\% gap.'),
            'params': {'file_otutable': get_data_path('tax_mock_counts.biom'),
                       'metadata': metadata,
                       'verbose': False,
                       'rank': 'Family',
                       'file_taxonomy': get_data_path('tax_mock_taxonomy.txt'),
                       'group_l1': 'phase',
                       'group_l0': 'hsid',
                       'fct_aggregate': np.mean,
                       'min_abundance_grayscale': 0.2,
                       'minreadnr': 10000,
                       'grayscale': True},
            'threshold': 2116}
        configs['mock_gray0.01'] = {
            'description': ('Reduce the gap to only 1\%.'),
            'params': {'file_otutable': get_data_path('tax_mock_counts.biom'),
                       'metadata': metadata,
                       'verbose': False,
                       'rank': 'Family',
                       'file_taxonomy': get_data_path('tax_mock_taxonomy.txt'),
                       'group_l1': 'phase',
                       'group_l0': 'hsid',
                       'fct_aggregate': np.mean,
                       'minreadnr': 10000,
                       'min_abundance_grayscale': 0.01,
                       'grayscale': True},
            'threshold': 2116}

    if not list_existing:
        sys.stderr.write("Plotting graphs (%i): " % len(configs))
        sys.stderr.flush()
        for name in configs:
            sys.stderr.write(".")
            sys.stderr.flush()
            f = plotTaxonomy(**configs[name]['params'])
            filename = outdir + name + extension
            f[0].savefig(filename)
            configs[name]['imagefile'] = filename
            plt.close(f[0])
        sys.stderr.write(" done.\n")
        sys.stderr.flush()
    else:
        for name in configs:
            filename = outdir + name + extension
            if os.path.exists(filename):
                configs[name]['imagefile'] = filename

    return configs


class TaxPlotTests(TestCase):
    def setUp(self):
        os.environ["MPLLOCALFREETYPE"] = 'True'

        # suppress the warning caused by biom load_table.
        warnings.simplefilter('ignore', ResourceWarning)
        self.filename_biom = get_data_path('taxplots.biom')
        self.filename_metadata = get_data_path('taxplots.xlsx')
        self.metadata = pd.read_excel(self.filename_metadata, index_col=0)
        self.baselinedir = get_data_path('plot_baseline/')
        self.taxonomy = get_data_path('97_otu_taxonomy.txt')
        genBaseline = False
        self.plots_baseline = generate_plots(self.filename_biom, self.metadata,
                                             self.taxonomy,
                                             outdir=self.baselinedir,
                                             list_existing=not genBaseline)

    def test_regression_plots(self):
        plots = generate_plots(self.filename_biom, self.metadata,
                               self.taxonomy)

        sys.stderr.write("Comparing images (%i): " % len(plots))
        sys.stderr.flush()
        testResults = []
        for name in plots:
            sys.stderr.write(".")
            sys.stderr.flush()
            res = None
            filename_diff_image = "%s.diff.png" % \
                self.plots_baseline[name]['imagefile'].split('.')[:-1][0]
            if (name not in self.plots_baseline) or \
               ('imagefile' not in self.plots_baseline[name]):
                sys.stdout.write(
                    ("Cannot find baseline plot '%s'. Maybe you need to "
                     "generate baseline plots first. Or check the self."
                     "baselinedir variable.") % name)
                sys.stdout.flush()
            else:
                res = compare_images(plots[name]['imagefile'],
                                     self.plots_baseline[name]['imagefile'],
                                     file_image_diff=filename_diff_image,
                                     threshold=plots[name]['threshold']+1,
                                     name=name)
            testResults.append({'name': name,
                                'res': res,
                                'threshold': plots[name]['threshold']})
        sys.stderr.write(" OK")
        sys.stderr.flush()

        for r in testResults:
            self.assertIn(r['name'], self.plots_baseline)
            self.assertIn('imagefile', self.plots_baseline[r['name']])
            self.assertLessEqual(r['res'][1], r['threshold'])

    def test_parameter_checks(self):
        field = 'notThere'
        out = StringIO
        with self.assertRaisesRegex(ValueError,
                                    ('Column "%s" for grouping level %i is '
                                     'not in metadata table!') % (field, 0)):
            f = plotTaxonomy(self.filename_biom, self.metadata, group_l0=field,
                             out=out)
            plt.close(f[0])

        field = 'notThere'
        with self.assertRaisesRegex(ValueError,
                                    ('Column "%s" for grouping level %i is '
                                     'not in metadata table!') % (field, 1)):
            f = plotTaxonomy(self.filename_biom, self.metadata, group_l1=field,
                             out=out)
            plt.close(f[0])

        field = 'notThere'
        with self.assertRaisesRegex(ValueError,
                                    ('Column "%s" for grouping level %i is '
                                     'not in metadata table!') % (field, 2)):
            f = plotTaxonomy(self.filename_biom, self.metadata, group_l2=field,
                             out=out)
            plt.close(f[0])

        with self.assertRaisesRegex(ValueError,
                                    'is not a valid taxonomic rank. Choose'):
            f = plotTaxonomy(self.filename_biom, self.metadata, rank='noRank',
                             out=out)
            plt.close(f[0])

        with self.assertRaisesRegex(IOError, 'Taxonomy file not found!'):
            f = plotTaxonomy(self.filename_biom, self.metadata,
                             file_taxonomy='noFile',
                             out=out)
            plt.close(f[0])

    def test_plotTaxonomy_filenotfound(self):
        out = StringIO
        with self.assertRaisesRegex(IOError,
                                    'OTU table file not found'):
            f = plotTaxonomy(self.filename_biom+'notthere', self.metadata,
                             file_taxonomy=self.taxonomy,
                             out=out)
            plt.close(f[0])

    def test_plotTaxonomy_outreduction(self):
        out = StringIO()
        f = plotTaxonomy(self.filename_biom, self.metadata, out=out,
                         file_taxonomy=self.taxonomy)
        plt.close(f[0])
        self.assertIn('142 samples left with metadata and counts.',
                      out.getvalue())

    def test_plotTaxonomy_collapse(self):
        out = StringIO()
        f = plotTaxonomy(self.filename_biom, self.metadata, out=out,
                         file_taxonomy=self.taxonomy)
        plt.close(f[0])
        self.assertIn('9 taxa left after collapsing to Phylum.',
                      out.getvalue())

    def test_plotTaxonomy_filter(self):
        out = StringIO()
        f = plotTaxonomy(self.filename_biom, self.metadata, out=out,
                         minreadnr=5000, file_taxonomy=self.taxonomy)
        plt.close(f[0])
        self.assertIn('7 taxa left after filtering low abundant.',
                      out.getvalue())

    def test_plotTaxonomy_giventaxa(self):
        out = StringIO()
        f = plotTaxonomy(self.filename_biom, self.metadata, out=out,
                         plottaxa=['p__Actinobacteria', 'p__Bacteroidetes'],
                         file_taxonomy=self.taxonomy)
        plt.close(f[0])
        self.assertIn('2 taxa left after restricting to provided list.',
                      out.getvalue())

    def test_plotTaxonomy_nogrouping(self):
        out = StringIO()
        with self.assertRaisesRegex(ValueError,
                                    ('Cannot aggregate samples, '
                                     'if no grouping is given!')):
            f = plotTaxonomy(self.filename_biom, self.metadata,
                             fct_aggregate=np.mean,
                             file_taxonomy=self.taxonomy, out=out)
            plt.close(f[0])

    def test_plotTaxonomy_report(self):
        out = StringIO()
        f = plotTaxonomy(self.filename_biom, self.metadata, out=out,
                         minreadnr=5000, file_taxonomy=self.taxonomy)
        plt.close(f[0])
        self.assertIn('raw counts: 142', out.getvalue())
        self.assertIn('raw meta: 287', out.getvalue())
        self.assertIn('meta with counts: 142 samples x 5 fields',
                      out.getvalue())
        self.assertIn('counts with meta: 142', out.getvalue())

        out = StringIO()
        f = plotTaxonomy(self.filename_biom, self.metadata, out=out,
                         minreadnr=8000, file_taxonomy=self.taxonomy,
                         min_abundance_grayscale=0.2,
                         grayscale=True)
        plt.close(f[0])
        self.assertIn('saved plotting 1 boxes.', out.getvalue())

    def test_plotTaxonomy_fillmissingranks(self):
        sample_names = ['sampleA', 'sampleB', 'sampleC', 'sampleD']
        taxstrings = ['k__bacteria',
                      'k__bacteria; p__fantasia',
                      'k__bacteria;p__fantasia;g__nona']
        otus = ['otu1', 'otu2', 'otu3']
        lineages = pd.Series(taxstrings, index=otus,
                             name='taxonomy').to_frame()
        file_lin = mkstemp()[1]
        lineages.to_csv(file_lin, sep='\t')

        counts = pd.DataFrame([[1, 2, 3, 4],
                               [5, 6, 7, 8],
                               [9, 10, 11, 12]],
                              index=otus,
                              columns=sample_names)
        file_dummy = mkstemp()[1]
        pandas2biom(file_dummy, counts)
        meta = pd.Series(['a', 'a', 'a', 'b'], index=sample_names,
                         name='dummy').to_frame()

        out = StringIO()
        f, rank_counts, _, vals, _ = plotTaxonomy(file_dummy,
                                                  meta, rank='Species',
                                                  file_taxonomy=file_lin,
                                                  minreadnr=0, out=out)
        plt.close(f)
        self.assertCountEqual(['s__'], rank_counts.index)

        f, rank_counts, _, vals, _ = plotTaxonomy(file_dummy,
                                                  meta, rank='Kingdom',
                                                  file_taxonomy=file_lin,
                                                  minreadnr=0, out=out)
        plt.close(f)
        self.assertCountEqual(['k__bacteria'], rank_counts.index)

        f, rank_counts, _, vals, _ = plotTaxonomy(file_dummy,
                                                  meta, rank='Phylum',
                                                  file_taxonomy=file_lin,
                                                  minreadnr=0, out=out)
        plt.close(f)
        self.assertCountEqual(['p__fantasia', 'p__'], rank_counts.index)

        remove(file_dummy)
        remove(file_lin)


if __name__ == '__main__':
    main()
