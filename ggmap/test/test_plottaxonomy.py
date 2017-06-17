from unittest import TestCase, main
import warnings
import matplotlib.pyplot as plt
import tempfile
import random
import os
import sys
import subprocess
import numpy as np

from skbio.util import get_data_path
import pandas as pd

from ggmap.snippets import plotTaxonomy
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
        'threshold': 1717}
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
        'threshold': 1276}
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
        'threshold': 1156}
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
        'threshold': 1336}
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
        'threshold': 1102}
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
                                     threshold=
                                     self.configs[name]['threshold']+1,
                                     name=name)
            testResults.append({'name': name,
                                'res': res,
                                'threshold': self.configs[name]['threshold']})
        sys.stderr.write(" OK")
        sys.stderr.flush()

        for r in testResults:
            self.assertIn(r['name'], self.plots_baseline)
            self.assertIn('imagefile', self.plots_baseline[r['name']])
            self.assertLessEqual(r['res'][1], r['threshold'])

    def test_parameter_checks(self):
        field = 'notThere'
        with self.assertRaisesRegex(ValueError,
                                    ('Column "%s" for grouping level %i is '
                                     'not in metadata table!') % (field, 0)):
            plotTaxonomy(self.filename_biom, self.metadata, group_l0=field)

        field = 'notThere'
        with self.assertRaisesRegex(ValueError,
                                    ('Column "%s" for grouping level %i is '
                                     'not in metadata table!') % (field, 1)):
            plotTaxonomy(self.filename_biom, self.metadata, group_l1=field)

        field = 'notThere'
        with self.assertRaisesRegex(ValueError,
                                    ('Column "%s" for grouping level %i is '
                                     'not in metadata table!') % (field, 2)):
            plotTaxonomy(self.filename_biom, self.metadata, group_l2=field)

        with self.assertRaisesRegex(ValueError,
                                    'is not a valid taxonomic rank. Choose'):
            plotTaxonomy(self.filename_biom, self.metadata, rank='noRank')

        with self.assertRaisesRegex(IOError, 'Taxonomy file not found!'):
            plotTaxonomy(self.filename_biom, self.metadata,
                         file_taxonomy='noFile')


if __name__ == '__main__':
    main()
