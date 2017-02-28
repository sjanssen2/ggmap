from unittest import TestCase, main
import warnings
import matplotlib.pyplot as plt
import tempfile
import random
import os
import sys
import subprocess

from skbio.util import get_data_path
import pandas as pd

from ggmap.snippets import plotTaxonomy

plt.switch_backend('Agg')
plt.rc('font', family='DejaVu Sans')


def generate_plots(biomfile, metadata, taxonomy, outdir=None, extension='.png',
                   list_existing=False):
    if outdir is None:
        outdir = tempfile.mkdtemp() + "/"
        print("Temdir is %s" % outdir)

    random.seed(1634)
    configs = dict()
    configs['tp_default'] = {
        'description': 'A plain plot with only default behaviour',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'file_taxonomy': taxonomy}}
    configs['tp_reorder'] = {
        'description': ('Check if samples are re-ordered according to most '
                        'abundant taxa.'),
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'reorder_samples': True,
                   'file_taxonomy': taxonomy}}
    configs['tp_samplenames'] = {
        'description': 'Can we plot sample names on the X-axis?',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': True,
                   'file_taxonomy': taxonomy}}
    configs['tp_reorder_samplenames'] = {
        'description': ('Get the sample names on the X-axis re-ordered '
                        'properly?'),
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': True,
                   'reorder_samples': True,
                   'file_taxonomy': taxonomy}}
    configs['tp_groupl1'] = {
        'description': 'Four l1 groups',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l1': 'diet_brief',
                   'file_taxonomy': taxonomy}}
    configs['tp_groupl1_sampellables'] = {
        'description': 'Do we have sample labels, if group l1 is given?',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': True,
                   'reorder_samples': True,
                   'group_l1': 'diet_brief',
                   'file_taxonomy': taxonomy}}
    configs['tp_groupl2'] = {
        'description': 'Can we further subdivide g1 into g2?',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l1': 'diet_brief',
                   'group_l2': 'genspec',
                   'file_taxonomy': taxonomy}}
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
                   'file_taxonomy': taxonomy}}
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
                   'file_taxonomy': taxonomy}}
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
                   'file_taxonomy': taxonomy}}
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
                   'file_taxonomy': taxonomy}}
    configs['tp_onlyg1'] = {
        'description': ('Can we have vertical lines for l2, even if no l1 is'
                        ' defined?'),
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l2': 'genspec',
                   'file_taxonomy': taxonomy}}
    configs['tp_onlyg0'] = {
        'description': 'Can we have rows if no l1 or l2 is defined?',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l0': 'Q2',
                   'file_taxonomy': taxonomy}}
    configs['tp_nog1'] = {
        'description': 'Several rows and vertical lines, but no l1 defined.',
        'params': {'file_otutable': biomfile,
                   'metadata': metadata,
                   'verbose': False,
                   'print_sample_labels': False,
                   'reorder_samples': True,
                   'group_l2': 'genspec',
                   'group_l0': 'Q2',
                   'file_taxonomy': taxonomy}}
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
                   'minreadnr': 5000}}

    if not list_existing:
        sys.stderr.write("Plotting graphs (%i): " % len(configs))
        for name in configs:
            sys.stderr.write(".")
            f = plotTaxonomy(**configs[name]['params'])
            filename = outdir + name + extension
            f[0].savefig(filename)
            configs[name]['imagefile'] = filename
            plt.close(f[0])
        sys.stderr.write(" done.\n")
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

        print("Comparing images (%i): " % len(plots), file=sys.stderr, end="")
        for name in plots:
            print(".", file=sys.stderr, end="")
            res = None
            if (name not in self.plots_baseline) or \
               ('imagefile' not in self.plots_baseline[name]):
                print(("Cannot find baseline plot '%s'. Maybe you need to "
                       "generate baseline plots first. Or check the self."
                       "baselinedir variable.") % name)
            else:
                filename_diff_image = "%s.diff.png" % \
                    self.plots_baseline[name]['imagefile'].split('.')[:-1][0]
                res = subprocess.check_output(["compare", "-metric", "AE",
                                               plots[name]['imagefile'],
                                               self.plots_baseline[name]
                                               ['imagefile'],
                                              filename_diff_image],
                                              stderr=subprocess.STDOUT)
            if res != b'0\n':
                print("Images differ for '%s'. Check differences in %s." %
                      (name, filename_diff_image))
            else:
                os.remove(filename_diff_image)

            self.assertIn(name, self.plots_baseline)
            self.assertIn('imagefile', self.plots_baseline[name])
            self.assertEqual(res, b'0\n')
        print(" OK", file=sys.stderr, end="\n")

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
