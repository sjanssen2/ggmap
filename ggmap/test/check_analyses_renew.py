from unittest import TestCase, main
import pandas as pd

from skbio.stats.distance import DistanceMatrix, mantel
from skbio.util import get_data_path
from scipy.stats import pearsonr

from ggmap.analyses import (beta_diversity,
                            alpha_diversity,
                            biom2pandas, rarefy, rarefaction_curves)
from ggmap.imgdiff import compare_images


class BetaTests(TestCase):
    def setUp(self):
        self.file_counts = {
            'deblur': get_data_path(
                'analyses/beta_diversity/counts_deblur.biom'),
            'closedref': get_data_path(
                'analyses/beta_diversity/counts_closedref.biom')}
        self.file_reftree_deblur = get_data_path(
            'analyses/beta_diversity/sepp.newick')
        self.metrics = ['bray_curtis', 'unweighted_unifrac',
                        'weighted_unifrac']

        self.true_dms = dict()
        for method in self.file_counts.keys():
            self.true_dms[method] = dict()
            for metric in self.metrics:
                self.true_dms[method][metric] = DistanceMatrix.read(
                    get_data_path('analyses/beta_diversity/beta_%s_%s.dm' %
                                  (method, metric)))

    def test_beta(self):
        for method in self.file_counts.keys():
            reftree = None
            if method == 'deblur':
                reftree = self.file_reftree_deblur
            res_beta = beta_diversity(
                biom2pandas(self.file_counts[method]),
                reference_tree=reftree,
                metrics=self.metrics,
                dry=False,
                wait=True,
                nocache=True)
            for metric in self.metrics:
                sum_test = res_beta['results'][metric].data.max()
                sum_truth = self.true_dms[method][metric].data.max()
                if sum_test == sum_truth == 0:
                    # if both matrices only contain zeros, mantel test reports
                    # nan, thus we cannot use it for testing
                    continue
                (corr, pval, n) = mantel(res_beta['results'][metric],
                                         self.true_dms[method][metric])


class AlphaTests(TestCase):
    def setUp(self):
        self.file_counts = {
            'deblur': get_data_path(
                'analyses/beta_diversity/counts_deblur.biom'),
            'closedref': get_data_path(
                'analyses/beta_diversity/counts_closedref.biom')}
        self.file_reftree_deblur = get_data_path(
            'analyses/beta_diversity/sepp.newick')
        self.metrics = ['PD_whole_tree', 'shannon', 'observed_otus']
        self.rarefaction_depths = [None, 1000]

        self.true_alphas = dict()
        for method in self.file_counts.keys():
            self.true_alphas[method] = dict()
            for rare in self.rarefaction_depths:
                self.true_alphas[method][rare] = pd.read_csv(
                    get_data_path(
                        'analyses/alpha_diversity/alpha_rare%s_i10_%s.tsv' % (
                            rare, method)), sep="\t", index_col=0)

    def test_beta(self):
        for method in self.file_counts.keys():
            reftree = None
            if method == 'deblur':
                reftree = self.file_reftree_deblur
            for rare in self.rarefaction_depths:
                res_alpha = alpha_diversity(
                    biom2pandas(self.file_counts[method]),
                    reference_tree=reftree,
                    rarefaction_depth=rare,
                    metrics=self.metrics,
                    dry=False,
                    wait=True,
                    nocache=True)
                for metric in self.metrics:
                    vals_truth = self.true_alphas[method][rare][metric].values
                    vals_comp = res_alpha['results'][metric].values
                    identical = all(map(lambda x: abs(x[0] - x[1]) < 0.0001,
                                        zip(vals_truth, vals_comp)))
                    corr, pval = pearsonr(vals_truth, vals_comp)
                    self.assertTrue((corr > 0.95) or identical)


class RarefyTests(TestCase):
    def setUp(self):
        self.file_count = get_data_path('25x25.biom')
        self.raredepth = 200

    def test_rarefy(self):
        obs_rarefy = rarefy(
            biom2pandas(self.file_count),
            rarefaction_depth=self.raredepth,
            dry=False, use_grid=False, nocache=True, wait=True)
        self.assertTrue(obs_rarefy['results'].sum().unique() == [200])


class RareCurvesTests(TestCase):
    def setUp(self):
        self.counts = biom2pandas(get_data_path(
            'analyses/rarefaction_curves/qiita10315_150nt_closedref.biom'))
        self.filename_rare = get_data_path(
            'analyses/rarefaction_curves/exp_rare.png')

    def test_rarefaction_curves(self):
        obs_rarefaction_curves = rarefaction_curves(
            self.counts.loc[
                self.counts.sum(axis=1).sort_values(
                    ascending=False).iloc[:50].index,
                :].iloc[:, :10],
            dry=False, use_grid=False, nocache=True, wait=True, dirty=True,
            num_iterations=2, num_steps=5)

        filename_obs = 'obs_rare.png'
        obs_rarefaction_curves['results'].savefig(filename_obs)

        filename_diff = '/tmp/diff_rare.png'
        same, pixdiff = compare_images(filename_obs, self.filename_rare,
                                       threshold=0,
                                       file_image_diff=filename_diff,
                                       name='rarecurves')
        self.assertTrue(same)
        self.assertTrue(pixdiff < 10)


if __name__ == '__main__':
    main()
