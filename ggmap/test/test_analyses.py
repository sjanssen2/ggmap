from unittest import TestCase, main

import pandas as pd
from pandas.util.testing import assert_frame_equal
from skbio.util import get_data_path
from skbio.stats.distance import DistanceMatrix

from ggmap.analyses import (alpha_diversity, beta_diversity)


class TreeTests(TestCase):
    def setUp(self):
        self.counts = pd.read_csv(
            get_data_path('analyses/raw_otu_table.csv'),
            sep='\t',
            dtype={'#SampleID': str})
        self.counts.set_index('#SampleID', inplace=True)

        self.alpha = pd.read_csv(
            get_data_path('analyses/alpha_20000.csv'),
            sep='\t',
            index_col=0)

        self.metrics_beta = ["unweighted_unifrac", "bray_curtis"]
        self.beta = dict()
        for metric in self.metrics_beta:
            self.beta[metric] = DistanceMatrix.read(
                get_data_path('analyses/beta_%s.dm' % metric))

    def test_alpha(self):
        obs_alpha = alpha_diversity(
            self.counts,
            ['PD_whole_tree', 'shannon'],
            20000,
            dry=False,
            use_grid=False,
            nocache=True)

        exp_alpha = self.alpha

        # shallow check if rarefaction based alpha div distributions are
        # similar
        assert_frame_equal(obs_alpha.loc[:, exp_alpha.columns].describe(),
                           exp_alpha.describe(),
                           check_less_precise=0)

    def test_beta(self):
        obs_beta = beta_diversity(
            self.counts,
            self.metrics_beta,
            dry=False,
            use_grid=False,
            nocache=True
        )

        for metric in self.metrics_beta:
            self.assertEqual(obs_beta[metric], self.beta[metric])


if __name__ == '__main__':
    main()
