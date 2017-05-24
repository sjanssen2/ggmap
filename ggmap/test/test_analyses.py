from unittest import TestCase, main

import pandas as pd
from pandas.util.testing import assert_frame_equal
from skbio.util import get_data_path

from ggmap.analyses import (alpha_diversity)


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

    def test_1(self):
        obs_alpha = alpha_diversity(
            self.counts,
            ['PD_whole_tree', 'shannon'],
            20000,
            dry=False,
            use_grid=False)

        exp_alpha = self.alpha

        # shallow check if rarefaction based alpha div distributions are
        # similar
        assert_frame_equal(obs_alpha.loc[exp_alpha.index, :].describe(),
                           exp_alpha.describe(),
                           check_less_precise=0)

if __name__ == '__main__':
    main()
