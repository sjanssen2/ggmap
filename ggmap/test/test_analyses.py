from unittest import TestCase, main

import subprocess
import sys
import os
import pandas as pd
from pandas.util.testing import assert_frame_equal
from skbio.util import get_data_path
from skbio.stats.distance import DistanceMatrix

from ggmap.analyses import (alpha_diversity, beta_diversity,
                            rarefaction_curves)


class TreeTests(TestCase):
    def setUp(self):
        self.counts = pd.read_csv(
            get_data_path('analyses/raw_otu_table.csv'),
            sep='\t',
            dtype={'#SampleID': str})
        self.counts.set_index('#SampleID', inplace=True)

        self.meta = pd.read_csv(get_data_path('analyses/meta.tsv'), sep="\t",
                                index_col=0)

        self.metrics_alpha = ['PD_whole_tree', 'shannon']
        self.alpha = pd.read_csv(
            get_data_path('analyses/alpha_20000.csv'),
            sep='\t',
            index_col=0)

        self.metrics_beta = ["unweighted_unifrac", "bray_curtis"]
        self.beta = dict()
        for metric in self.metrics_beta:
            self.beta[metric] = DistanceMatrix.read(
                get_data_path('analyses/beta_%s.dm' % metric))

        self.filename_rare = get_data_path('analyses/rare.png')

    def test_alpha(self):
        obs_alpha = alpha_diversity(
            self.counts,
            self.metrics_alpha,
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

    def test_rare(self):
        DIFF_THRESHOLD = 900

        obs_rare = rarefaction_curves(
            self.counts,
            self.meta,
            self.metrics_alpha,
            dry=False,
            use_grid=False,
            num_steps=5,
            nocache=True
        )

        filename_obs = 'obs_rare.png'
        obs_rare.savefig(filename_obs)

        filename_diff = 'diff_rare.png'
        res = subprocess.check_output(["compare", "-metric", "AE",
                                       "-dissimilarity-threshold", "1",
                                       filename_obs,
                                       self.filename_rare,
                                       filename_diff],
                                      stderr=subprocess.STDOUT)
        if int(res.decode().split('\n')[0]) > DIFF_THRESHOLD:
            sys.stdout.write(
                "Images differ for '%s'. Check differences in %s.\n" %
                ('rare', filename_diff))
            cmd = ('echo "==== start file contents (%s)"; '
                   'cat %s | base64; '
                   'echo "=== end file contents ==="') % (
                filename_diff,
                filename_diff)
            rescmd = subprocess.check_output(
                cmd, shell=True).decode().split('\n')
            for line in rescmd:
                print(line)
        else:
            os.remove(filename_diff)
            os.remove(filename_obs)

        self.assertLessEqual(int(res.decode().split('\n')[0]),
                             DIFF_THRESHOLD)

if __name__ == '__main__':
    main()
