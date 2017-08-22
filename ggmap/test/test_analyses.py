from unittest import TestCase, main
import pandas as pd
from pandas.util.testing import assert_frame_equal

from skbio.stats.distance import DistanceMatrix
from skbio.util import get_data_path

from ggmap.analyses import (_executor, alpha_diversity, beta_diversity,
                            rarefaction_curves)
from ggmap.imgdiff import compare_images


class ExecutorTests(TestCase):
    def test__executor(self):
        def pre_execute(workdir, args):
            file_mapping = workdir + '/headermap.tsv'
            m = open(file_mapping, 'w')
            m.write('pre_execute test\n')
            m.close()

        def commands(workdir, ppn, args):
            commands = ['python --version >> %s 2>&1' %
                        (workdir+'/headermap.tsv')]
            return commands

        def post_execute(workdir, args, pre_data):
            cont = []
            f = open(workdir+'/headermap.tsv', 'r')
            for line in f.readlines():
                cont.append(line)
            f.close()
            return cont

        res = _executor('travistest',
                        {'seqs': 'fake'},
                        pre_execute,
                        commands,
                        post_execute,
                        dry=False, wait=True, use_grid=False, nocache=True,
                        timing=False)
        self.assertIn('Python 2.7.',
                      res['results'][1])
        self.assertNotIn('Python 3.', res['results'][1])


class AlphaTests(TestCase):
    def setUp(self):
        self.counts = pd.read_csv(
            get_data_path('analyses/raw_otu_table.csv'),
            sep='\t',
            dtype={'#SampleID': str})
        self.counts.set_index('#SampleID', inplace=True)

        self.metrics_alpha = ['PD_whole_tree', 'shannon']
        self.alpha = pd.read_csv(
            get_data_path('analyses/alpha_10000.csv'),
            sep='\t',
            index_col=0)
        self.alpha_none = pd.read_csv(
            get_data_path('analyses/alpha_None.csv'),
            sep='\t',
            index_col=0)

    def test_alpha(self):
        obs_alpha = alpha_diversity(
            self.counts,
            10000,
            self.metrics_alpha,
            dry=False,
            use_grid=False,
            nocache=True,
            ppn=1)

        # shallow check if rarefaction based alpha div distributions are
        # similar
        self.assertTrue(all((self.alpha - obs_alpha['results']).max() < 1))

    def test_alpha_norarefaction(self):
        obs_alpha = alpha_diversity(
            self.counts,
            None,
            self.metrics_alpha,
            dry=False,
            use_grid=False,
            nocache=True,
            ppn=1)

        assert_frame_equal(obs_alpha['results'], self.alpha_none)


class BetaTests(TestCase):
    def setUp(self):
        self.counts = pd.read_csv(
            get_data_path('analyses/raw_otu_table.csv'),
            sep='\t',
            dtype={'#SampleID': str})
        self.counts.set_index('#SampleID', inplace=True)

        self.metrics_beta = ["unweighted_unifrac", "bray_curtis"]

        self.beta = dict()
        for metric in self.metrics_beta:
            self.beta[metric] = DistanceMatrix.read(
                get_data_path('analyses/beta_%s.dm.txt' % metric))

    def test_beta(self):
        obs_beta = beta_diversity(
            self.counts,
            self.metrics_beta,
            dry=False,
            use_grid=False,
            nocache=True
        )

        for metric in self.metrics_beta:
            self.assertEqual(
                obs_beta['results'][metric].filter(self.beta[metric].ids),
                self.beta[metric])


class RarefactionTests(TestCase):
    def setUp(self):
        self.counts = pd.read_csv(
            get_data_path('analyses/raw_otu_table.csv'),
            sep='\t',
            dtype={'#SampleID': str})
        self.counts.set_index('#SampleID', inplace=True)

        self.metrics_alpha = ['PD_whole_tree', 'shannon']

        self.filename_rare = get_data_path('analyses/rare.png')

    def test_rare(self):
        DIFF_THRESHOLD = 0

        obs_rare = rarefaction_curves(
            self.counts.iloc[:5, :5],
            self.metrics_alpha,
            dry=False,
            use_grid=False,
            num_steps=5,
            num_iterations=3,
            nocache=True,
            ppn=1
        )

        filename_obs = 'obs_rare.png'
        obs_rare['results'].savefig(filename_obs)

        filename_diff = '/tmp/diff_rare.png'
        res = compare_images(filename_obs, self.filename_rare,
                             threshold=DIFF_THRESHOLD,
                             file_image_diff=filename_diff,
                             name='rarecurves')
        self.assertTrue(res)


if __name__ == '__main__':
    main()
