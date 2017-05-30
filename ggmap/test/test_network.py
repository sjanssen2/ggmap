from unittest import TestCase, main

from skbio.util import get_data_path
from skbio.stats.distance import DistanceMatrix
import pandas as pd

from ggmap.snippets import detect_distant_groups


class NetworkTests(TestCase):
    def setUp(self):
        self.dm = DistanceMatrix.read(get_data_path('weampq2_uuni.dm'))
        self.meta = pd.read_csv(get_data_path('weampq2.meta.tsv'),
                                sep='\t',
                                index_col=0).iloc[:, 0]
        self.exp_nw = {'metric_name': 'unweighted_unifrac',
                       'network':
                       {'Near': {'mainland': {'avgdist': 0.7703032874912592},
                                 'Andreanof': {'avgdist': 0.7627412877386373}},
                        'mainland': {'Andreanof': {'avgdist': 0.76297984119}}},
                       'num_permutations': 999,
                       'n_per_group':
                       pd.Series(data=[45, 33, 25],
                                 index=['Near', 'mainland', 'Andreanof'],
                                 name='Unnamed: 1'),
                       'min_group_size': 5}
        self.exp_pvalues = dict({
            'Near': {'mainland': 0.71399999999999997,
                     'Andreanof': 0.115},
            'mainland': {'Andreanof': 0.64600000000000}})

    def test_detect_distant_groups(self):
        obs_nw = detect_distant_groups(self.dm, 'unweighted_unifrac',
                                       self.meta)

        obs_pvalues = dict()
        for a in obs_nw['network'].keys():
            obs_pvalues[a] = dict()
            for b in obs_nw['network'][a].keys():
                obs_pvalues[a][b] = obs_nw['network'][a][b]['p-value']
                del obs_nw['network'][a][b]['p-value']

        # all but p-value are static
        self.assertCountEqual(obs_nw, self.exp_nw)
        for a in self.exp_pvalues.keys():
            for b in self.exp_pvalues[a].keys():
                self.assertAlmostEqual(self.exp_pvalues[a][b],
                                       obs_pvalues[a][b], 1)


if __name__ == '__main__':
    main()
