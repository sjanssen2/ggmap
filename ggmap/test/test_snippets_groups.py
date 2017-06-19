from unittest import TestCase, main
import pandas as pd

import matplotlib.pyplot as plt
from skbio.util import get_data_path

from ggmap.snippets import (detect_distant_groups_alpha)

plt.switch_backend('Agg')
plt.rc('font', family='DejaVu Sans')


class ReadWriteTests(TestCase):
    def setUp(self):
        self.exp_alpha = dict()

        self.exp_alpha['AGE'] = {
            'min_group_size': 21,
            'n_per_group':
            pd.Series(data=[72, 60, 27],
                      index=['AD', 'AHY', 'HY'],
                      name='AGE'),
            'group_name': 'AGE',
            'network': {'AD': {'AHY': {'p-value': 0.36674407481201665},
                               'HY': {'p-value': 0.39396076805143765}},
                        'AHY': {'HY': {'p-value': 0.17018258749773929}}},
            'metric_name': 'PD_whole_tree',
            'num_permutations': None}

        self.exp_alpha['coll_year'] = {
            'min_group_size': 21,
            'n_per_group':
            pd.Series(data=[38, 25],
                      index=['1999', '2000'],
                      name='coll_year'),
            'group_name': 'coll_year',
            'network': {'1999': {'2000': {'p-value': 0.082736338951025959}}},
            'metric_name': 'PD_whole_tree',
            'num_permutations': None}

        self.exp_alpha['diet_brief'] = {
            'min_group_size': 21,
            'n_per_group':
            pd.Series(data=[65, 53, 46],
                      index=['arthropods', 'herbivore', 'omnivore'],
                      name='diet_brief'),
            'group_name': 'diet_brief',
            'network':
            {'arthropods': {
                'omnivore': {'p-value': 0.29899394534610324},
                'herbivore': {'p-value': 0.32478261637443484}},
             'herbivore': {'omnivore': {'p-value': 0.050714833224854913}}},
            'metric_name': 'PD_whole_tree',
            'num_permutations': None}

        self.exp_alpha['norm_genpop'] = {
            'min_group_size': 21,
            'n_per_group':
            pd.Series(data=[],
                      name='norm_genpop', dtype=int),
            'group_name':
            'norm_genpop',
            'network': {},
            'metric_name':
            'PD_whole_tree',
            'num_permutations': None}

        self.exp_alpha['norm_q2_genpop'] = {
            'min_group_size': 21,
            'n_per_group':
            pd.Series(data=[],
                      index=[],
                      name='norm_q2_genpop', dtype=int),
            'group_name': 'norm_q2_genpop',
            'network': {},
            'metric_name': 'PD_whole_tree',
            'num_permutations': None}

        self.exp_alpha['Q2'] = {
            'min_group_size': 21,
            'n_per_group':
            pd.Series(data=[73, 57, 52],
                      index=['Near', 'mainland', 'Andreanof'],
                      name='Q2'),
            'group_name': 'Q2',
            'network':
            {'mainland': {'Andreanof': {'p-value': 0.13011439773562078}},
             'Near': {'mainland': {'p-value': 0.75324563183982074},
                      'Andreanof': {'p-value': 0.22449658255463945}}},
            'metric_name': 'PD_whole_tree',
            'num_permutations': None}

        self.exp_alpha['sample substance'] = {
            'min_group_size': 21,
            'n_per_group':
            pd.Series(data=[105, 57],
                      index=['G', 'P'],
                      name='sample substance'),
            'group_name': 'sample substance',
            'network': {'G': {'P': {'p-value': 0.39602006721574157}}},
            'metric_name': 'PD_whole_tree',
            'num_permutations': None}

        self.exp_alpha['seasons'] = {
            'min_group_size': 21,
            'n_per_group':
            pd.Series(data=[84, 61, 35],
                      index=['summer', 'spring', 'fall'],
                      name='seasons'),
            'group_name': 'seasons',
            'network':
            {'summer': {'fall': {'p-value': 0.95581571284465361},
             'spring': {'p-value': 0.13364106719123708}},
             'spring': {'fall': {'p-value': 0.34134279625829611}}},
            'metric_name': 'PD_whole_tree',
            'num_permutations': None}

        self.exp_alpha['sex'] = {
            'min_group_size': 21,
            'n_per_group':
            pd.Series(data=[111, 57],
                      index=['M', 'F'],
                      name='sex'),
            'group_name': 'sex',
            'network': {'M': {'F': {'p-value': 0.50285430241897211}}},
            'metric_name': 'PD_whole_tree',
            'num_permutations': None}

        self.exp_alpha['smj_genusspecies'] = {
            'min_group_size': 21,
            'n_per_group':
            pd.Series(data=[28, 26, 25, 24, 22],
                      index=['Anas crecca', 'Corvus corax',
                             'Troglodytes pacificus', 'Calidris ptilocnemis',
                             'Lagopus muta'],
                      name='smj_genusspecies'),
            'group_name': 'smj_genusspecies',
            'network':
            {'Corvus corax':
             {'Calidris ptilocnemis': {'p-value': 0.0012242807652588685},
              'Lagopus muta': {'p-value': 0.12830141671704251},
              'Troglodytes pacificus': {'p-value': 0.82845334670109438}},
             'Calidris ptilocnemis':
                {'Lagopus muta': {'p-value': 0.030310229970707401}},
             'Troglodytes pacificus':
                {'Calidris ptilocnemis': {'p-value': 0.00033067796144021903},
                 'Lagopus muta': {'p-value': 0.15012082224192319}},
             'Anas crecca':
                {'Corvus corax': {'p-value': 0.00051847227126772082},
                 'Calidris ptilocnemis': {'p-value': 0.46842423431347424},
                 'Lagopus muta': {'p-value': 0.0076357283167251561},
                 'Troglodytes pacificus': {'p-value':
                                           0.00021806456438494139}}},
            'metric_name': 'PD_whole_tree',
            'num_permutations': None}

        self.exp_alpha['weight_log'] = {
            'min_group_size': 21,
            'n_per_group':
            pd.Series(data=[91, 46, 24],
                      index=['1.0', '2.0', '3.0'],
                      name='weight_log'),
            'group_name': 'weight_log',
            'network':
            {'1.0': {'2.0': {'p-value': 0.46446128418804788},
                     '3.0': {'p-value': 0.0078136142778079693}},
             '2.0': {'3.0': {'p-value': 0.0017829155222652295}}},
            'metric_name': 'PD_whole_tree',
            'num_permutations': None}

    def compareNetworks(self, a, b):
        for key in a.keys():
            if key == 'network':
                for gr_x in a[key].keys():
                    if (gr_x not in a[key]) | (gr_x not in b[key]):
                        return False
                    for gr_y in a[key][gr_x].keys():
                        if (gr_y not in a[key][gr_x]) |\
                           (gr_y not in b[key][gr_x]):
                            return False
                        for field in a[key][gr_x][gr_y].keys():
                            if field == 'p-value':
                                continue
                            if a[key][gr_x][gr_y][field] !=\
                               b[key][gr_x][gr_y][field]:
                                return False
                continue
            if key == 'n_per_group':
                if a[key].equals(b[key]) is False:
                    return False
                continue
            if a[key] != b[key]:
                return False

        return True

    def test_detect_distant_groups_alpha(self):
        fields = ['AGE', 'coll_year', 'diet_brief', 'norm_genpop',
                  'norm_q2_genpop', 'Q2', 'sample substance', 'seasons',
                  'sex', 'smj_genusspecies', 'weight_log']

        for field in fields:
            alpha = pd.read_csv(get_data_path(
                'detectGroups/Alpha/alpha_%s.tsv' % field),
                sep="\t", header=None, index_col=0).iloc[:, 0]
            alpha.name = 'PD_whole_tree'
            meta = pd.read_csv(get_data_path(
                'detectGroups/Alpha/meta_%s.tsv' % field),
                sep="\t", header=None, index_col=0,
                names=['index', field], dtype=str).loc[:, field]
            obs = detect_distant_groups_alpha(alpha, meta)

            res = self.compareNetworks(obs, self.exp_alpha[field])
            self.assertTrue(res)


if __name__ == '__main__':
    main()
