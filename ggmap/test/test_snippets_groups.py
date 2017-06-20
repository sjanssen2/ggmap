from unittest import TestCase, main
import pandas as pd
from math import isclose
from tempfile import mkstemp
from os import remove

import matplotlib.pyplot as plt
from skbio.util import get_data_path
from skbio.stats.distance import DistanceMatrix

from ggmap.snippets import (detect_distant_groups_alpha,
                            detect_distant_groups,
                            plotDistant_groups,
                            plotGroup_histograms)
from ggmap.imgdiff import compare_images

plt.switch_backend('Agg')
plt.rc('font', family='DejaVu Sans')


class SnippetTests(TestCase):
    def setUp(self):
        self.fields = ['AGE', 'coll_year', 'diet_brief', 'norm_genpop',
                       'norm_q2_genpop', 'Q2', 'sample_substance', 'seasons',
                       'sex', 'smj_genusspecies', 'weight_log']

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

        self.exp_alpha['sample_substance'] = {
            'min_group_size': 21,
            'n_per_group':
            pd.Series(data=[105, 57],
                      index=['G', 'P'],
                      name='sample_substance'),
            'group_name': 'sample_substance',
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

        self.exp_beta = dict()

        self.exp_beta['AGE'] = {
            'min_group_size': 5,
            'n_per_group':
            pd.Series(data=[72, 60, 27, 19],
                      index=['AD', 'AHY', 'HY', 'U'],
                      name='AGE'),
            'group_name': 'AGE',
            'network':
            {'HY': {'U': {'avgdist': 0.84064493756431768,
                          'p-value': 0.084500000000000006}},
             'AD': {'HY': {'avgdist': 0.8415325628083512,
                           'p-value': 0.01},
                    'U': {'avgdist': 0.84032227902278234,
                          'p-value': 0.071999999999999995},
                    'AHY': {'avgdist': 0.83563944028957649,
                            'p-value': 0.0814}},
             'AHY': {'HY': {'avgdist': 0.83954659855097846,
                            'p-value': 0.0079000000000000008},
                     'U': {'avgdist': 0.83127508285072371,
                           'p-value': 0.80869999999999997}}},
            'metric_name': 'unweighted_unifrac',
            'num_permutations': 99}

        self.exp_beta['coll_year'] = {
            'min_group_size': 10,
            'n_per_group':
            pd.Series(data=[38, 25, 17, 16, 16, 12],
                      index=['1999', '2000', '2001', '2002', '2004', '2009'],
                      name='coll_year'),
            'group_name': 'coll_year',
            'network':
            {'1999': {'2009': {'avgdist': 0.8367942007544058,
                               'p-value': 0.00020000000000000001},
                      '2004': {'avgdist': 0.83548860103803613,
                               'p-value': 0.1202},
                      '2001': {'avgdist': 0.85008860405209141,
                               'p-value': 0.0041999999999999997},
                      '2002': {'avgdist': 0.82522628895448191,
                               'p-value': 0.071400000000000005},
                      '2000': {'avgdist': 0.84016257676219686,
                               'p-value': 0.0073000000000000001}},
             '2004': {'2009': {'avgdist': 0.82480414108520306,
                               'p-value': 0.0025000000000000001}},
             '2001': {'2009': {'avgdist': 0.86086958880499509,
                               'p-value': 0.00020000000000000001},
                      '2004': {'avgdist': 0.84905730466059193,
                               'p-value': 0.0030999999999999999},
                      '2002': {'avgdist': 0.83651097104808447,
                               'p-value': 0.0037000000000000002}},
             '2002': {'2009': {'avgdist': 0.80666959000218752,
                               'p-value': 0.0018},
                      '2004': {'avgdist': 0.81150384766870709,
                               'p-value': 0.32519999999999999}},
             '2000': {'2009': {'avgdist': 0.84255014669849659,
                               'p-value': 0.0001},
                      '2004': {'avgdist': 0.84175633825045992,
                               'p-value': 0.00080000000000000004},
                      '2001': {'avgdist': 0.85989716436618358,
                               'p-value': 0.0001},
                      '2002': {'avgdist': 0.82842597848702515,
                               'p-value': 0.00050000000000000001}}},
            'metric_name': 'unweighted_unifrac',
            'num_permutations': 99}

        self.exp_beta['diet_brief'] = {
            'min_group_size': 5,
            'n_per_group':
            pd.Series(data=[65, 53, 46, 18],
                      index=['arthropods', 'herbivore',
                             'omnivore', 'granivore'],
                      name='diet_brief'),
            'group_name': 'diet_brief',
            'network':
            {'arthropods': {'granivore': {'avgdist': 0.82738912829585476,
                                          'p-value': 0.019400000000000001},
                            'omnivore': {'avgdist': 0.83162115781211532,
                                         'p-value': 0.0001},
             'herbivore': {'avgdist': 0.84974796038977285,
                           'p-value': 0.0001}},
             'omnivore': {'granivore': {'avgdist': 0.81693619251780802,
                                        'p-value': 0.1636}},
             'herbivore': {'granivore': {'avgdist': 0.85066144456070658,
                                         'p-value': 0.0001},
                           'omnivore': {'avgdist': 0.85445401485936756,
                                        'p-value': 0.0001}}},
            'metric_name': 'unweighted_unifrac',
            'num_permutations': 99}

        self.exp_beta['norm_genpop'] = {
            'min_group_size': 5,
            'n_per_group':
            pd.Series(data=[12, 5],
                      index=['Aleutian', 'Old World'],
                      name='norm_genpop', dtype=int),
            'group_name':
            'norm_genpop',
            'network':
            {'Aleutian': {'Old World': {'avgdist': 0.81772526690298331,
                                        'p-value': 0.314}}},
            'metric_name': 'unweighted_unifrac',
            'num_permutations': 99}

        self.exp_beta['norm_q2_genpop'] = {
            'min_group_size': 5,
            'n_per_group':
            pd.Series(data=[6, 5, 5],
                      index=['Andreanof: Aleutian',
                             'Near: Old World', 'Near: Aleutian'],
                      name='norm_q2_genpop', dtype=int),
            'group_name': 'norm_q2_genpop',
            'network':
            {'Andreanof: Aleutian':
                {'Near: Aleutian': {'avgdist': 0.84850247518656652,
                                    'p-value': 0.083599999999999994},
                 'Near: Old World': {'avgdist': 0.80923487646360004,
                                     'p-value': 0.34100000000000003}},
             'Near: Old World':
                {'Near: Aleutian': {'avgdist': 0.83360898661304017,
                                    'p-value': 0.13159999999999999}}},
            'metric_name': 'unweighted_unifrac',
            'num_permutations': 99}

        self.exp_beta['Q2'] = {
            'min_group_size': 5,
            'n_per_group':
            pd.Series(data=[73, 57, 52],
                      index=['Near', 'mainland', 'Andreanof'],
                      name='Q2'),
            'group_name': 'Q2',
            'network':
            {'Near': {'Andreanof': {'avgdist': 0.83953930115707776,
                                    'p-value': 0.01},
                      'mainland': {'avgdist': 0.83383070542603732,
                                   'p-value': 0.062799999999999995}},
             'mainland': {'Andreanof': {'avgdist': 0.84382955854433805,
                                        'p-value': 0.091300000000000006}}},
            'metric_name': 'unweighted_unifrac',
            'num_permutations': 99}

        self.exp_beta['sample_substance'] = {
            'min_group_size': 5,
            'n_per_group':
            pd.Series(data=[105, 57, 13],
                      index=['G', 'P', 'P/G'],
                      name='sample_substance'),
            'group_name': 'sample_substance',
            'network':
            {'G': {'P/G': {'avgdist': 0.83669917714891728,
                           'p-value': 0.55359999999999998},
                   'P': {'avgdist': 0.83941861561237507,
                         'p-value': 0.0424}},
             'P': {'P/G': {'avgdist': 0.84009551170351826,
                           'p-value': 0.71179999999999999}}},
            'metric_name': 'unweighted_unifrac',
            'num_permutations': 99}

        self.exp_beta['seasons'] = {
            'min_group_size': 5,
            'n_per_group':
            pd.Series(data=[84, 61, 35],
                      index=['summer', 'spring', 'fall'],
                      name='seasons'),
            'group_name': 'seasons',
            'network':
            {'summer': {'spring': {'avgdist': 0.83774360775004986,
                                   'p-value': 0.28739999999999999},
             'fall': {'avgdist': 0.83432598621655241,
                      'p-value': 0.030200000000000001}},
             'spring': {'fall': {'avgdist': 0.84045022044782058,
                                 'p-value': 0.032099999999999997}}},
            'metric_name': 'unweighted_unifrac',
            'num_permutations': 99}

        self.exp_beta['sex'] = {
            'min_group_size': 5,
            'n_per_group':
            pd.Series(data=[111, 57, 11],
                      index=['M', 'F', 'U'],
                      name='sex'),
            'group_name': 'sex',
            'network':
            {'F': {'U': {'avgdist': 0.83483232215002878,
                         'p-value': 0.67120000000000002}},
             'M': {'U': {'avgdist': 0.83872511753036849,
                         'p-value': 0.81730000000000003},
                   'F': {'avgdist': 0.83532630343035663,
                         'p-value': 0.28739999999999999}}},
            'metric_name': 'unweighted_unifrac',
            'num_permutations': 99}

        self.exp_beta['smj_genusspecies'] = {
            'min_group_size': 5,
            'n_per_group':
            pd.Series(data=[28, 26, 25, 24, 22, 20, 18, 16],
                      index=['Anas crecca', 'Corvus corax',
                             'Troglodytes pacificus', 'Calidris ptilocnemis',
                             'Lagopus muta', 'Melospiza melodia',
                             'Plectrophenax nivalis', 'Calcarius lapponicus'],
                      name='smj_genusspecies'),
            'group_name': 'smj_genusspecies',
            'network':
            {'Calidris ptilocnemis':
                {'Lagopus muta': {'avgdist': 0.83829058606745466,
                                  'p-value': 0.0001},
                 'Calcarius lapponicus': {'avgdist': 0.85611180382685692,
                                          'p-value': 0.0001},
                 'Melospiza melodia': {'avgdist': 0.8232349250790354,
                                       'p-value': 0.0001},
                 'Plectrophenax nivalis': {'avgdist': 0.8351043927280648,
                                           'p-value': 0.0001}},
             'Lagopus muta':
                {'Calcarius lapponicus': {'avgdist': 0.86097590003862512,
                                          'p-value': 0.0001},
                 'Melospiza melodia': {'avgdist': 0.84527449273463862,
                                       'p-value': 0.0001},
                 'Plectrophenax nivalis': {'avgdist': 0.84095003672914648,
                                           'p-value': 0.0001}},
             'Plectrophenax nivalis':
                {'Calcarius lapponicus': {'avgdist': 0.82838601064362161,
                                          'p-value': 0.1244}},
             'Corvus corax':
                {'Calcarius lapponicus': {'avgdist': 0.85305757491399037,
                                          'p-value': 0.00059999999999999995},
                 'Calidris ptilocnemis': {'avgdist': 0.84821346703077405,
                                          'p-value': 0.0001},
                 'Plectrophenax nivalis': {'avgdist': 0.82629915369720719,
                                           'p-value': 0.0126},
                 'Lagopus muta': {'avgdist': 0.84539844224103844,
                                  'p-value': 0.0001},
                 'Melospiza melodia': {'avgdist': 0.82437591628707307,
                                       'p-value': 0.00069999999999999999},
                 'Troglodytes pacificus': {'avgdist': 0.825216648861406,
                                           'p-value': 0.00089999999999999998}},
             'Anas crecca':
                {'Calcarius lapponicus': {'avgdist': 0.86617530466183479,
                                          'p-value': 0.00029999999999999997},
                 'Calidris ptilocnemis': {'avgdist': 0.83472858132931704,
                                          'p-value': 0.0001},
                 'Plectrophenax nivalis': {'avgdist': 0.85235468721186114,
                                           'p-value': 0.0001},
                 'Corvus corax': {'avgdist': 0.86181429935340648,
                                  'p-value': 0.00020000000000000001},
                 'Lagopus muta': {'avgdist': 0.8473999314432451,
                                  'p-value': 0.0001},
                 'Melospiza melodia': {'avgdist': 0.84782768446828394,
                                       'p-value': 0.0001},
                 'Troglodytes pacificus': {'avgdist': 0.85356773864006863,
                                           'p-value': 0.0001}},
             'Melospiza melodia':
                {'Calcarius lapponicus': {'avgdist': 0.82240317253223449,
                                          'p-value': 0.022800000000000001},
                 'Plectrophenax nivalis': {'avgdist': 0.80476434298458877,
                                           'p-value': 0.16800000000000001}},
             'Troglodytes pacificus':
                {'Lagopus muta': {'avgdist': 0.83817085927644008,
                                  'p-value': 0.0001},
                 'Calcarius lapponicus': {'avgdist': 0.84054899880657752,
                                          'p-value': 0.00089999999999999998},
                 'Calidris ptilocnemis': {'avgdist': 0.83658015441226663,
                                          'p-value': 0.0001},
                 'Melospiza melodia': {'avgdist': 0.81535501251727194,
                                       'p-value': 0.00059999999999999995},
                 'Plectrophenax nivalis': {'avgdist': 0.81934446973836206,
                                           'p-value': 0.0067999999999999996}}},
            'metric_name': 'unweighted_unifrac',
            'num_permutations': 99}

        self.exp_beta['weight_log'] = {
            'min_group_size': 5,
            'n_per_group':
            pd.Series(data=[91, 46, 24, 6],
                      index=['1.0', '2.0', '3.0', '0.0'],
                      name='weight_log'),
            'group_name': 'weight_log',
            'network':
            {'1.0': {'0.0': {'avgdist': 0.82210886773573444,
                             'p-value': 0.31009999999999999},
                     '2.0': {'avgdist': 0.84844882889132023,
                             'p-value': 0.0001},
                     '3.0': {'avgdist': 0.83665460684904569,
                             'p-value': 0.0001}},
             '2.0': {'0.0': {'avgdist': 0.84566597694949996,
                             'p-value': 0.0229},
                     '3.0': {'avgdist': 0.85638366400852717,
                             'p-value': 0.0001}},
             '3.0': {'0.0': {'avgdist': 0.83326796627222222,
                             'p-value': 0.051499999999999997}}},
            'metric_name': 'unweighted_unifrac',
            'num_permutations': 99}

    def compareNetworks(self, a, b):
        for key in a.keys():
            if key == 'network':
                elems_a = set().union(a['network'].keys(),
                                      *a['network'].values())
                elems_b = set().union(b['network'].keys(),
                                      *b['network'].values())
                self.assertCountEqual(elems_a, elems_b)

                for a_x in a['network'].keys():
                    for a_y in a['network'][a_x].keys():
                        b_x, b_y = None, None
                        if a_x in b['network']:
                            if a_y in b['network'][a_x]:
                                b_x, b_y = a_x, a_y
                        if b_x is None:
                            for i in b['network'].keys():
                                if a_x in b['network'][i]:
                                    b_x, b_y = a_y, a_x

                        for field in a['network'][a_x][a_y].keys():
                            if field == 'p-value':
                                continue
                            elif field == 'avgdist':
                                self.assertTrue(
                                    isclose(a['network'][a_x][a_y][field],
                                            b['network'][b_x][b_y][field],
                                            rel_tol=1e-4))
                            else:
                                self.assertEqual(a['network'][a_x][a_y][field],
                                                 b['network'][b_x][b_y][field])
            elif key == 'n_per_group':
                sorted_a = a[key].sort_index()
                sorted_b = b[key].sort_index()
                self.assertTrue(sorted_a.equals(sorted_b))
            else:
                self.assertEqual(a[key], b[key])

        return True

    def test_detect_distant_groups_alpha(self):
        for field in self.fields:
            alpha = pd.read_csv(get_data_path(
                'detectGroups/Alpha/alpha_%s.tsv' % field),
                sep="\t", header=None, index_col=0).iloc[:, 0]
            alpha.name = 'PD_whole_tree'
            meta = pd.read_csv(get_data_path(
                'detectGroups/meta_%s.tsv' % field),
                sep="\t", header=None, index_col=0,
                names=['index', field], dtype=str).loc[:, field]
            obs = detect_distant_groups_alpha(alpha, meta)

            res = self.compareNetworks(obs, self.exp_alpha[field])
            self.assertTrue(res)

    def test_detect_distant_groups(self):
        for field in self.fields:
            beta = DistanceMatrix.read(
                get_data_path('detectGroups/Beta/beta_%s.dm' % field))
            meta = pd.read_csv(get_data_path(
                'detectGroups/meta_%s.tsv' % field),
                sep="\t", header=None, index_col=0,
                names=['index', field], dtype=str).loc[:, field]
            min_group_size = 5
            if field == 'coll_year':
                min_group_size = 10
            obs = detect_distant_groups(beta, 'unweighted_unifrac', meta,
                                        num_permutations=99,
                                        min_group_size=min_group_size)

            res = self.compareNetworks(obs, self.exp_beta[field])
            self.assertTrue(res)

    def test_plotDistant_groups(self):
        for field in self.fields:
            fig, ax = plt.subplots()
            plotDistant_groups(**(self.exp_alpha[field]),
                               pthresh=0.05,
                               _type='alpha', draw_edgelabel=True, ax=ax)
            file_plotname = 'alpha_network_%s.png' % field
            file_dummy = mkstemp('.png', prefix=file_plotname+'.')[1]
            plt.savefig(file_dummy)
            plt.close()
            res = compare_images(
                get_data_path('detectGroups/Alpha/alpha_network_%s.png' %
                              field),
                file_dummy,
                file_image_diff='./diff.'+file_plotname, threshold=10)
            if res[0] is True:
                remove(file_dummy)
            else:
                print(res)
            self.assertTrue(res[0])

    def test_plotGroup_histograms(self):
        for field in self.fields:
            fig, ax = plt.subplots()
            alpha = pd.read_csv(get_data_path(
                'detectGroups/Alpha/alpha_%s.tsv' % field),
                sep="\t", header=None, index_col=0).iloc[:, 0]
            alpha.name = 'PD_whole_tree'
            meta = pd.read_csv(get_data_path(
                'detectGroups/meta_%s.tsv' % field),
                sep="\t", header=None, index_col=0,
                names=['index', field], dtype=str).loc[:, field]

            plotGroup_histograms(alpha, meta, ax=ax)
            file_plotname = 'alpha_histogram_%s.png' % field
            file_dummy = mkstemp('.png', prefix=file_plotname+'.')[1]
            plt.savefig(file_dummy)
            plt.close()
            res = compare_images(
                get_data_path('detectGroups/Alpha/alpha_histogram_%s.png' %
                              field),
                file_dummy,
                file_image_diff='./diff.'+file_plotname)
            if res[0] is True:
                remove(file_dummy)
            else:
                print(res)
            self.assertTrue(res[0])


if __name__ == '__main__':
    main()
