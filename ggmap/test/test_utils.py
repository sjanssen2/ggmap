from unittest import TestCase, main
from copy import deepcopy
from io import StringIO

from skbio.util import get_data_path

from ggmap.utils import update_taxids, \
                        _convert_metaphlan_profile_to_greengenes, \
                        convert_profiles
from ggmap.readwrite import read_taxid_list, read_ncbi_merged, \
                            read_clade2otus_map, read_metaphlan_profile


class UtilsTests(TestCase):
    def setUp(self):
        self.file_merged = get_data_path('head_merged.dmp')
        self.file_mpmarkers = get_data_path('subset_markers_info.txt')
        self.file_mptaxids = get_data_path('subset_taxids_metaphlan.txt')
        self.true_old_mptaxids = {
            'NC': {
                 'NC_012493.1': 575918, 'NC_002560.1': 134606,
                 'NC_002566.1': 134606, 'NC_004904.1': 234829},
            'gi': {
                '389575461': 633697, '483970126': 1157633,
                '225074862': 556267, '512550081': 1303518},
            'GeneID': {
                '2658371': 244590, '11117645': 1054461,
                '1259969': 205879, '11117646': 1054461}}

        self.true_new_mptaxids = deepcopy(self.true_old_mptaxids)
        self.true_new_mptaxids['GeneID']['1259969'] = 74313
        self.true_new_mptaxids['NC']['NC_004904.1'] = 29
        self.file_mp_gg_map = get_data_path('true_map.txt')
        self.file_mock_mp_profile = get_data_path('mock_metaphlan.bacprofile')
        self.true_mock_mp_profile = {243587: 0.4781166666666666,
                                     13988: 0.4000666666666667,
                                     11054: 0.12181666666666667}
        self.file_example_mp_profile = get_data_path(
            'example_metaphlan.bacprofile')

    def test_update_taxids(self):
        mptaxids = read_taxid_list(self.file_mptaxids)
        self.assertEqual(mptaxids, self.true_old_mptaxids)

        mod_mptaxids = update_taxids(mptaxids,
                                     read_ncbi_merged(self.file_merged))

        # self.assertNotEqual(mod_mptaxids, self.true_old_mptaxids)
        self.assertEqual(mod_mptaxids, self.true_new_mptaxids)

    def test__convert_metaphlan_profile_to_greengenes(self):
        mp2gg = read_clade2otus_map(self.file_mp_gg_map)
        mp_profile1 = read_metaphlan_profile(self.file_mock_mp_profile)
        res = _convert_metaphlan_profile_to_greengenes(mp_profile1, mp2gg)
        self.assertCountEqual(res, self.true_mock_mp_profile)

        mp_profile2 = read_metaphlan_profile(self.file_example_mp_profile)
        err = StringIO()
        res = _convert_metaphlan_profile_to_greengenes(mp_profile2, mp2gg, err)
        self.assertIn("Due to 15 unmatched MetaPhlAn lineages", err.getvalue())
        self.assertIn("erium|t__Bacillus_megaterium	1.24861", err.getvalue())

    def test_convert_profiles(self):
        mp2gg = read_clade2otus_map(self.file_mp_gg_map)

        with self.assertRaises(IOError):
            convert_profiles([self.file_mock_mp_profile, '/tmp/nonexist'],
                             mp2gg)

        err = StringIO()
        res = convert_profiles([self.file_mock_mp_profile,
                                self.file_example_mp_profile], mp2gg, out=err)
        self.assertCountEqual(['mock', 'example'], res.columns)
        self.assertCountEqual([243587, 13988, 11054], res.index)
        self.assertCountEqual([1.0, 0.0], res.sum(axis=0))

        err = StringIO()
        res = convert_profiles([self.file_mock_mp_profile,
                                self.file_example_mp_profile], mp2gg, out=err,
                               prefix='test_')
        self.assertCountEqual(['test_mock', 'test_example'], res.columns)

if __name__ == '__main__':
    main()
