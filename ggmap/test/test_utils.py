from unittest import TestCase, main
from io import StringIO
from copy import deepcopy

from skbio.util import get_data_path

from ggmap.utils import update_taxids
from ggmap.readwrite import read_taxid_list, read_ncbi_merged


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

    def test_update_taxids(self):
        mptaxids = read_taxid_list(self.file_mptaxids)
        self.assertEqual(mptaxids, self.true_old_mptaxids)

        mod_mptaxids = update_taxids(mptaxids,
                                     read_ncbi_merged(self.file_merged))

        # self.assertNotEqual(mod_mptaxids, self.true_old_mptaxids)
        self.assertEqual(mod_mptaxids, self.true_new_mptaxids)


if __name__ == '__main__':
    main()
