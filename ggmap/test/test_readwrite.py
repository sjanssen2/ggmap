from unittest import TestCase, main

from skbio.util import get_data_path

from ggmap.readwrite import read_ncbi_nodes, read_metaphlan_markers_info


class ContactsTests(TestCase):
    def setUp(self):
        self.file_nodes = get_data_path('head_nodes.dmp')
        self.true_nodes = {16: 32011,
                           1: 1,
                           2: 131567,
                           6: 335928,
                           7: 6,
                           9: 32199,
                           10: 1706371,
                           11: 1707,
                           13: 203488,
                           14: 13}
        self.file_names = get_data_path('head_names.dmp')
        self.file_mpmarkers = get_data_path('subset_markers_info.txt')
        self.true_marker = {
            's__Sulfolobus_spindle_shaped_virus_2': {'GeneID': {'2658371'}},
            's__Escherichia_phage_vB_EcoP_G7C': {'GeneID': {'11117645',
                                                            '11117646'}},
            's__Cypovirus_15': {'NC': {'NC_002560.1', 'NC_002566.1'}},
            's__Helicobacter_winghamensis': {'gi': {'225074862'}},
            'p__Armatimonadetes': {'gi': {'512550081'}},
            's__Tomato_leaf_curl_Patna_betasatellite': {'NC': {'NC_012493.1'}},
            's__Mycobacterium_phage_Omega': {'GeneID': {'1259969'}},
            's__Eubacterium_cellulosolvens': {'gi': {'389575461'}},
            's__Streptomyces_sp_KhCrAH_244': {'gi': {'483970126'}},
            's__Tomato_begomovirus_satellite_DNA_beta': {'NC': {'NC_004904.1'}}
        }

    def test_read_ncbi_nodes(self):
        nodes = read_ncbi_nodes(self.file_nodes)
        self.assertEqual(self.true_nodes, nodes)

        with self.assertRaises(ValueError):
            read_ncbi_nodes(self.file_names)

        with self.assertRaises(IOError):
            read_ncbi_nodes('/tmp/non')

    def test_read_metaphlan_markers_info(self):
        self.assertEqual(self.true_marker,
                         read_metaphlan_markers_info(self.file_mpmarkers))

        with self.assertRaises(IOError):
            read_metaphlan_markers_info('/tmp/non')

        self.assertEqual({}, read_metaphlan_markers_info(self.file_nodes))

if __name__ == '__main__':
    main()
