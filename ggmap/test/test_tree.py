from unittest import TestCase, main
from io import StringIO

from skbio.util import get_data_path

from ggmap.readwrite import read_ncbi_nodes, read_metaphlan_markers_info, \
                            read_taxid_list
from ggmap.tree import get_lineage, build_ncbi_tree, map_metaphlan_onto_ncbi


class TreeTests(TestCase):
    def setUp(self):
        self.file_nodes = get_data_path('top_nodes.dmp')
        self.file_nodes_head = get_data_path('head_nodes.dmp')
        self.taxonomy = read_ncbi_nodes(self.file_nodes)
        self.file_nodes_mock = get_data_path('mock_nodes.dmp')
        self.file_mpmarkers = get_data_path('subset_markers_info.txt')
        self.file_mptaxids = get_data_path('subset_taxids_metaphlan.txt')

    def test_get_lineage(self):
        self.assertEqual(get_lineage(2, self.taxonomy),
                         [1, 131567, 2])
        self.assertEqual(get_lineage(1, self.taxonomy), [1])
        with self.assertRaises(ValueError):
            get_lineage(3, self.taxonomy)

    def test_build_ncbi_tree(self):
        tree = build_ncbi_tree(self.taxonomy)
        self.assertCountEqual(list(map(lambda node: node.name, tree.tips())),
                              [28384, 2, 2759, 2157, 12884, 12908, 10239])

        out = StringIO()
        tree = build_ncbi_tree(self.taxonomy, verbose=True, out=out)
        self.assertCountEqual(list(map(lambda node: node.name, tree.tips())),
                              [28384, 2, 2759, 2157, 12884, 12908, 10239])
        self.assertEqual(out.getvalue().strip(),
                         "build ncbi tree for 7 tips: ....... done.")

        with self.assertRaises(KeyError):
            build_ncbi_tree(read_ncbi_nodes(self.file_nodes_head))

    def test_map_metaphlan_onto_ncbi(self):
        tree_ncbi = build_ncbi_tree(read_ncbi_nodes(self.file_nodes_mock))
        clades_metaphlan = read_metaphlan_markers_info(self.file_mpmarkers)
        taxids_metaphlan = read_taxid_list(self.file_mptaxids)
        out = StringIO()
        tree_mp = map_metaphlan_onto_ncbi(tree_ncbi, clades_metaphlan,
                                          taxids_metaphlan, verbose=True,
                                          out=out)

        self.assertEqual(tree_ncbi.count(), 35)
        self.assertEqual(tree_mp.count(), 19)

        clades = set()
        for node in tree_mp.find_by_func(lambda node:
                                         hasattr(node, 'mp_clades')):
            clades |= node.mp_clades
        self.assertCountEqual({'s__Helicobacter_winghamensis',
                               's__Sulfolobus_spindle_shaped_virus_2',
                               's__Streptomyces_sp_KhCrAH_244',
                               's__Escherichia_phage_vB_EcoP_G7C',
                               's__Eubacterium_cellulosolvens',
                               's__Mycobacterium_phage_Omega',
                               'p__Armatimonadetes'}, clades)
        self.assertNotIn('s__Tomato_leaf_curl_Patna_betasatellite', clades)
        self.assertNotIn('s__Tomato_begomovirus_satellite_DNA_beta', clades)
        self.assertNotIn('s__Cypovirus_15', clades)

        self.assertIn('Starting deep copy (might take 40 seconds): ... done.',
                      out.getvalue().strip())
        self.assertIn(("Cannot find taxid 575918 in taxonomy for clade "
                       "'s__Tomato_leaf_curl_Patna_betasatellite'"),
                      out.getvalue().strip())


if __name__ == '__main__':
    main()
