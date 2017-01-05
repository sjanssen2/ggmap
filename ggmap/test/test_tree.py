from unittest import TestCase, main
from io import StringIO

from skbio.util import get_data_path

from ggmap.readwrite import read_ncbi_nodes, read_metaphlan_markers_info, \
                            read_taxid_list, read_gg_accessions, \
                            read_gg_otu_map
from ggmap.tree import get_lineage, build_ncbi_tree, map_onto_ncbi, \
                       match_metaphlan_greengenes, _get_otus_from_clade


class TreeTests(TestCase):
    def setUp(self):
        self.file_nodes = get_data_path('top_nodes.dmp')
        self.file_nodes_head = get_data_path('head_nodes.dmp')
        self.taxonomy = read_ncbi_nodes(self.file_nodes)
        self.file_nodes_mock = get_data_path('mock_nodes.dmp')
        self.file_mpmarkers = get_data_path('subset_markers_info.txt')
        self.file_mptaxids = get_data_path('subset_taxids_metaphlan.txt')
        self.file_gg_accessions = \
            get_data_path('subset_gg_13_5_accessions.txt')
        self.file_gg_taxids = get_data_path('subset_taxids_gg.txt')
        self.file_gg_otumap = get_data_path('subset_97_otu_map.txt')

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

    def test_map_onto_ncbi_mp(self):
        tree_ncbi = build_ncbi_tree(read_ncbi_nodes(self.file_nodes_mock))
        clades_metaphlan = read_metaphlan_markers_info(self.file_mpmarkers)
        taxids_metaphlan = read_taxid_list(self.file_mptaxids)
        out = StringIO()
        tree_mp = map_onto_ncbi(tree_ncbi, clades_metaphlan, taxids_metaphlan,
                                attribute_name='mp_clades', verbose=True,
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
        self.assertIn(("Cannot find taxid 575918 in taxonomy for mp_clades "
                       "'s__Tomato_leaf_curl_Patna_betasatellite'"),
                      out.getvalue().strip())

    def test_map_onto_ncbi_gg(self):
        tree_ncbi = build_ncbi_tree(read_ncbi_nodes(self.file_nodes_mock))

        gg_ids_accessions = read_gg_accessions(self.file_gg_accessions)
        gg_taxids = read_taxid_list(self.file_gg_taxids)
        gg_otumap_97_orig = read_gg_otu_map(self.file_gg_otumap,
                                            gg_ids_accessions)

        out = StringIO()
        tree_gg = map_onto_ncbi(tree_ncbi, gg_otumap_97_orig, gg_taxids,
                                attribute_name='otus', verbose=True,
                                out=out)
        otus = set()
        for node in tree_gg.find_by_func(lambda node:
                                         hasattr(node, 'otus')):
            otus |= node.otus
        self.assertCountEqual({11054, 13988, 243587}, otus)
        self.assertNotIn('2328237', otus)

        self.assertIn('Starting deep copy (might take 40 seconds): ... done.',
                      out.getvalue().strip())
        self.assertNotIn("Cannot find taxid", out.getvalue().strip())

    def test__get_otus_from_clade(self):
        tree_ncbi = build_ncbi_tree(read_ncbi_nodes(self.file_nodes_mock))

        gg_ids_accessions = read_gg_accessions(self.file_gg_accessions)
        gg_taxids = read_taxid_list(self.file_gg_taxids)
        gg_otumap_97_orig = read_gg_otu_map(self.file_gg_otumap,
                                            gg_ids_accessions)
        tree_gg = map_onto_ncbi(tree_ncbi, gg_otumap_97_orig, gg_taxids,
                                attribute_name='otus', verbose=False)

        clades_metaphlan = read_metaphlan_markers_info(self.file_mpmarkers)
        taxids_metaphlan = read_taxid_list(self.file_mptaxids)
        out = StringIO()
        tree_mp = map_onto_ncbi(tree_ncbi, clades_metaphlan, taxids_metaphlan,
                                attribute_name='mp_clades', verbose=False,
                                out=out)

        clade = 's__Sulfolobus_spindle_shaped_virus_2'
        self.assertCountEqual(_get_otus_from_clade(clade, tree_mp, 'mp_clades',
                                                   tree_gg, 'otus'),
                              {243587})

        clade = 's__Mycobacterium_phage_Omega'
        self.assertCountEqual(_get_otus_from_clade(clade, tree_mp, 'mp_clades',
                                                   tree_gg, 'otus'),
                              {243587, 13988, 11054})

        with self.assertRaises(ValueError):
            clade = 's__Cypovirus_15'
            _get_otus_from_clade(clade, tree_mp, 'mp_clades', tree_gg, 'otus')

    def test_match_metaphlan_greengenes(self):
        tree_ncbi = build_ncbi_tree(read_ncbi_nodes(self.file_nodes_mock))

        gg_ids_accessions = read_gg_accessions(self.file_gg_accessions)
        gg_taxids = read_taxid_list(self.file_gg_taxids)
        gg_otumap_97_orig = read_gg_otu_map(self.file_gg_otumap,
                                            gg_ids_accessions)
        tree_gg = map_onto_ncbi(tree_ncbi, gg_otumap_97_orig, gg_taxids,
                                attribute_name='otus', verbose=False)

        clades_metaphlan = read_metaphlan_markers_info(self.file_mpmarkers)
        taxids_metaphlan = read_taxid_list(self.file_mptaxids)
        out = StringIO()
        tree_mp = map_onto_ncbi(tree_ncbi, clades_metaphlan, taxids_metaphlan,
                                attribute_name='mp_clades', verbose=False,
                                out=out)
        self.assertIn("575918", out.getvalue())
        self.assertIn('s__Tomato_leaf_curl_Patna_betasatellite',
                      out.getvalue())

        err = StringIO()
        self.assertCountEqual(match_metaphlan_greengenes(clades_metaphlan,
                                                         tree_mp, 'mp_clades',
                                                         tree_gg, 'otus', err),
                              {'s__Helicobacter_winghamensis': {11054},
                               'p__Armatimonadetes': {243587},
                               's__Eubacterium_cellulosolvens': {13988},
                               's__Streptomyces_sp_KhCrAH_244': {13988},
                               's__Mycobacterium_phage_Omega': {243587, 13988,
                                                                11054},
                               's__Sulfolobus_spindle_shaped_virus_2':
                               {243587},
                               's__Escherichia_phage_vB_EcoP_G7C': {243587,
                                                                    13988,
                                                                    11054}})
        self.assertIn("Clade 's__Cypovirus_15' omitted, since it is not",
                      err.getvalue())
        self.assertIn("Clade 's__Tomato_leaf_curl_Patna_betasatellite'",
                      err.getvalue())
        self.assertIn("Clade 's__Tomato_begomovirus_satellite_DNA_beta'",
                      err.getvalue())

if __name__ == '__main__':
    main()
