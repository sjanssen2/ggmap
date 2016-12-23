from unittest import TestCase, main
import filecmp
import tempfile

from skbio.util import get_data_path

from ggmap.readwrite import read_ncbi_nodes, read_metaphlan_markers_info, \
                            read_taxid_list, _read_ncbitaxonomy_file, \
                            read_ncbi_merged, read_gg_accessions, \
                            read_gg_otu_map, write_clade2otus_map, \
                            read_clade2otus_map, read_metaphlan_profile


class ReadWriteTests(TestCase):
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
        self.file_mptaxids = get_data_path('subset_taxids_metaphlan.txt')
        self.true_mptaxids = {
            'NC': {
                 'NC_012493.1': 575918, 'NC_002560.1': 134606,
                 'NC_002566.1': 134606, 'NC_004904.1': 234829},
            'gi': {
                '389575461': 633697, '483970126': 1157633,
                '225074862': 556267, '512550081': 1303518},
            'GeneID': {
                '2658371': 244590, '11117645': 1054461,
                '1259969': 205879, '11117646': 1054461}}
        self.file_merged = get_data_path('head_merged.dmp')
        self.true_merged = {80: 155892, 67: 32033, 36: 184914, 37: 42,
                            76: 155892, 234829: 29, 12: 74109, 77: 74311,
                            46: 39, 205879: 74313}
        self.file_accessions = get_data_path('subset_gg_13_5_accessions.txt')
        self.true_accessions = {
            'IMG': {4486318: '2508501010', 4485750: '2504756013',
                    4486315: '2515154132', 4486317: '2509276016',
                    4486316: '2517093047', 4486300: '2509276007'},
            'Genbank': {4485548: 'NZ_GG661973.1', 470510: 'ACDO01000013.1',
                        4175983: 'JN530275.1', 266930: 'EU509548.1',
                        4466933: 'JQ694525.1', 2338842: 'JQ941795.1',
                        13988: 'X71860.1', 336267: 'FJ193781.1',
                        243587: 'AM749780.1'}}
        self.file_gg_taxids = get_data_path('subset_taxids_gg.txt')
        self.true_gg_taxids = {
            'Genbank': {'JQ694525.1': 57045, 'FJ193781.1': 165433,
                        'EU509548.1': 77133, 'ACDO01000013.1': 556267,
                        'JQ941795.1': 77133, 'X71860.1': 633697,
                        'NZ_GG661973.1': 556267, 'AM749780.1': 1303518,
                        'JN530275.1': 155900},
            'IMG': {'2504756013': 2157, '2508501120': 443254,
                    '2515154175': 1122605, '2509601042': 987045,
                    '2509276007': 633697}}
        self.file_gg_otumap = get_data_path('subset_97_otu_map.txt')
        self.true_gg_otumap = {243587: {'Genbank': {'AM749780.1'}},
                               13988: {'Genbank': {'X71860.1'}},
                               2328237: {},
                               11054: {'Genbank': {'ACDO01000013.1'}}}
        self.true_map = {'s__Helicobacter_winghamensis': {11054},
                         'p__Armatimonadetes': {243587},
                         's__Eubacterium_cellulosolvens': {13988},
                         's__Streptomyces_sp_KhCrAH_244': {13988},
                         's__Mycobacterium_phage_Omega': {243587, 13988,
                                                          11054},
                         's__Sulfolobus_spindle_shaped_virus_2':
                         {243587},
                         's__Escherichia_phage_vB_EcoP_G7C': {243587,
                                                              13988,
                                                              11054}}
        self.file_true_map = get_data_path('true_map.txt')
        self.file_example_profile = get_data_path(
            'example_metaphlan.bacprofile')
        self.true_example_profile = {
            ('k__Bacteria|p__Firmicutes|c__Bacilli|o__Bacillales|f__Staphyloco'
             'ccaceae|g__Staphylococcus|s__Staphylococcus_caprae_capitis|t__St'
             'aphylococcus_caprae_capitis'): 0.13874,
            ('k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetal'
             'es|f__Corynebacteriaceae|g__Corynebacterium|s__Corynebacterium_p'
             'seudogenitalium|t__GCF_000156615'): 9.6582,
            ('k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetal'
             'es|f__Propionibacteriaceae|g__Propionibacteriaceae'
             ): 0.22942,
            ('k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetal'
             'es|f__Corynebacteriaceae|g__Corynebacterium|s__Corynebacterium_t'
             'uberculostearicum|t__GCF_000175635'): 19.35194,
            ('k__Bacteria|p__Firmicutes|c__Bacilli|o__Bacillales|f__Bacillacea'
             'e|g__Bacillus|s__Bacillus_licheniformis|t__Bacillus_licheniformi'
             's'): 0.02527,
            ('k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetal'
             'es|f__Brevibacteriaceae|g__Brevibacterium|s__Brevibacterium'
             ): 0.09897,
            ('k__Bacteria|p__Firmicutes|c__Bacilli|o__Bacillales|f__Staphyloco'
             'ccaceae|g__Staphylococcus|s__Staphylococcus_epidermidis|t__Staph'
             'ylococcus_epidermidis'): 54.60581,
            ('k__Bacteria|p__Firmicutes|c__Bacilli|o__Bacillales|f__Bacillacea'
             'e|g__Bacillus|s__Bacillus_megaterium|t__Bacillus_megaterium'
             ): 1.24861,
            ('k__Bacteria|p__Firmicutes|c__Bacilli|o__Bacillales|f__Staphyloco'
             'ccaceae|g__Staphylococcus|s__Staphylococcus_lugdunensis|t__Staph'
             'ylococcus_lugdunensis'): 1.93655,
            ('k__Bacteria|p__Firmicutes|c__Bacilli|o__Bacillales|f__Staphyloco'
             'ccaceae|g__Staphylococcus|s__Staphylococcus_hominis|t__Staphyloc'
             'occus_hominis'): 10.61197,
            ('k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetal'
             'es|f__Propionibacteriaceae|g__Propionibacterium|s__Propionibacte'
             'rium_acnes|t__Propionibacterium_acnes'): 0.58313,
            ('k__Bacteria|p__Proteobacteria|c__Betaproteobacteria|o__Burkholde'
             'riales|f__Burkholderiaceae|g__Burkholderia|s__Burkholderia'
             ): 0.1153,
            ('k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetal'
             'es|f__Propionibacteriaceae|g__Propionibacterium|s__Propionibacte'
             'rium_granulosum|t__Propionibacterium_granulosum'
             ): 0.3842,
            ('k__Bacteria|p__Firmicutes|c__Bacilli|o__Bacillales|f__Staphyloco'
             'ccaceae|g__Staphylococcus|s__Staphylococcus_haemolyticus|t__Stap'
             'hylococcus_haemolyticus'): 0.44687,
            ('k__Bacteria|p__Firmicutes|c__Bacilli|o__Bacillales|f__Bacillacea'
             'e|g__Bacillus|s__Bacillus_subtilis|t__Bacillus_subtilis'
             ): 0.56501}
        self.file_mock_profile = get_data_path('mock_metaphlan.bacprofile')
        self.true_mock_profile = {
            ('k__Viruses|p__Viruses_noname|c__Viruses_noname|o__Caudovirales|f'
             '__Siphoviridae|g__Siphoviridae_noname|s__Mycobacterium_phage_Ome'
             'ga'): 0.005,
            ('k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Euba'
             'cteriaceae|g__Eubacterium|s__Eubacterium_cellulosolvens'
             ): 40.005,
            ('k__Viruses|p__Viruses_noname|c__Viruses_noname|o__Viruses_noname'
             '|f__Fuselloviridae|g__Alphafusellovirus|s__Sulfolobus_spindle_sh'
             'aped_virus_2'): 12.83,
            'k__Bacteria|p__Armatimonadetes': 34.98,
            ('k__Bacteria|p__Proteobacteria|c__Epsilonproteobacteria|o__Campyl'
             'obacterales|f__Helicobacteraceae|g__Helicobacter|s__Helicobacter'
             '_winghamensis'): 12.18}

    def test_read_ncbi_nodes(self):
        nodes = read_ncbi_nodes(self.file_nodes)
        self.assertEqual(self.true_nodes, nodes)

        with self.assertRaises(ValueError):
            read_ncbi_nodes(self.file_names)

        with self.assertRaises(IOError):
            read_ncbi_nodes('/tmp/non')

    def test_read_ncbi_merged(self):
        nodes = read_ncbi_merged(self.file_merged)
        self.assertEqual(self.true_merged, nodes)

        with self.assertRaises(ValueError):
            read_ncbi_merged(self.file_names)

        with self.assertRaises(IOError):
            read_ncbi_merged('/tmp/non')

    def test_read_metaphlan_markers_info(self):
        self.assertEqual(self.true_marker,
                         read_metaphlan_markers_info(self.file_mpmarkers))

        with self.assertRaises(IOError):
            read_metaphlan_markers_info('/tmp/non')

        self.assertEqual({}, read_metaphlan_markers_info(self.file_nodes))

    def test_read_taxid_list(self):
        self.assertEqual(self.true_mptaxids,
                         read_taxid_list(self.file_mptaxids))
        self.assertEqual(self.true_gg_taxids,
                         read_taxid_list(self.file_gg_taxids))

        with self.assertRaises(IOError):
            read_taxid_list('/tmp/non')

        with self.assertRaises(ValueError):
            read_taxid_list(self.file_names)

    def test__read_ncbitaxonomy_file(self):
        self.assertEqual(self.true_nodes,
                         _read_ncbitaxonomy_file(self.file_nodes))
        self.assertEqual(self.true_merged,
                         _read_ncbitaxonomy_file(self.file_merged))

        with self.assertRaises(ValueError):
            _read_ncbitaxonomy_file(self.file_names)

        with self.assertRaises(IOError):
            _read_ncbitaxonomy_file('/tmp/non')

    def test_read_gg_accessions(self):
        self.assertEqual(self.true_accessions,
                         read_gg_accessions(self.file_accessions))

        with self.assertRaises(IOError):
            read_gg_accessions('/tmp/non')

        with self.assertRaises(ValueError):
            read_gg_accessions(self.file_names)

    def test_read_gg_otu_map(self):
        gg_accessions = read_gg_accessions(self.file_accessions)

        self.assertEqual(self.true_gg_otumap,
                         read_gg_otu_map(self.file_gg_otumap, gg_accessions))

        with self.assertRaises(IOError):
            read_gg_otu_map('/tmp/non', gg_accessions)

        with self.assertRaises(ValueError):
            read_gg_otu_map(self.file_names, gg_accessions)

    def test_write_clade2otus_map(self):
        fh, filename = tempfile.mkstemp()
        write_clade2otus_map(filename, self.true_map)
        self.assertTrue(filecmp.cmp(filename, self.file_true_map))

    def test_read_clade2otus_map(self):
        self.assertCountEqual(read_clade2otus_map(self.file_true_map),
                              self.true_map)

    def test_read_metaphlan_profile(self):
        res = read_metaphlan_profile(self.file_example_profile)
        self.assertCountEqual(res, self.true_example_profile)
        self.assertNotIn('k__Bacteria', res)
        self.assertNotIn(('k__Bacteria|p__Firmicutes|c__Bacilli|o__Bacillales|'
                          'f__Bacillaceae|g__Bacillus'), res)
        s = sum(res.values())
        self.assertTrue(s <= 100)
        self.assertTrue(s > 0)

        res = read_metaphlan_profile(self.file_mock_profile)
        self.assertCountEqual(res, self.true_mock_profile)
        s = sum(res.values())
        self.assertTrue(s <= 100)
        self.assertTrue(s > 0)


if __name__ == '__main__':
    main()
