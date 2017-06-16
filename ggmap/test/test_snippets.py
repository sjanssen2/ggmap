from unittest import TestCase, main
import pandas as pd
import warnings
import tempfile
import biom

from skbio.util import get_data_path

from ggmap.snippets import (biom2pandas, pandas2biom, parse_splitlibrarieslog,
                            _repMiddleValues, _shiftLeft)


class ReadWriteTests(TestCase):
    def setUp(self):
        # suppress the warning caused by biom load_table.
        warnings.simplefilter('ignore', ResourceWarning)
        self.filename_minibiom = get_data_path('25x25.biom')
        self.index_minibiom = list(map(str,
                                       [518040, 4407475, 540982, 1008941,
                                        2397566, 4414257, 574471, 4339160,
                                        4316113, 4435370, 4432796, 919806,
                                        252547, 3746876, 4384802, 4434294,
                                        3567151, 1871, 1726408, 4387208,
                                        4435982, 4449851, 4456606, 158967,
                                        4408129]))
        self.columns_minibiom = ["weampp05E09", "weampp05E08", "weampp05E07",
                                 "weampp05E05", "weampp05E02", "weampp05E01",
                                 "weampp03G06", "weampp16F09", "weampp03H10",
                                 "weampp02F10", "weampp03F02", "weampp03F01",
                                 "weampp03F07", "weampp03F04", "weampp03F05",
                                 "weampp03F08", "weampp15D10", "weampp15H07",
                                 "weampp15H04", "weampp04B02", "weampp05A01",
                                 "weampp14C12", "weampp05F04", "weampp05B12",
                                 "weampp05B11"]

        self.filename_withtax = get_data_path('withTax.biom')
        self.index_withtax = [("CACCGGCAGCTCTAGTGGTAGCAGTTTTTATTGGGCCTAAAGCGTC"
                               "CGTAGCCGGTTTAATAAGTCTCTGGTGAAATCCTGCAGCTTAACTG"
                               "TGGGAATTGCTGGAGATACTATTAGACTTGAGATCGGGAGAGGTTA"
                               "GAGGTACTCCCA"),
                              ("TACGGAGGATGCGAGCGTTATCCGGATTTATTGGGTTTAAAGGGTG"
                               "CGTAGGTGGTGATTTAAGTCAGCGGTGAAAGTTTGTGGCTCAACCA"
                               "TAAAATTGCCGTTGAAACTGGGTTACTTGAGTGTGTTTGAGGTAGG"
                               "CGGAATGCGTGG"),
                              ("AACGTAGGTCACAAGCGTTGTCCGGAATTACTGGGTGTAAAGGGAG"
                               "CGCAGGCGGGAGAACAAGTTGGAAGTGAAATCCATGGGCTCAACCC"
                               "ATGAACTGCTTTCAAAACTGTTTTTCTTGAGTAGTGCAGAGGTAGG"
                               "CGGAATTCCCGG"),
                              ("TACGTAGGTGGCAAGCGTTGTCCGGATTTACTGGGTGTAAAGGGCG"
                               "TGTAGCCGGGAAGGCAAGTCAGATGTGAAATCCACGGGCTTAACTC"
                               "GTGAACTGCATTTGAAACTACTTTTCTTGAGTATCGGAGAGGCAAT"
                               "CGGAATTCCTAG"),
                              ("TACGGAGGATCCGAGCGTTATCCGGATTTATTGGGTTTAAAGGGAG"
                               "CGTAGGCGGACGCTTAAGTCAGTTGTGAAAGTTTGCGGCTCAACCG"
                               "TAAAATTGCAGTTGATACTGGGTGTCTTGAGTACAGTAGAGGCAGG"
                               "CGGAATTCGTGG"),
                              ("TACGGAGGATTCAAGCGTTATCCGGATTTATTGGGTTTAAAGGGTG"
                               "CGTAGGCGGTTAGATAAGTTAGAGGTGAAATCCCGGGGCTTAACTC"
                               "CGAAATTGCCTCTAATACTGTTTGACTAGAGAGTAGTTGCGGTAGG"
                               "CGGAATGTATGG"),
                              ("TACGTATGGTGCAAGCGTTATCCGGATTTACTGGGTGTAAAGGGTG"
                               "CGTAGGTGGTATGGCAAGTCAGAAGTGAAAGGCTGGGGCTCAACCC"
                               "CGGGACTGCTTTTGAAACTGTCAAACTAGAGTACAGGAGAGGAAAG"
                               "CGGAATTCCTAG"),
                              ("AACGTAGGTCACAAGCGTTGTCCGGAATTACTGGGTGTAAAGGGAG"
                               "CGCAGGCGGGAAGACAAGTTGGAAGTGAAATCTATGGGCTCAACCC"
                               "ATAAACTGCTTTCAAAACTGTTTTTCTTGAGTAGTGCAGAGGTAGG"
                               "CGGAATTCCCGG"),
                              ("AACGTAGGTCACAAGCGTTGTCCGGAATTACTGGGTGTAAAGGGAG"
                               "CGCAGGCGGGAAGACAAGTTGGAAGTGAAATCCATGGGCTCAACCC"
                               "ATGAACTGCTTTCAAAACTGTTTTTCTTGAGTAGTGCAGAGGTAGG"
                               "CGGAATTCCCGG"),
                              ("TACGGAAGGTCCGGGCGTTATCCGGATTTATTGGGTTTAAAGGGAG"
                               "CGTAGGCCGTGAGGTAAGCGTGTTGTGAAATGTAGGCGCCCAACGT"
                               "CTGCACTGCAGCGCGAACTGCCCCACTTGAGTGCGCGCAACGCCGG"
                               "CGGAACTCGTCG")]
        self.columns_withtax = ["10283.LS.1.14.2015", "10283.LS.9.24.2013",
                                "10283.LS.6.14.2015", "10283.LS.4.19.2015",
                                "10283.LS.7.11.2015", "10283.LS.11.16.2014",
                                "10283.LS.7.10.2015", "10283.LS.12.28.2011",
                                "10283.LS.4.7.2013", "10283.LS.8.22.2014"]
        self.families_withtax = ['f__Ruminococcaceae', 'f__Ruminococcaceae',
                                 'f__Ruminococcaceae',
                                 'f__Methanobacteriaceae', 'f__Prevotellaceae',
                                 'f__Bacteroidaceae', 'f__Porphyromonadaceae',
                                 'f__Rikenellaceae', 'f__Ruminococcaceae',
                                 'f__Lachnospiraceae']

        self.filename_float = get_data_path('float.biom')
        self.columns_float = ["10283.LS.7.19.2015_right",
                              "10283.LS.1.26.2013_right",
                              "10283.LS.4.4.2015_left",
                              "10283.LS.7.26.2015_right",
                              "10283.LS.8.22.2014_left",
                              "10283.LS.6.16.2014_left",
                              "10283.LS.9.28.2014_left",
                              "10283.LS.11.16.2014_right",
                              "10283.LS.8.7.2012_right",
                              "10283.LS.2.17.2014_left"]
        self.index_float = ["Two-component system", "Transcription factors",
                            "Pyrimidine metabolism", "Peptidases",
                            "Purine metabolism", "Ribosome",
                            "DNA repair and recombination proteins",
                            "ABC transporters",
                            "General function prediction only", "Transporters"]

    def test_biom2pandas_minibiom(self):
        with self.assertRaises(IOError):
            biom2pandas('/dev')
        with self.assertRaises(IOError):
            biom2pandas('/tmp/non')

        b = biom2pandas(self.filename_minibiom)
        self.assertCountEqual(b.index, self.index_minibiom)
        self.assertCountEqual(b.columns, self.columns_minibiom)
        self.assertEqual(b.sum().sum(), 45456)

    def test_biom2pandas_withTax(self):
        b, t = biom2pandas(self.filename_withtax, withTaxonomy=True)
        self.maxDiff = None
        self.assertCountEqual(b.index, self.index_withtax)
        self.assertCountEqual(b.columns, self.columns_withtax)
        self.assertEqual(b.sum().sum(), 100273)

        with self.assertRaisesRegex(ValueError,
                                    'No taxonomy information found in biom f'):
            b, t = biom2pandas(self.filename_minibiom, withTaxonomy=True)

        with self.assertRaisesRegex(ValueError, "too many values to unpack"):
            b, t = biom2pandas(self.filename_withtax, withTaxonomy=False)

        b, t = biom2pandas(self.filename_withtax, withTaxonomy=True)
        self.assertCountEqual(t['family'].values, self.families_withtax)

    def test_biom2pandas_float(self):
        b = biom2pandas(self.filename_float, astype=int)
        self.assertEqual(b.sum().sum(), 0)

        b = biom2pandas(self.filename_float, astype=float)
        self.assertAlmostEqual(b.sum().sum(), 2.69624884712, places=5)

    def test_pandas2biom(self):
        fh, filename = tempfile.mkstemp()
        p = pd.read_csv(get_data_path('float.tsv'), sep='\t', index_col=0)
        with self.assertRaisesRegex(IOError, 'Cannot write to file'):
            pandas2biom('/dev/', p)
        pandas2biom(filename, p)
        b = biom.load_table(filename)
        self.assertCountEqual(b.ids(), p.columns)
        self.assertCountEqual(b.ids(axis='observation'), p.index)

    def test_parse_splitlibrarieslog(self):
        with self.assertRaisesRegex(IOError, 'Cannot read file'):
            parse_splitlibrarieslog('/dev/')
        c = parse_splitlibrarieslog(get_data_path('split_library_log_2p.txt'))
        self.assertEqual(c['counts'].sum(), 86167277)

    def test__repMiddleValues(self):
        self.assertEqual([1, 1, 2, 2, 3, 3, 4, 4],
                         _repMiddleValues([1, 2, 3, 4]))
        self.assertEqual(['a', 'a', 'b', 'b'],
                         _repMiddleValues(['a', 'b']))

    def test__shiftLeft(self):
        self.assertEqual(_shiftLeft([1, 2, 3]), [2, 3, 4])


if __name__ == '__main__':
    main()
