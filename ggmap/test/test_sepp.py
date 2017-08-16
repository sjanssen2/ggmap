from unittest import TestCase, main
from io import StringIO
from tempfile import mkstemp
from os import remove

from skbio.util import get_data_path

from ggmap.sepp import (read_otumap, load_sequences_pynast, add_mutations)


class SeppTests(TestCase):
    def setUp(self):
        self.file_pynast = get_data_path('gg_13_5_pynast.fasta')
        self.file_otumap = get_data_path('61_otu_map.txt')
        self.exp_fragments = [
            {'sequence':
             ('TACGAAGGGACCTAGCGTAGTTCGGAATTACTGGGCTTAAAGAGTTCGTAGGTGGTTAAAAAA'
              'GTTGGTGGTGAAAGCCCAGAGCTTAACTCTGGAACGGCCATCAAAACTTTTTAGCTAGAGTAT'
              'GATAGAGGAAAGCAGAATTTCTAG'),
             'OTUID': '4336814'},
            {'sequence':
             ('TACGGAGGTGGCACGCGTTGTCCGGAATTATTGGGCGTAAAGCGCGCGGGTGCCCGCACTAGC'
              'TGGAGTGGGCGTATAGCTCGCGATGGTGGTTCCTTAAGTGTGATGTGAAAAGCTCCCGGCTCA'
              'ACCGGGGAGAGTCATTGGAACTGG'),
             'OTUID': '734152'},
            {'sequence':
             ('AACCAGCTCTTCAAGTGGTCGGGATAATTATTGGGCTTAAAGTGTCCGTAGCTTGTATAATAA'
              'GTTCCTGGTAAAATCTAATAGCTTAACTATNAGTATGCTAGGAATACTGTTGTACTAGAGGGC'
              'GGGAGAGGTCTGAGGTACTTCAGG'),
             'OTUID': '1928988'},
            {'sequence':
             ('GACATAGGTCGCGAACGTTATCCGGAATTATTGGGCGTAAAGGATGCGTAGATGGTTCAGTAA'
              'GTTACTGGTGGGAAATCGAGGCCTAACCTCGTGGAAGTCAGTAATACTGTTGAACTTGAGTGC'
              'AGGAGAGGTTAACGGAACTTCATG'),
             'OTUID': '696036'},
            {'sequence':
             ('TACGGAGGGGGCAAGCGTTGTCGGAATAACTGGGCCTAAAGCCGCGCCGTAGGCGGGTTTGTT'
              'AAGTCAGATGTGAAAGCCCTCGGCTCAACCGGGGACGTGGCATTTGAACTGGCCAACTTGAGT'
              'ACTGGAGGGGGGGGGAATCCCGTG'),
             'OTUID': '4459468'},
            {'sequence':
             ('CACCGGCGGCCCGAGTGGTGACCGTTATTATTGGGTCTAAAGGGTCCGTAGCCGGTTTGGTCA'
              'GTCCTCCGGGAAATCTGATAGCTTAACTGTTAGGCTTTCGGGGGATACTGCCAGGCTTGGAAC'
              'CGGGAGAGGTAAGAGGTACTACAG'),
             'OTUID': '1128285'},
            {'sequence':
             ('GAACCTCGGCTCGAGTGGTGGCCGCTTTTATTGGGCTTAAAGCGTTCGTAGCTGGGTTGTTAA'
              'GTCTCTTGGGAAATCTGGCGGCTTAACCGTCAGGCGTCTAAGGGATACTGGCAATCTTGGAAC'
              'CGGGAGAGGTGAGGGGTACTTCGG'),
             'OTUID': '4455990'}]

    def test_read_otumap(self):
        with self.assertRaisesRegex(IOError, "Cannot read file"):
            read_otumap('/dev/aucguacguguac')

        obs = read_otumap(self.file_otumap)[0]
        self.assertEqual(obs.shape, (22, ))
        self.assertEqual(obs.index.name, 'representative')
        self.assertEqual(obs.name, 'non-representatives')
        self.assertIn('229854', obs.index)
        self.assertEqual(sorted(obs['229854']),
                         sorted(['2014493', '951205', '734152']))
        self.assertIn('2107103', obs.index)
        self.assertEqual(obs['2107103'], [])

    def test_load_sequences_pynast(self):
        out = StringIO()
        obs = load_sequences_pynast(self.file_pynast,
                                    self.file_otumap,
                                    2263, 3794, 150, out=out,
                                    cache_verbose=False)
        obs_out = out.getvalue()
        self.assertIn("%i rows and %i cols in alignment '%s'" %
                      (9, 7682, self.file_pynast.split('/')[-1]), obs_out)
        self.assertIn("%i sequences in OTU map '%s'" %
                      (33, self.file_otumap.split('/')[-1]), obs_out)
        self.assertIn(("%i sequences selected from OTU map and alignment. "
                       "Surprise: %i sequences of OTU map are NOT in "
                       "alignment!") %
                      (9, 24), obs_out)
        self.assertIn(("%i -> %i cols: trimming alignment to fragment "
                       "coordinates") %
                      (7682, 1531), obs_out)
        self.assertIn(("%i fragments with ungapped length >= %int. "
                       "Surprise: %i fragments are too short and %i "
                       "fragments where too long "
                       "(and have been trimmed)!") %
                      (7, 150, 2, 3), obs_out)
        self.assertCountEqual(obs, self.exp_fragments)

    def test_load_sequences_pynast_testcache(self):
        file_dummy = mkstemp()[1]
        remove(file_dummy)

        # first run: create cache file
        err = StringIO()
        obs = load_sequences_pynast(self.file_pynast,
                                    self.file_otumap,
                                    2263, 3794, 150, cache_filename=file_dummy,
                                    out=StringIO(), cache_err=err)
        self.assertIn('stored results in cache "%s"' % file_dummy,
                      err.getvalue())
        self.assertCountEqual(obs, self.exp_fragments)

        # second run: load cached results
        err = StringIO()
        obs = load_sequences_pynast(self.file_pynast,
                                    self.file_otumap,
                                    2263, 3794, 150, cache_filename=file_dummy,
                                    out=StringIO(), cache_err=err)
        self.assertIn('retrieved results from cache "%s"' % (file_dummy),
                      err.getvalue())
        self.assertCountEqual(obs, self.exp_fragments)

    def test_add_mutations(self):
        def _get_num_snips(seqA, seqB):
            return sum([1 for (a, b) in zip(seqA, seqB) if a != b])

        out = StringIO()
        err = StringIO()
        obs = add_mutations(self.exp_fragments, out=out, err=err,
                            cache_verbose=False)
        self.assertIn(("%i fragments to start with") % (7),
                      out.getvalue())
        self.assertIn(("%i fragments after collapsing by sequence") % (7),
                      out.getvalue())
        self.assertIn(("%i fragments generated with 0 to "
                       "%i point mutations.") % (77, 10),
                      out.getvalue())
        self.assertEqual(len(obs), 77)

        self.assertEqual(_get_num_snips(obs[0]['sequence'],
                                        obs[0]['sequence']),
                         obs[0]['num_pointmutations'])
        self.assertEqual(_get_num_snips(obs[0]['sequence'],
                                        obs[5]['sequence']),
                         obs[5]['num_pointmutations'])

    def test_add_mutations_testcache(self):
        err = StringIO()
        out = StringIO()
        file_dummy = mkstemp()[1]
        remove(file_dummy)

        # first run: create cache file
        obs = add_mutations(self.exp_fragments, verbose=False, out=out,
                            cache_err=err, cache_filename=file_dummy)
        self.assertIn('stored results in cache "%s"' % file_dummy,
                      err.getvalue())
        self.assertEqual(len(obs), 77)

        # second run: load cached results
        out = StringIO()
        obs = add_mutations(self.exp_fragments, verbose=False, out=out,
                            cache_err=err, cache_filename=file_dummy)
        self.assertIn('retrieved results from cache "%s"' %
                      (file_dummy), err.getvalue())
        self.assertEqual(len(obs), 77)


if __name__ == '__main__':
    main()
