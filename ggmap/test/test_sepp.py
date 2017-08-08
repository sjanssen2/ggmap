from unittest import TestCase, main

from skbio.util import get_data_path

from ggmap.sepp import (read_otumap)


class SeppTests(TestCase):
    def test_read_otumap(self):
        with self.assertRaisesRegex(IOError, "Cannot read file"):
            read_otumap('/dev/aucguacguguac')

        obs = read_otumap(get_data_path('61_otu_map.txt'))
        self.assertEqual(obs.shape, (22, ))
        self.assertEqual(obs.index.name, 'representative')
        self.assertEqual(obs.name, 'non-representatives')
        self.assertIn('229854', obs.index)
        self.assertEqual(sorted(obs['229854']),
                         sorted(['2014493', '951205', '734152']))
        self.assertIn('2107103', obs.index)
        self.assertEqual(obs['2107103'], [])


if __name__ == '__main__':
    main()
