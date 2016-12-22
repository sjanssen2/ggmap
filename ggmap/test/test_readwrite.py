from unittest import TestCase, main

from skbio.util import get_data_path

from ggmap.readwrite import read_ncbi_nodes


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

    def test_read_ncbi_nodes(self):
        nodes = read_ncbi_nodes(self.file_nodes)
        self.assertEqual(self.true_nodes, nodes)

        with self.assertRaises(ValueError):
            read_ncbi_nodes(self.file_names)

        with self.assertRaises(IOError):
            read_ncbi_nodes('/tmp/non')

if __name__ == '__main__':
    main()
