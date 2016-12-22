from unittest import TestCase, main
from io import StringIO

from skbio.util import get_data_path

from ggmap.readwrite import read_ncbi_nodes
from ggmap.tree import get_lineage, build_ncbi_tree


class ContactsTests(TestCase):
    def setUp(self):
        self.file_nodes = get_data_path('top_nodes.dmp')
        self.file_nodes_head = get_data_path('head_nodes.dmp')
        self.taxonomy = read_ncbi_nodes(self.file_nodes)

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

if __name__ == '__main__':
    main()
