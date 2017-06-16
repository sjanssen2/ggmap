from unittest import TestCase, main
from io import StringIO
import pickle

from skbio.util import get_data_path

from ggmap.tree import _get_otus_from_clade


class TreeFullTests(TestCase):
    def setUp(self):
        self.file_pickled_trees = get_data_path('trees.pickle')

        pkl_file = open(self.file_pickled_trees, 'rb')
        self.tree_mp, self.tree_gg = pickle.load(pkl_file)
        pkl_file.close()

    def test_mapping(self):
        # no match
        clade = 's__Homalodisca_vitripennis_reovirus'
        out = StringIO()
        res = _get_otus_from_clade(clade,
                                   self.tree_mp, 'mp_clades',
                                   self.tree_gg, 'otus', out=out)
        self.assertEqual(res, set())
        self.assertIn(("'%s' not a cellular organism" % clade), out.getvalue())

        # positive match, two iterations
        clade = 's__Thioalkalivibrio_sp_ALE12'
        out = StringIO()
        res = _get_otus_from_clade(clade,
                                   self.tree_mp, 'mp_clades',
                                   self.tree_gg, 'otus', out=out)
        self.assertCountEqual({71169, 563169, 782597, 344620, 799885, 658668,
                               553455, 4478544, 339792, 759515, 1110174},
                              res)
        self.assertEqual("", out.getvalue())

        # positive match, one iteration
        clade = 's__Dickeya_dadantii'
        out = StringIO()
        res = _get_otus_from_clade(clade,
                                   self.tree_mp, 'mp_clades',
                                   self.tree_gg, 'otus', out=out)
        self.assertCountEqual({568033, 4401857, 794948, 4475144, 2775244,
                               555725, 693166, 93777, 775089, 763313, 4336055,
                               263449, 707770, 9787},
                              res)
        self.assertEqual("", out.getvalue())

        # positive match, many iterations
        clade = 'f__Leptosphaeriaceae'
        out = StringIO()
        res = _get_otus_from_clade(clade,
                                   self.tree_mp, 'mp_clades',
                                   self.tree_gg, 'otus', out=out)
        self.assertCountEqual({4457889}, res)
        self.assertEqual("", out.getvalue())


if __name__ == '__main__':
    main()
