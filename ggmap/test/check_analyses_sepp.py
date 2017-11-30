from unittest import TestCase, main
import pandas as pd
from io import StringIO
from pandas.util.testing import assert_frame_equal

from skbio.util import get_data_path
from skbio.tree import TreeNode

from ggmap.analyses import sepp, QIIME2_ENV


class SeppTests(TestCase):
    def setUp(self):
        self.file_ref_phylo = get_data_path(
            'analyses/sepp/reference_phylogeny_small.qza')
        self.file_ref_aln = get_data_path(
            'analyses/sepp/reference_alignment_small.qza')
        self.fragments = pd.read_csv(get_data_path(
            'analyses/sepp/fragments.tsv'), sep='\t', index_col=0)

        self.exp_taxonomy = pd.read_csv(get_data_path(
            'analyses/sepp/exp_taxonomy.tsv'), sep='\t', index_col=0)
        self.exp_tree = TreeNode.read(get_data_path(
            'analyses/sepp/exp_tree.nwk'))

    def test_sepp_onechunk(self):
        obs_sepp = sepp(
            self.fragments,
            chunksize=1000,
            reference_phylogeny=self.file_ref_phylo,
            reference_alignment=self.file_ref_aln,
            dry=False,
            nocache=True,
            dirty=False,
            ppn=20,
            use_grid=False)

        assert_frame_equal(obs_sepp['results']['taxonomy'].sort_index(),
                           self.exp_taxonomy.sort_index())

        obs_tips = {tip.name for tip in TreeNode.read(
            StringIO(obs_sepp['results']['tree'])).tips()}
        exp_tips = {tip.name for tip in self.exp_tree.tips()}
        self.assertCountEqual(obs_tips, exp_tips)

    def test_sepp_10chunk(self):
        obs_sepp = sepp(
            self.fragments,
            chunksize=10,
            reference_phylogeny=self.file_ref_phylo,
            reference_alignment=self.file_ref_aln,
            dry=False,
            nocache=True,
            dirty=False,
            ppn=20,
            use_grid=False)

        assert_frame_equal(obs_sepp['results']['taxonomy'].sort_index(),
                           self.exp_taxonomy.sort_index())

        obs_tips = {tip.name for tip in TreeNode.read(
            StringIO(obs_sepp['results']['tree'])).tips()}
        exp_tips = {tip.name for tip in self.exp_tree.tips()}
        self.assertCountEqual(obs_tips, exp_tips)


if __name__ == '__main__':
    main()
