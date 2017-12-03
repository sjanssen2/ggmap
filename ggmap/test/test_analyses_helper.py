from unittest import TestCase, main
import shutil
import tempfile
import pandas as pd
from pandas.util.testing import assert_frame_equal

from skbio.util import get_data_path

from ggmap.analyses import (_parse_alpha_div_collated, _get_ref_phylogeny,
                            _parse_timing)


class AnalysesHelperTests(TestCase):
    def setUp(self):
        self.workdir = get_data_path(
            'analyses/rarefaction_curves/rare_workdir')
        self.samplenames = [
            '10315.APIGRA.112', '10315.F.C', '10315.Apalw18',
            '10315.APIGRA.103', '10315.APIGRA.97', '10315.APF116',
            '10315.APX.37', '10315.APX91', '10315.APX18', '10315.APX102']
        self.exp_metrics = ['observed_otus', 'PD_whole_tree', 'shannon']

    def test__parse_alpha_div_collated(self):
        obs = _parse_alpha_div_collated(self.workdir, self.samplenames)
        for metric in self.exp_metrics:
            exp = pd.read_csv(
                get_data_path(
                    'analyses/rarefaction_curves/exp_%s.tsv' % metric),
                sep="\t", index_col=0)
            exp.index = list(range(exp.shape[0]))
            assert_frame_equal(obs[metric], exp)

    def test__get_ref_phylogeny(self):
        self.assertEqual(_get_ref_phylogeny('testfile'), 'testfile')

        # with self.assertRaises(ValueError):
        #     _get_ref_phylogeny(env='wrong')

    def test__parse_timing(self):
        jobname = 'unittest'
        dir_tmp = tempfile.mkdtemp(prefix='ana_%s_' % jobname,
                                   dir=tempfile.gettempdir())
        file_timing = dir_tmp + '/cr_ana_%s.t' % jobname
        content = ['line1\n', 'line2\n']
        f = open(file_timing, 'w')
        f.write("".join(content))
        f.close()
        obs = _parse_timing(dir_tmp, jobname)
        shutil.rmtree(dir_tmp)
        self.assertEqual(obs, content)

        obs = _parse_timing('/dev/', jobname)
        self.assertEqual(obs, None)


if __name__ == '__main__':
    main()
