from unittest import TestCase, main
import shutil
import tempfile

from skbio.util import get_data_path

from ggmap.analyses import (_parse_alpha_div_collated, _get_ref_phylogeny,
                            _parse_timing)


class AnalysesHelperTests(TestCase):
    def setUp(self):
        self.file_collated = get_data_path('collated_PD_whole_tree.txt')

    def test__parse_alpha_div_collated(self):
        exp = _parse_alpha_div_collated(self.file_collated)
        self.assertEqual(exp.shape, (22071, 3))
        self.assertCountEqual(list(exp.columns),
                              ['rarefaction depth',
                               'sample_name',
                               'collated_PD_whole_tree'])

        # check that metric name can be passed
        exp = _parse_alpha_div_collated(self.file_collated,
                                        metric='PD_whole_tree')
        self.assertCountEqual(list(exp.columns),
                              ['rarefaction depth',
                               'sample_name',
                               'PD_whole_tree'])

        with self.assertRaisesRegex(IOError, 'Cannot read file'):
            _parse_alpha_div_collated('/dev/null/test')

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
