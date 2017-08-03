from unittest import TestCase, main

from skbio.util import get_data_path

from ggmap.analyses import (_parse_alpha_div_collated)


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

if __name__ == '__main__':
    main()
