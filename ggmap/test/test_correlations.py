from unittest import TestCase, main
from io import StringIO
import pandas as pd

from skbio.util import get_data_path
import matplotlib.pyplot as plt

from ggmap.correlations import correlate_metadata

plt.switch_backend('Agg')
plt.rc('font', family='DejaVu Sans')


class CorrelationsTests(TestCase):
    def setUp(self):
        self.metadata = pd.read_csv(get_data_path('Correlations/meta_sel.tsv'), sep='\t', dtype=str, index_col=0)

    def test_correlate_metadata(self):
        # check that error is raised if columns are used more than once
        with self.assertRaisesRegex(ValueError, 'You have repeatedly used metadata'):
            correlate_metadata(self.metadata, categorials=['bird_family'], ordinals={'bird_family': None}, intervals=['bird_family'])
        with self.assertRaisesRegex(ValueError, 'You have repeatedly used metadata'):
            correlate_metadata(self.metadata, categorials=['bird_family'], ordinals={'bird_family': None}, intervals=[])

        # check that all used columns are in metadata
        with self.assertRaisesRegex(ValueError, 'The column\(s\) "agee" is\/are not in the metadata\!'):
            correlate_metadata(self.metadata, categorials=['bird_family'], ordinals={'agee': None}, intervals=[])
        with self.assertRaisesRegex(ValueError, 'The column\(s\) "agee", "kurt" is\/are not in the metadata\!'):
            correlate_metadata(self.metadata, categorials=['bird_family'], ordinals={'agee': None}, intervals=['kurt'])

        with self.assertRaisesRegex(ValueError, "You need to specify at least 2 columns!"):
            correlate_metadata(self.metadata, categorials=[], ordinals={'age': ['AD', 'HY ?', 'IM']}, intervals=[])

        with self.assertRaisesRegex(ValueError, 'Not all values in column "age" can be interpreted as floats!'):
            correlate_metadata(self.metadata, categorials=['bird_family'], ordinals={}, intervals=['age'])

        with self.assertRaisesRegex(ValueError, '"ordinals" need to be a dictionary!'):
            correlate_metadata(self.metadata, categorials=[], ordinals={'age'}, intervals=[])

        with self.assertRaisesRegex(ValueError, '"dates" need to be a dictionary!'):
            correlate_metadata(self.metadata, categorials=[], ordinals={}, dates=[], intervals=[])

        with self.assertRaisesRegex(ValueError, 'Mapping for ordinal "age" must be either None or a list but not a "<class \'str\'>"!'):
            correlate_metadata(self.metadata, categorials=['bird_family'], ordinals={'age': 'AD'}, intervals=[])

        with self.assertRaisesRegex(ValueError, "time data '05\/23\/1999' does not match format '\%d\/\%Y'"):
            correlate_metadata(self.metadata, categorials=['bird_family'], ordinals={}, dates={'collection_timestamp': "%d/%Y"}, intervals=[])

        e = StringIO()
        x = correlate_metadata(
            self.metadata,
            ordinals={'age': ['AD', 'AHY', 'DNY', 'JUV'], 'plate': None},
            categorials=['bird_family', 'bird_order','diet_brief', 'q2', 'sex', 'sample_substance', 'smj_genus', 'smj_species'],
            dates={'collection_timestamp': "%m/%d/%Y"},
            intervals=['latitude', 'longitude', 'weight', 'collection_second'],
            err=e)
        self.assertEqual(e.getvalue(), 'Ordinal "age" does not specify label(s) "HY", "HY ?", "IM", "U".\n')

        #should be OK
        correlate_metadata(self.metadata, categorials=['bird_family'], ordinals={'age': None}, intervals=[])
        correlate_metadata(self.metadata, categorials=['bird_family'], ordinals={}, intervals=['latitude'])

if __name__ == '__main__':
    main()
