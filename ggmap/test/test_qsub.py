from unittest import TestCase, main
import pandas as pd
import warnings
import tempfile
import biom

from skbio.util import get_data_path

from ggmap.snippets import cluster_run


class QsubTests(TestCase):
    def test_cluster_run(self):
        with self.assertRaisesRegex(ValueError, "You need to set a jobname!"):
            cluster_run([], None, None)
        with self.assertRaisesRegex(ValueError,
                                    "You need to set non empty jobname!"):
            cluster_run([], "", None)

if __name__ == '__main__':
    main()
