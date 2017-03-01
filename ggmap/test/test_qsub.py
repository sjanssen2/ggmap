from unittest import TestCase, main
import pandas as pd
import warnings
import tempfile
import biom

from skbio.util import get_data_path

from ggmap.snippets import cluster_run


class QsubTests(TestCase):
    def test_cluster_run_exceptions(self):
        with self.assertRaisesRegex(ValueError, "You need to set a jobname!"):
            cluster_run([], None, "/tmp/newfile_acgjkdh")
        with self.assertRaisesRegex(ValueError,
                                    "You need to set non empty jobname!"):
            cluster_run([], "", "/tmp/newfile_acgjkdh")
        with self.assertRaisesRegex(ValueError, "Conda environment"):
            cluster_run([], "jobname", "/tmp/newfile_acgjkdh",
                        environment='envNotThere')
        with self.assertRaisesRegex(ValueError,
                                    "You need to specify a result path."):
            cluster_run([], "jobname", None)
        with self.assertRaisesRegex(ValueError, " is not writable!"):
            cluster_run([], "jobname", "/dev/null")
        with self.assertRaisesRegex(ValueError, "commands contain a ' char. "):
            cluster_run("find /tmp/ -name '*.png'", "myjob",
                        "/tmp/newfile_acgjkdh", environment="qiime_env")

    def test_cluster_run(self):
        res = cluster_run([], "jobname", get_data_path("float.tsv"))
        self.assertEqual(res, 'Result already present!')

if __name__ == '__main__':
    main()
