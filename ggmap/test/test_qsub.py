from unittest import TestCase, main
from io import StringIO

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

        # test if exception is raised early if qid file cannot be written
        with self.assertRaisesRegex(ValueError, "Cannot write qid file"):
            cluster_run("find /tmp/ -name *.png", "myjob",
                        "/tmp/newfile_acgjkdh", file_qid='/dev/null')

    def test_cluster_run(self):
        res = cluster_run([], "jobname", get_data_path("float.tsv"))
        self.assertEqual(res, 'Result already present!')

    def test_cluster_run_slurm(self):
        exp = """#!/bin/bash

#SBATCH --job-name=cr_jobname
#SBATCH --output=/home/sjanssen/GreenGenes/ggmap/%A.log
#SBATCH --partition=hii02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=0-4:0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sjanssen@ucsd.edu

srun uname -a
srun find /tmp/ -name *.png


"""
        out = StringIO()
        cluster_run("find /tmp/ -name *.png", "jobname",
                    "/tmp/teststxxx", slurm=True, out=out)
        self.assertEqual(out.getvalue(), exp)

    def test_cluster_run_highmem(self):
        out = StringIO()
        cluster_run("find /tmp/ -name *.png", "jobname",
                    "/tmp/teststxxx", out=out, pmem='1000GB')
        self.assertIn(':highmem:', out.getvalue())

    def test_cluster_run_env(self):
        out = StringIO()
        cluster_run("find /tmp/ -name *.png", "jobname",
                    "/tmp/teststxxx", out=out, pmem='1000GB',
                    environment="qiime_env")
        self.assertIn('source activate qiime_env && ', out.getvalue())


if __name__ == '__main__':
    main()
