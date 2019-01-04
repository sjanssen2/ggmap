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
        err = StringIO()
        res = cluster_run([], "jobname", get_data_path("float.tsv"), err=err)
        self.assertEqual(res, 'Result already present!')
        self.assertEqual(err.getvalue(), 'jobname already computed\n')

    def test_cluster_run_slurm(self):
        exp = """CONTENT OF /tmp/tmp7purx0c8.slurm.sh:
#!/bin/bash

#SBATCH --job-name=cr_jobname
#SBATCH --output=/home/sjanssen/GreenGenes/ggmap/%A.log
#SBATCH --partition=hii02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=0-4:0
#SBATCH --array=1-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sjanssen@ucsd.edu

srun uname -a
srun find /tmp/ -name *.png


sbatch /tmp/tmp7purx0c8.slurm.sh
"""

        def _cleanHome(text):
            """A helper function to remove absolute home dir information."""
            out = []
            for i, line in enumerate(text.split('\n')):
                # skip first two lines, because they are only a information
                # for the user if run dry
                if i == 0:
                    continue
                if line.startswith('#SBATCH --output='):
                    out.append('#SBATCH --output=%A.log')
                elif line.startswith('sbatch'):
                    out.append('sbatch\n')
                else:
                    out.append(line)
            return "\n".join(out)

        out = StringIO()
        cluster_run("find /tmp/ -name *.png", "jobname",
                    "/tmp/teststxxx", force_slurm=True, out=out, err=None)
        self.assertEqual(_cleanHome(out.getvalue()), _cleanHome(exp))

    def test_cluster_run_highmem(self):
        out = StringIO()
        cluster_run("find /tmp/ -name *.png", "jobname",
                    "/tmp/teststxxx", out=out, pmem='1000GB', err=None)
        self.assertIn(':highmem:', out.getvalue())

    def test_cluster_timing(self):
        out = StringIO()
        cluster_run("find /tmp/ -name *.png", "jobname",
                    "/tmp/teststxxx", out=out, timing=True, err=None)
        self.assertIn('uname -a', out.getvalue())
        self.assertIn('time -v -o', out.getvalue())

    # not possible to test unless I find a way of having multiple conda envs
    # in Travis
    # def test_cluster_run_env(self):
    #     out = StringIO()
    #     cluster_run("find /tmp/ -name *.png", "jobname",
    #                 "/tmp/teststxxx", out=out, pmem='1000GB',
    #                 environment="qiime_env")
    #     self.assertIn('source activate qiime_env && ', out.getvalue())


if __name__ == '__main__':
    main()
