from unittest import TestCase, main
from ggmap.analyses import demultiplex
from contextlib import redirect_stderr
import shutil
import io

class DemuxTests(TestCase):
    def setUp(self):
        self.orig_tmp_dir = tempfile.tempdir
        tempfile.tempdir = os.path.join(tempfile.tempdir, 'demux_unit_tests')
        os.makedirs(tempfile.tempdir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(tempfile.tempdir)
        tempfile.tempdir = self.orig_tmp_dir

    def test_prechecks(self):
        with self.assertRaisesRegex(ValueError,
                                    ('Provide fps_illuminadata as list, not single str!')):
            _ = demultiplex('/vol/jlab/Sequencing/Illumina/260303_VH01369_55_AAF2G2THV.tar.gz',
                            '', './',
                            dry=True, use_grid=False, environment=None)

        with self.assertRaisesRegex(ValueError,
                                    ('Provide fps_demuxsheet as list, not single str!')):
            _ = demultiplex(['/vol/jlab/Sequencing/Illumina/260303_VH01369_55_AAF2G2THV.tar.gz'],
                            '/vol/jlab/Sequencing/Samplesheets/260303_VH01369_55_AAF2G2THV.csv', './',
                            dry=True, use_grid=False, environment=None)
        with self.assertRaisesRegex(ValueError,
                                    ('Numbers mismatch')):
            _ = demultiplex(['/vol/jlab/Sequencing/Illumina/260303_VH01369_55_AAF2G2THV.tar.gz'],
                            ['/vol/jlab/Sequencing/Samplesheets/260303_VH01369_55_AAF2G2THV.csv', ''], './',
                            dry=True, use_grid=False, environment=None)
        with self.assertRaisesRegex(ValueError,
                                    (' does not exist!')):
            _ = demultiplex(
                ['/vol/jlab/5_AAF-broke-2G2THV.tar.gz',
                 '/vol/jlab/Sequencing/Illumina/200512_NB552271_0007_AHHJNVBGXC.7z'],
                ['/vol/jlab/Sequencing/Samplesheets/260303_VH01369_55_AAF2G2THV.csv',
                 '/vol/jlab/Sequencing/Samplesheets/200512_NB552271_0007_AHHJNVBGXC.csv'], './',
                dry=True, use_grid=False, environment=None)
        with self.assertRaisesRegex(ValueError,
                                    (' does not exist!')):
            _ = demultiplex(
                ['/vol/jlab/Sequencing/Illumina/260303_VH01369_55_AAF2G2THV.tar.gz',
                 '/vol/jlab/Sequencing/Illumina/200512_NB552271_0007_AHHJNVBGXC.7z'],
                ['/vol/jla_AAF-broke-2G2THV.csv',
                 '/vol/jlab/Sequencing/Samplesheets/200512_NB552271_0007_AHHJNVBGXC.csv'], './',
                dry=True, use_grid=False, environment=None)
        with self.assertRaisesRegex(ValueError,
                                    (' does not look like a typical Illumina demultiplexing sheet. Please double check. If you are certain it is correct, swit')):
            _ = demultiplex(['/vol/jlab/Sequencing/Illumina/260303_VH01369_55_AAF2G2THV.tar.gz',
                             '/vol/jlab/Sequencing/Illumina/200512_NB552271_0007_AHHJNVBGXC.7z'],
                            ['%s/.bashrc' % os.environ['HOME'],
                             '/vol/jlab/Sequencing/Samplesheets/200512_NB552271_0007_AHHJNVBGXC.csv'], './',
                            dry=True, use_grid=False, environment=None)

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            _ = demultiplex(['/vol/jlab/Sequencing/Illumina/260303_VH01369_55_AAF2G2THV.tar.gz',
                             '/vol/jlab/Sequencing/Illumina/200512_NB552271_0007_AHHJNVBGXC.7z'],
                            ['%s/.bashrc' % os.environ['HOME'],
                             '/vol/jlab/Sequencing/Samplesheets/200512_NB552271_0007_AHHJNVBGXC.csv'], './',
                            ignore_warnings=True,
                            dry=True, use_grid=False, verbose=None, environment=None)
            self.assertIn("does not look like a typical Illumina demultiplexing sheet", stderr.getvalue())

        with self.assertRaisesRegex(ValueError,
                                    ('not look like a typical Illumina raw data directory. Please double check. If you are certain it is correct, swit')):
            r = demultiplex([os.environ['HOME'],
                             '/vol/jlab/Sequencing/Illumina/200512_NB552271_0007_AHHJNVBGXC.7z'],
                            ['/vol/jlab/Sequencing/Samplesheets/260303_VH01369_55_AAF2G2THV.csv',
                             '/vol/jlab/Sequencing/Samplesheets/200512_NB552271_0007_AHHJNVBGXC.csv'], './',
                            dry=True, use_grid=False, verbose=None, environment=None)

    def test__write(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            r = demultiplex([os.environ['HOME'],
                             '/vol/jlab/Sequencing/Illumina/200512_NB552271_0007_AHHJNVBGXC.7z'],
                            ['/vol/jlab/Sequencing/Samplesheets/260303_VH01369_55_AAF2G2THV.csv',
                             '/vol/jlab/Sequencing/Samplesheets/200512_NB552271_0007_AHHJNVBGXC.csv'], './',
                            ignore_warnings=True,
                            dry=True, use_grid=False, verbose=None, environment=None)
            self.assertIn("not look like a typical Illumina raw data directory.", stderr.getvalue())

        with self.assertRaisesRegex(ValueError, ('flags for use_bclconvert!')):
            r = demultiplex(['/vol/jlab/Sequencing/Illumina/260303_VH01369_55_AAF2G2THV.tar.gz',
                             '/vol/jlab/Sequencing/Illumina/200512_NB552271_0007_AHHJNVBGXC.7z'],
                            ['/vol/jlab/Sequencing/Samplesheets/260303_VH01369_55_AAF2G2THV.csv',#'%s/.bashrc' % ,
                             '/vol/jlab/Sequencing/Samplesheets/200512_NB552271_0007_AHHJNVBGXC.csv'], './', use_bclconvert=[True],
                            dry=True, use_grid=False, verbose=None, environment=None)

        with self.assertRaisesRegex(ValueError, ('No write permission in')):
            r = demultiplex(['/vol/jlab/Sequencing/Illumina/260303_VH01369_55_AAF2G2THV.tar.gz',
                             '/vol/jlab/Sequencing/Illumina/200512_NB552271_0007_AHHJNVBGXC.7z'],
                            ['/vol/jlab/Sequencing/Samplesheets/260303_VH01369_55_AAF2G2THV.csv',#'%s/.bashrc' % ,
                             '/vol/jlab/Sequencing/Samplesheets/200512_NB552271_0007_AHHJNVBGXC.csv'], fp_output='/',
                            use_bclconvert=[True, False],
                            dry=True, use_grid=False, environment=None, verbose=None)

    def test__demux_decompression(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            _ = _demux_decompression('/vol/jlab/Sequencing/Illumina/200512_NB552271_0007_AHHJNVBGXC.7z', './', 0, verbose=sys.stderr)
            self.assertIn("Found a compressed archive (7zip)", stderr.getvalue())
            self.assertIn("will be automatically extract into", stderr.getvalue())
            self.assertIn("./200512_NB552271_0007_AHHJNVBGXC", stderr.getvalue())

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            _ = _demux_decompression('/vol/jlab/Sequencing/Illumina/260303_VH01369_55_AAF2G2THV.tar.gz', './', 0, verbose=sys.stderr)
            self.assertIn("Found a compressed archive (tgz)", stderr.getvalue())
            self.assertIn("will be automatically extract into", stderr.getvalue())
            self.assertIn("./260303_VH01369_55_AAF2G2THV", stderr.getvalue())

        temp_dir = tempfile.TemporaryDirectory()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            os.makedirs(os.path.join(temp_dir.name, '260303_VH01369_55_AAF2G2THV', 'Data', 'Intensities', 'BaseCalls'), exist_ok=True)
            with open(os.path.join(temp_dir.name, '260303_VH01369_55_AAF2G2THV', 'RunInfo.xml'), 'w') as f:
                f.write("fake")

            obs = _demux_decompression('/vol/jlab/Sequencing/Illumina/260303_VH01369_55_AAF2G2THV.tar.gz', temp_dir.name, 0, verbose=sys.stderr)
            self.assertIn("which is already extracted in", stderr.getvalue())
            self.assertEqual(obs['cmds'], [])
        self.addCleanup(temp_dir.cleanup)

if __name__ == '__main__':
    main()
