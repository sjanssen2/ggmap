from unittest import TestCase, main
from io import StringIO
from tempfile import mkstemp
from os import remove
import os
import yaml

from ggmap import settings


class SettingsTests(TestCase):
    def setUp(self):
        self.file_fake_settings = mkstemp()[1]
        self.err = StringIO()
        remove(self.file_fake_settings)

    def tearDown(self):
        if os.path.exists(self.file_fake_settings):
            remove(self.file_fake_settings)

    def test_write_primer(self):
        # ensure that a new configuration file is written as primer with
        # default values if config file does not exists. This will help users
        # to change settings.
        settings.FP_SETTINGS = self.file_fake_settings
        settings.init(err=self.err)
        self.assertEqual(self.err.getvalue(), 'New config file "%s" created.' %
                         self.file_fake_settings)

    def test_defaults(self):
        # check that default value is used after init()
        settings.FP_SETTINGS = self.file_fake_settings
        settings.init(err=self.err)
        self.assertIn('Kingdom', settings.RANKS)
        self.assertEqual('/usr/bin/time', settings.EXEC_TIME)
        self.assertEqual(self.err.getvalue(), 'New config file "%s" created.' %
                         self.file_fake_settings)

    def test_partial_defined(self):
        # test that values set by file are used and non-set fall back to
        # default
        with open(self.file_fake_settings, 'w') as f:
            yaml.dump({'list_ranks': ['R1', 'R2']}, f)
        settings.FP_SETTINGS = self.file_fake_settings
        settings.init(err=self.err)
        self.assertEqual('/usr/bin/time', settings.EXEC_TIME)
        self.assertEqual(['R1', 'R2'], settings.RANKS)


if __name__ == '__main__':
    main()
