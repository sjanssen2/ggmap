from unittest import TestCase, main
from io import StringIO
from tempfile import mkstemp
import os

from ggmap.snippets import (cache)
from ggmap.analyses import _executor


@cache
def fct_example(a, b, c):
    return a + b * c


class Snippets_CacheTests(TestCase):
    def setUp(self):
        self.file_cache = mkstemp('.cache')[1]
        os.remove(self.file_cache)

    def tearDown(self):
        os.remove(self.file_cache)

    def test_cache(self):
        # create cache file
        err = StringIO()
        self.assertFalse(os.path.exists(self.file_cache))
        obs = fct_example(2, 4, 6,
                          cache_filename=self.file_cache, cache_err=err)
        self.assertTrue(os.path.exists(self.file_cache))
        self.assertEqual(err.getvalue(),
                         '%s: stored results in cache "%s".\n' %
                         (fct_example.__name__, self.file_cache))
        self.assertEqual(obs, 2+4*6)

        # read from cache file
        err = StringIO()
        obs = fct_example(2, 4, 6,
                          cache_filename=self.file_cache, cache_err=err)
        self.assertTrue(os.path.exists(self.file_cache))
        self.assertEqual(err.getvalue(),
                         '%s: retrieved results from cache "%s".\n' %
                         (fct_example.__name__, self.file_cache))
        self.assertEqual(obs, 2+4*6)

        # check that verbose=False really mutes reporting status
        err = StringIO()
        obs = fct_example(2, 4, 6,
                          cache_filename=self.file_cache, cache_err=err,
                          cache_verbose=False)
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(obs, 2+4*6)

        # check enforcing re-computation
        err = StringIO()
        obs = fct_example(2, 4, 6,
                          cache_filename=self.file_cache, cache_err=err,
                          cache_force_renew=True)
        self.assertEqual(err.getvalue(),
                         '%s: stored results in cache "%s".\n' %
                         (fct_example.__name__, self.file_cache))
        self.assertEqual(obs, 2+4*6)

        # check if errors are raised
        with self.assertRaises(Exception):
            fct_example(2, 4, 6, cache_filename='/dev/null/', cache_err=err)

        with self.assertRaises(PermissionError):
            fct_example(2, 4, 6, cache_filename='/dev/test', cache_err=err)

    def test_cache_paramchange(self):
        err = StringIO()
        obs1 = fct_example(1, 3, 5,
                           cache_filename=self.file_cache, cache_err=err)
        self.assertEqual(obs1, 1+3*5)

        err = StringIO()
        obs2 = fct_example(2, 4, 6,
                           cache_filename=self.file_cache, cache_err=err)
        # since reading from cache does not ensure that parameters are
        # identical, expected results will differ.
        self.assertFalse(obs2 == 2+4*6)

    def test_cache_removeempty(self):
        err = StringIO()
        f = open(self.file_cache, 'wb')
        f.close()
        # check that an empty but existing cache file (a likely result of some
        # function abortion), is removed and not considered a valid cache.
        fct_example(1, 3, 5,
                    cache_filename=self.file_cache, cache_err=err)
        self.assertIn('fct_example: removed empty cache.', err.getvalue())
        self.assertIn('fct_example: stored results in cache "', err.getvalue())


class Snippets_tmpdir(TestCase):
    renamed = False
    dir_tmp = os.environ['HOME'] + '/TMP'

    def setUp(self):
        if os.path.exists(self.dir_tmp):
            os.rename(self.dir_tmp, self.dir_tmp + '.unittest')
            self.renamed = True
        else:
            self.renamed = False

    def tearDown(self):
        if self.renamed:
            os.rename(self.dir_tmp + '.unittest', self.dir_tmp)

    def test_tmpdir_existence(self):
        with self.assertRaisesRegex(
            ValueError,
            ('Temporary directory "%s/" does not exist. '
             'Please create it and restart.') % self.dir_tmp):
                _executor('testTMP', {'fake': 'test'}, None, ['ls -la'], None,
                          dry=False, use_grid=True)


if __name__ == '__main__':
    main()
