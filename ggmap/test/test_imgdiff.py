from unittest import TestCase, main
from io import StringIO

from skbio.util import get_data_path
import numpy
from tempfile import mkstemp
from os import remove, path

from ggmap.imgdiff import compare_images


class TaxPlotTests(TestCase):
    def setUp(self):
        self.file_img_orig = get_data_path('imageDiff/original.png')
        self.file_img_similar = get_data_path('imageDiff/similar.png')
        self.file_img_changed = get_data_path('imageDiff/changed.png')
        self.file_img_diffdims = get_data_path('imageDiff/diffdim.png')

    def test_compare_images_content(self):
        # verify that identical images are equal
        obs = compare_images(self.file_img_orig, self.file_img_orig,
                             threshold=0)
        self.assertTrue(obs[0])
        self.assertEqual(obs[1], 0)

        # verify that slightly different images are not exactly equal ...
        obs = compare_images(self.file_img_orig, self.file_img_similar,
                             threshold=0)
        self.assertTrue(obs[0] is not True)
        self.assertEqual(obs[1], 8)

        # ... but are considered equal if threshold is raised
        obs = compare_images(self.file_img_orig, self.file_img_similar,
                             threshold=9)
        self.assertTrue(obs[0])
        self.assertEqual(obs[1], 8)

        # verify that major differences are recognized
        obs = compare_images(self.file_img_orig, self.file_img_changed,
                             threshold=9)
        self.assertTrue(obs[0] is not True)
        self.assertTrue(obs[1] > 9)

    def test_compare_images_dimdiff(self):
        # verify that images of different sizes are recognized as beeing
        # different
        err = StringIO()
        obs = compare_images(self.file_img_orig, self.file_img_diffdims,
                             threshold=9, err=err)
        self.assertFalse(obs[0])
        self.assertEqual(obs[1], -1 * numpy.infty)
        self.assertEqual(err.getvalue(),
                         ('Images differ in dimensions: (80, 60) '
                          'and (90, 60).\n'))

    def test_compare_images_misc(self):
        # test default behaviour
        err = StringIO()
        obs = compare_images(self.file_img_orig, self.file_img_diffdims,
                             err=err)
        self.assertTrue(obs[0] is not True)
        self.assertIn('Images differ in dimensions', err.getvalue())

        # test that name is printed if given
        err = StringIO()
        obs = compare_images(self.file_img_orig, self.file_img_diffdims,
                             err=err, name='kurt')
        self.assertTrue(obs[0] is not True)
        self.assertIn('Images for \'kurt\' differ in dimensions',
                      err.getvalue())

        # check that diff image can be written to specified filepath
        # first, create a placeholder filename and delete the file
        file_dummy = mkstemp('.dummy.png')[1]
        remove(file_dummy)
        # verify that the file does not exist at the moment
        self.assertTrue(path.exists(file_dummy) is False)
        obs = compare_images(self.file_img_orig, self.file_img_changed,
                             err=err, file_image_diff=file_dummy)
        self.assertTrue(path.exists(file_dummy))
        self.assertTrue(obs[0] is not True)
        remove(file_dummy)


if __name__ == '__main__':
    main()
