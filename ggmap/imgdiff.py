import subprocess
import sys
from tempfile import mkstemp
from os import remove, path, access, W_OK
import numpy
from PIL import Image


def compare_images(file_image_a, file_image_b, threshold=0,
                   file_image_diff=None, name=None, err=sys.stderr):
    """Compares content of two images by means of pixel identity.

    Parameters
    ----------
    file_image_a : Filepath
        Filepath to image A
    file_image_b : Filepath
        Filepath to image B
    threshold : int
        The allowed pixel threshold for images A vs. B to be considered equal.
        Default is 0, i.e. images needs to be exactly identical.
    file_image_idff : Filepath
        Optional. Filepath for the differences images for manual visual
        inspection.
        Will be deleted if images are equal (up to given threshold).
    name : str
        A name for the image comparison to be more speaking to the user.

    Returns
    -------
    If images are equal: (True, num pixel distance)
    If one image file cannot be read or diff image file cannot be written:
        (False, numpy.infty)
    Otherwise: (Filepath of diff image, num pixel distance)
        Plus: content of diff image is written as base64 ascii to STDOUT
              (need for Travis to be able to 'look' at the image)
    """

    for file in [file_image_a, file_image_b]:
        if not path.exists(file):
            err.write('Image file "%s" cannot be read.\n' % file)
            return (False, numpy.infty)

    if file_image_diff is None:
        file_image_diff = mkstemp(suffix='.'+file_image_a.split('.')[-1])[1]

    if not access('/'.join(file_image_diff.split('/')[:-1]), W_OK):
        err.write("Cannot write to tmp image file '%s'." % file_image_diff)
        return (False, numpy.infty)

    label = ""
    if name is not None:
        label = "for '%s' " % name

    size_a = Image.open(file_image_a).size
    size_b = Image.open(file_image_b).size
    if size_a != size_b:
        err.write('Images %sdiffer in dimensions: %s and %s.\n' %
                  (label, size_a, size_b))
        return (False, -1 * numpy.infty)

    res = subprocess.check_output(["compare", "-metric", "AE",
                                   "-dissimilarity-threshold", "1",
                                   file_image_a,
                                   file_image_b,
                                   file_image_diff],
                                  stderr=subprocess.STDOUT)
    res = int(res.decode().split('\n')[0])
    if res > threshold:
        sys.stdout.write(("Images differ %s by %i pixels. "
                          "Check differences in %s.\n") %
                         (label, res, file_image_diff))
        cmd = ('echo "==== start file contents (%s)"; '
               'cat %s | base64; '
               'echo "=== end file contents ===";'
               'echo "copy above file content into a file.txt and convert by";'
               'echo "cat file.txt | base64 --decode > file.png"') % (
            file_image_diff,
            file_image_diff)
        rescmd = subprocess.check_output(cmd, shell=True).decode().split('\n')
        for line in rescmd:
            print(line)
        return (file_image_diff, res)
    else:
        remove(file_image_diff)
        return (True, res)
