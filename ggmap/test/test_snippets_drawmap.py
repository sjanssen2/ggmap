from unittest import TestCase, main
import pandas as pd
from tempfile import mkstemp, gettempdir
import numpy as np
from os import remove
from os.path import join

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from skbio.util import get_data_path

from ggmap.snippets import (drawMap)
from ggmap.imgdiff import compare_images

plt.switch_backend('Agg')
plt.rc('font', family='DejaVu Sans')


class ReadWriteTests(TestCase):
    def setUp(self):
        # set following var to True to generate a new set of baseline plots
        # without testing and aborting after too many differences
        self.mode_generate_baseline = False
        self.dpi = 80
        self.meta_basemap = pd.read_csv(get_data_path(
            'drawMap/basemap.meta.tsv'), sep='\t', index_col=0)
        self.basemap_alaska = Basemap(llcrnrlat=43.,
                                      llcrnrlon=168.,
                                      urcrnrlat=63.,
                                      urcrnrlon=-110,
                                      resolution='i',
                                      projection='cass',
                                      lat_0=90.0,
                                      lon_0=-155.0)

        self.meta_basemap_migration = pd.read_csv(
            get_data_path('drawMap/basemap.meta.migration.tsv'),
            sep='\t', index_col=0)

    def tearDown(self):
        self.assertFalse(self.mode_generate_baseline)

    def help_compare_drawmap(self, name, fig):
        """ Helper function to compare drawMap images """
        file_plotname = 'basemap.%s.png' % name
        file_dummy = join(gettempdir(), file_plotname)
        fig.set_size_inches(16, 11)
        fig.savefig(file_dummy, dpi=self.dpi)
        res = compare_images(get_data_path('drawMap/'+file_plotname),
                             file_dummy,
                             file_image_diff='./diff.'+file_plotname)
        plt.close(fig)
        if not self.mode_generate_baseline:
            if res[0] is True:
                remove(file_dummy)
        return res[0]

    def test_drawMap_alaska(self):
        # create plot
        contrast = 'Anas crecca'
        availcolors = ['blue', 'green', 'orange', 'magenta']
        li = []
        meta_others = self.meta_basemap[
            self.meta_basemap['smj_genusspecies'] != contrast]
        for i, (n, g) in enumerate(meta_others.groupby('Q2')):
            li.append({'label': "%s: %i" % (n, g.shape[0]),
                       'color': availcolors[i],
                       'alpha': 1,
                       'coords': g})
        anas = self.meta_basemap[
            self.meta_basemap.smj_genusspecies == contrast]
        li.append({'label': "%s: %i" % (anas.smj_genusspecies.unique()[0],
                                        anas.smj_genusspecies.shape[0]),
                   'color': availcolors[len(li)],
                   'coords': anas,
                   'size': 10})

        fig, ax = plt.subplots()
        drawMap(li, basemap=self.basemap_alaska, ax=ax)

        # compare image
        self.assertTrue(self.help_compare_drawmap('alaska', fig))

    def test_drawMap_migration(self):
        pn = 0
        meta_mig = self.meta_basemap_migration.groupby(['q1_route',
                                                        'q1_habitat'])
        fix, axarr = fig, ax = plt.subplots(4, 1)
        for pn, (n, g) in enumerate(meta_mig):
            li = [
                {'label': 'summer, with data',
                 'color': 'red',
                 'coords': g[(g['q1_season'] == 'summer') &
                             (g['hasData'] == np.True_)],
                 'alpha': 1},
                {'label': 'winter, with data',
                 'color': 'blue',
                 'coords': g[(g['q1_season'] == 'winter') &
                             (g['hasData'] == np.True_)],
                 'alpha': 1},
            ]
            li.append(
                {'label': 'summer, without data',
                 'color': 'orange',
                 'coords': g[(g['q1_season'] == 'summer')],
                 'alpha': 1})
            li.append(
                {'label': 'winter, without data',
                 'color': 'purple',
                 'coords': g[(g['q1_season'] == 'winter')],
                 'alpha': 1})
            drawMap(reversed(li), ax=axarr[pn], no_legend=pn < 3)
            axarr[pn].set_title("%s %s (winter n=%i->%i, summer n=%i->%i)" % (
                                " | ".join(g.smj_genus.unique()),
                                " | ".join(g.smj_species.unique()),
                                li[1]['coords'].shape[0] +
                                li[3]['coords'].shape[0],
                                li[1]['coords'].shape[0],
                                li[0]['coords'].shape[0] +
                                li[2]['coords'].shape[0],
                                li[0]['coords'].shape[0]))

        # compare image
        self.assertTrue(self.help_compare_drawmap('migration', fig))

    def test_drawMap_default(self):
        li = [{'coords': self.meta_basemap_migration}]
        fig, ax = plt.subplots()
        drawMap(li, ax=ax)
        self.assertTrue(self.help_compare_drawmap('default', fig))

    def test_drawMap_color(self):
        li = [{'coords': self.meta_basemap_migration,
              'color': 'black'}]
        fig, ax = plt.subplots()
        drawMap(li, ax=ax)
        self.assertTrue(self.help_compare_drawmap('color', fig))

    def test_drawMap_size(self):
        li = [{'coords': self.meta_basemap_migration,
              'size': 200}]
        fig, ax = plt.subplots()
        drawMap(li, ax=ax)
        self.assertTrue(self.help_compare_drawmap('size', fig))

    def test_drawMap_alpha(self):
        li = [{'coords': self.meta_basemap_migration,
              'alpha': 0.1}]
        fig, ax = plt.subplots()
        drawMap(li, ax=ax)
        self.assertTrue(self.help_compare_drawmap('alpha', fig))

    def test_drawMap_label(self):
        li = [{'coords': self.meta_basemap_migration,
              'label': 'Voegel'}]
        fig, ax = plt.subplots()
        drawMap(li, ax=ax)
        self.assertTrue(self.help_compare_drawmap('label', fig))

    def test_missing_coords(self):
        allcols = set(self.meta_basemap_migration.columns)
        with self.assertRaisesRegex(ValueError, '"coords" for every dict!'):
            drawMap([{'color': 'blue'}])

        with self.assertRaisesRegex(ValueError,
                                    'need to have column "latitude"'):
            cols = allcols - set(['latitude'])
            drawMap([{'coords': self.meta_basemap_migration.loc[:, cols]}])

        with self.assertRaisesRegex(ValueError,
                                    'need to have column "longitude"'):
            cols = allcols - set(['longitude'])
            drawMap([{'coords': self.meta_basemap_migration.loc[:, cols]}])


if __name__ == '__main__':
    main()
