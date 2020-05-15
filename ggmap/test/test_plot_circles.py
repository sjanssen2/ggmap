from unittest import TestCase, main

import pandas as pd
from skbio.util import get_data_path

from ggmap.snippets import plot_circles


class PlotCirclesTests(TestCase):
    def setUp(self):
        self.meta = pd.read_csv(get_data_path('meta_cyto.csv'), sep="\t", index_col=0)
        self.meta_cno = pd.read_csv(get_data_path('meta_cno.csv'), sep="\t", index_col=0)
        self.links = pd.read_csv(get_data_path('links_cyto.csv'), sep="\t", index_col=0)
        self.colors_circleplot = {
            'trio': {'etv6': 'cyan', 'hyperdiploid': 'orange'},
            'family_matchedpair': {
                'healthy_family': 'green', 'diseased_family': 'red'},
            'host_subject_id': {
                'father': '#b3e2cd',
                'mother': '#fdcdac',
                'patient': '#cbd5e8',
                'sibling': '#cccccc',
                'child': '#cccccc'}}

    def tearDown(self):
        pass

    def test_cyto(self):
        # real situation, but no colors
        meta = self.meta.reset_index().set_index(['trio', 'family_matchedpair', 'host_subject_id']).sort_values(by=['trio_phenotype'])
        plot_circles(meta)

        # color schema defined, but no colors given
        colgrp = {'trio': 'trio_phenotype', 'family_matchedpair': 'family_type', 'host_subject_id': 'family_degree'}
        with self.assertRaisesRegex(
                AttributeError,
                "'NoneType' object has no attribute 'get'"):
            plot_circles(meta, cols_grps=colgrp)

        # as above, but colors given
        plot_circles(meta, cols_grps=colgrp, colors=self.colors_circleplot)

        # with links
        plot_circles(
            meta,
            cols_grps=colgrp,
            colors=dict(self.colors_circleplot, **{
                'links': {'to father': 'blue',
                          'to mother': 'red'}}),
            links=[((row['a_trio'], row['a_family_matchedpair'], row['a_host_subject_id']),
                    (row['b_trio'], row['b_family_matchedpair'], row['b_host_subject_id']), row['relation']) for _, row in
                   self.links.iterrows()]
        )

        # no links and only two levels
        meta = self.meta.reset_index().set_index(['trio', 'family_matchedpair']).sort_values(by=['trio_phenotype'])
        plot_circles(
            meta,
            cols_grps=colgrp,
            colors=dict(self.colors_circleplot, **{
                'links': {'to father': 'blue',
                          'to mother': 'red'}}),
        )

        # only one level
        meta = self.meta.reset_index().set_index(['trio']).sort_values(by=['trio_phenotype'])
        plot_circles(
            meta,
            cols_grps=colgrp,
            colors=dict(self.colors_circleplot, **{
                'links': {'to father': 'blue',
                          'to mother': 'red'}}),
        )

        # only one level + 2 links
        meta = self.meta.reset_index().set_index('trio').sort_values(by=['trio_phenotype'])
        plot_circles(
            meta,
            cols_grps=colgrp,
            colors=dict(self.colors_circleplot, **{
                'links': {'to father': 'blue',
                          'to mother': 'red'}}),
            links=[(('CP03'), ('CP11'), 'to mother'), (('CP04'), ('CP10'), 'to father')],
        )

        # only one level + 1 link
        meta = self.meta.reset_index().set_index('trio').sort_values(by=['trio_phenotype'])
        plot_circles(
            meta,
            cols_grps=colgrp,
            colors=dict(self.colors_circleplot, **{
                'links': {'to father': 'blue',
                          'to mother': 'red'}}),
            links=[(('CP03'), ('CP11'), 'to mother')],
        )

    def test_cno(self):
        meta = self.meta_cno[pd.notnull(self.meta_cno['timepoint'])].reset_index().set_index(['family_id', 'ort', 'family_role'])
        colors_circleplot = {
            'timepoint': {'0': 'lightgreen', '3': 'darkgreen'},
            'ort': {'tongue_left': 'gold', 'tongue_center': 'purple'},
            'family_id': {x: 'black' for x in list(self.meta_cno[pd.notnull(self.meta_cno['timepoint'])]['family_id'].unique())},
            'family_role': {'patient': 'blue', 'father': 'red', 'mother': 'magenta'}}
        plot_circles(meta,
                     cols_grps={
                        #'timepoint': 'timepoint',
                        'ort': 'ort',
                        'family_id': 'family_id',
                        'family_role': 'family_role'
                     },
                     colors=colors_circleplot)


if __name__ == '__main__':
    main()
