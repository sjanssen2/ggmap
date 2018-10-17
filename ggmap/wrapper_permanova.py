from itertools import combinations
from skbio.stats.distance import DistanceMatrix, permanova
import pandas as pd
import sys
import numpy as np


def permanova_groups(file_dm, file_meta, min_group_size, num_permutations,
                     pairindex, dir_out):
    meta = pd.read_csv(file_meta, sep="\t", index_col=0, dtype=str,
                       squeeze=True)
    dm = DistanceMatrix.read(file_dm)
    (a, b) = list(combinations(meta.dropna().unique(), 2))[pairindex-1]
    idx_meta = set(meta[meta.isin([a, b])].dropna().index)
    idx_counts = set(dm.ids)

    meta_pair = meta.loc[idx_meta & idx_counts]
    if (meta_pair.value_counts().min() >= min_group_size) and \
       (meta_pair.value_counts().shape[0] == 2):
        dm_pair = dm.filter(idx_meta & idx_counts)
        res = permanova(dm_pair, meta_pair, permutations=num_permutations)
    else:
        res = pd.Series()
    distances = [dm[x, y]
                 for x in meta_pair[meta_pair == a].index
                 for y in meta_pair[meta_pair == b].index]
    if len(distances) > 0:
        res['avgdist'] = np.mean(distances)
    else:
        res['avgdist'] = np.nan
    for x in [a, b]:
        try:
            res['num_%s' % x] = meta_pair.value_counts().loc[x]
        except KeyError:
            res['num_%s' % x] = 0
    res.to_csv(dir_out+'/permanova_%i.tsv' % pairindex, sep="\t")


if __name__ == '__main__':
    permanova_groups(sys.argv[1],       # file_dm
                     sys.argv[2],       # file_meta
                     int(sys.argv[3]),  # min_group_size
                     int(sys.argv[4]),  # num_permutations,
                     int(sys.argv[5]),  # pairindex
                     sys.argv[6])       # dir_out
