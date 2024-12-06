#!/usr/bin/env python

import sys

import numpy as np

from sklearn.metrics import pairwise_distances

from biom.table import Table
from biom.util import biom_open
from skbio.stats.distance import DistanceMatrix

from qiime2 import Artifact

def compute_beta(metric, counts, ppn:int=8):
    # use sklearn function to parallely compute distances
    print("2/4: computing distances", file=sys.stderr)
    distances = pairwise_distances(
        np.asarray(counts.transpose().matrix_data.todense()),
        metric=metric,
        n_jobs=ppn)

    # import distances into a qiime artifact
    print("3/4: converting to qiime artifact", file=sys.stderr)
    imported_artifact = Artifact.import_data(
        "DistanceMatrix",
        DistanceMatrix(distances, ids=counts.ids('sample')))

    return imported_artifact


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: beta_parallel.py <input=file path biom> <output=file path qza> <metric> <threads>")
        sys.exit(1)

    fp_input = sys.argv[1]
    fp_output = sys.argv[2]
    metric = sys.argv[3]
    ppn = int(sys.argv[4])

	# load feature table from biom file
    print("1/4: loading feature table", file=sys.stderr)
    with biom_open(fp_input) as f:
      counts = Table.from_hdf5(f)

    # trigger diversity computation
    distances = compute_beta(metric, counts, ppn=ppn)

    # store distances to file
    print("4/4: storing to disk", file=sys.stderr)
    distances.save(fp_output)
