from ggmap.snippets import biom2pandas

c = biom2pandas('ggmap/test/data/25x25.biom')
assert(c.shape[0] == 25)
assert(c.shape[1] == 25)
