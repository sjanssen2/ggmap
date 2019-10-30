library(pldist)
library(ape)

ggmap.otus <- as.matrix(read.table("test_counts.tsv", header=TRUE, sep = "\t", row.names = 1, as.is=TRUE))
ggmap.meta <- read.table("test_meta.tsv", header=TRUE, sep = "\t", row.names = 1)

res <- pldist(ggmap.otus, ggmap.meta, paired=FALSE, binary=FALSE, method="bray", clr=FALSE)
write.table(res$D, "test_obs_UW.tsv", sep="\t", row.names=TRUE, col.names=NA)
