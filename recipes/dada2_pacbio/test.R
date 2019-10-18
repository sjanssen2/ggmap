library(dada2); packageVersion("dada2")
library(Biostrings)
library(ShortRead)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(phyloseq)
path.out <- "Figures/"
path.rds <- "RDS/"
F27 <- "AGRGTTYGATYMTGGCTCAG"
R1492 <- "RGYTACCTTGTTACGACTT"
rc <- dada2:::rc
theme_set(theme_bw())

# list all input files: fastq
fns <- c(
    "test_L644.tp40.bam.Q20.fastq",
    "test_L833.tp40.bam.Q20.fastq")

# set tmp filename for primer less fastq
nops <- c(
    "01_noprimer/11953.L644.tp40.fastq",
    "01_noprimer/11953.L833.tp40.fastq")

# perform primer removal
prim <- removePrimers(fns, nops, primer.fwd=F27, primer.rev=dada2:::rc(R1492), orient=TRUE)

# inspect length distribution
lens.fn <- lapply(nops, function(fn) nchar(getSequences(fn)))
lens <- do.call(c, lens.fn)
write.table(lens, file="results_lengthdistribution.csv", sep="\t")

# filter
filts <- c(
    "/home/jansses/TMP/ana_dada2_pacbio_c_mnmidq/02_filtered/11953.L644.tp40.fastq",
    "/home/jansses/TMP/ana_dada2_pacbio_c_mnmidq/02_filtered/11953.L833.tp40.fastq")
track <- filterAndTrim(nops, filts, minQ=3, minLen=1000, maxLen=1600, maxN=0, rm.phix=FALSE, maxEE=2)

# dereplicate sequences
drp <- derepFastq(filts, verbose=TRUE)

# learn error model
errmodel <- learnErrors(drp, errorEstimationFunction=PacBioErrfun, BAND_SIZE=32, multithread=TRUE)
write.table(getErrors(errmodel), file="results_errors_table.csv")
saveRDS(errmodel, "dada2_error_model.rds")

# Denoise
dd2 <- dada(drp, err=errmodel, BAND_SIZE=32, multithread=TRUE)
saveRDS(dd2, "dada2_error_model_dd2.rds")
st2 <- makeSequenceTable(dd2)
write.table(st2, "results_feature-table.csv", sep="	")
write.table(cbind(sample_name=c("11953.L644.tp40","11953.L833.tp40"), ccs=prim[,1], primers=prim[,2], filtered=track[,2], denoised=sapply(dd2, function(x) sum(x$denoised))), file="results_summary.csv", sep="\t")

