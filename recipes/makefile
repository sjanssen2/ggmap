PLATFORM=linux-64

package-feast:
	conda build feast/ -c conda-forge -c aghozlane

upload-feast:
	anaconda upload ${CONDA_PREFIX}/conda-bld/${PLATFORM}/feast-*.tar.bz2

env-feast:
	conda create --name ggmap_feast -c conda-forge -c aghozlane -c sjanssen2 feast -y

package-picrust:
	conda build picrust/ -c bioconda
