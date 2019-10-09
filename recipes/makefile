PLATFORM=linux-64
ENV_PREFIX=ggmap_

package-feast:
	conda build feast/ -c conda-forge -c aghozlane

upload-feast:
	anaconda upload ${CONDA_PREFIX}/conda-bld/${PLATFORM}/feast-*.tar.bz2

env-feast:
	conda create --name ${ENV_PREFIX}_feast -c conda-forge -c aghozlane -c sjanssen2 feast -y


package-picrust1:
	conda build picrust1/ -c bioconda -c conda-forge

upload-picrust1:
	anaconda upload ${CONDA_PREFIX}/conda-bld/${PLATFORM}/picrust1-*.tar.bz2

env-picrust1:
	conda create --name ${ENV_PREFIX}_picrust1 -c bioconda -c conda-forge -c sjanssen2 picrust1 -y
