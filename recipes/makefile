PLATFORM=linux-64
ENV_PREFIX=ggmap_

package-feast:
	conda build feast/ -c conda-forge -c aghozlane

package-picrust1:
	conda build picrust1/ -c bioconda -c conda-forge

package-bugbase:
	conda build bugbase/ -c conda-forge


all-upload: upload-feast upload-picrust1
upload-%: ${CONDA_PREFIX}/conda-bld/${PLATFORM}/%-*.tar.bz2
	anaconda upload ${CONDA_PREFIX}/conda-bld/${PLATFORM}/${*}*.tar.bz2


all-env: env-feast env-picrust1
env-%: %/env.yaml
	@if [ -d "${CONDA_PREFIX}/envs/${ENV_PREFIX}${*}/" ]; then \
		echo "conda environment '${ENV_PREFIX}${*}' already exists."; \
	else \
		echo "now building new environment '${ENV_PREFIX}${*}'"; \
		conda env create --name ${ENV_PREFIX}${*} --file ${*}/env.yaml -y; \
	fi
