PLATFORM=linux-64
ENV_PREFIX=ggmap_

package-feast:
	conda build feast/ -c conda-forge -c aghozlane

package-picrust1:
	conda build picrust1/ -c bioconda -c conda-forge

package-bugbase:
	conda build bugbase/ -c conda-forge

package-metalonda:
	conda build metalonda/ -c conda-forge

package-dada2_pacbio:
	conda build dada2_pacbio -c conda-forge -c bioconda

package-sourcetracker2:
	conda build sourcetracker2 -c conda-forge -c biocore

package-pldist:
	conda build pldist -c conda-forge -c bioconda -c r

all-upload: upload-feast upload-picrust1 upload-bugbase upload-metalonda upload-dada2_pacbio upload-pldist upload-sourcetracker2
upload-%: ${CONDA_PREFIX}/conda-bld/${PLATFORM}/%-*.tar.bz2
	anaconda upload ${CONDA_PREFIX}/conda-bld/${PLATFORM}/${*}*.tar.bz2


all-env: env-feast env-picrust1 env-picrust2 env-bugbase env-metalonda env-dada2_pacbio env-pldist env-sourcetracker2
env-%: %/env.yaml
	@if [ -d "${CONDA_PREFIX}/envs/${ENV_PREFIX}${*}/" ]; then \
		echo "conda environment '${ENV_PREFIX}${*}' already exists."; \
	else \
		echo "now building new environment '${ENV_PREFIX}${*}'"; \
		conda env create --name ${ENV_PREFIX}${*} --file ${*}/env.yaml; \
	fi
