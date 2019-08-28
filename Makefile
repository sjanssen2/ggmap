# ----------------------------------------------------------------------------
# Copyright (c) 2015--, ggmap development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

SHELL=/bin/bash
.DEFAULT_GOAL := help

ifeq ($(WITH_COVERAGE), TRUE)
	TEST_COMMAND = COVERAGE_FILE=.coverage coverage run --rcfile .coveragerc setup.py nosetests --with-doctest
else
	TEST_COMMAND = nosetests --with-doctest
endif

help:
	@echo 'Use "make test" to run all the unit tests and docstring tests.'
	@echo 'Use "make pep8" to validate PEP8 compliance.'
	@echo 'Use "make html" to create html documentation with sphinx'
	@echo 'Use "make all" to run all the targets listed above.'
	@echo 'Use "make check" to run all tests not possible on Travis.'
test:
	$(TEST_COMMAND)
pep8:
	flake8 ggmap setup.py
html:
	make -C doc clean html
check:
	python ggmap/test/check_analyses_renew.py
	python ggmap/test/check_analyses_sepp.py

all: test pep8

# make targets to install ggmap in a fresh Linux machine

install: conda_install
	if [ -z "${CONDA_PREFIX}" ]; then \
		conda create -n ggmap python=3.5 -y; \
	  source activate ggmap; \
	fi;
	conda install basemap --override-channels --channel conda-forge
	conda install --file ci/conda_requirements.txt -c conda-forge -y;
	pip install -r ci/pip_requirements.txt;
	python setup.py develop;

conda_install:
	if ! which conda > /dev/null; then \
		if [ ! -f miniconda.sh ]; then \
			wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh; \
		fi; \
		bash miniconda.sh -b -p ${HOME}/miniconda; \
		export PATH="${HOME}/miniconda/bin:${PATH}"; \
	fi
