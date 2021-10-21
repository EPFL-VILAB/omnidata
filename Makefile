.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard ./*.ipynb)

all: omnidata-tools docs

omnidata-tools: $(SRC)
	nbdev_build_lib
	touch omnidata-tools

sync:
	nbdev_update_lib

docs_serve: docs
	cd docs && bundle exec jekyll serve  --host 0.0.0.0

docs: $(SRC)
	nbdev_build_docs --mk_readme=False
	touch docs

test:
	nbdev_test_nbs

release: pypi 
	nbdev_bump_version

conda_release:
	fastrelease_conda_package

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist