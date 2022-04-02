########################################
SHELL := /bin/bash
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
ifeq (,$(wildcard $(ROOT_DIR)/venv/.done))
$(error Please install the virtual evironment first by running `make venv` in $(ROOT_DIR))
endif
CONDA_ENV_NAME := skelevision
CONDA_ENV_DIR := $(ROOT_DIR)/venv/envs/$(CONDA_ENV_NAME)
CONDA_SH := $(abspath $(ROOT_DIR)/venv/etc/profile.d/conda.sh)
ACTIVATE := . $(CONDA_SH) && conda deactivate && conda activate $(CONDA_ENV_NAME)
PYTHON := $(ACTIVATE) && python
########################################

.PHONY: training_data
training_data: \
	coco/.done lasot-person/.done

# ***** COCO ***** #

coco: ; mkdir -p $@

MAP_COCO_ZIP_URI-annotations := http://images.cocodataset.org/annotations/annotations_trainval2017.zip
MAP_COCO_ZIP_URI-train2017 := http://images.cocodataset.org/zips/train2017.zip
MAP_COCO_ZIP_URI-val2017 := http://images.cocodataset.org/zips/val2017.zip

coco/annotations/.done \
coco/train2017/.done \
coco/val2017/.done: | coco
	$(eval $@_ZIP_URI := $(MAP_COCO_ZIP_URI-$(notdir $(@D))))
	$(eval $@_ZIP_FILE := $(lastword $(subst /, ,$($@_ZIP_URI))))
	wget $($@_ZIP_URI)
	unzip $($@_ZIP_FILE) -d $|
	rm $($@_ZIP_FILE)
	touch $@

coco/preprocessed/%.h5: \
		coco/annotations/.done\
		coco/train2017/.done \
		coco/val2017/.done
	$(PYTHON) -m skelevision.bin.data.prepare_coco \
		--data-dir $(abspath $(@D)/..) --split $*

coco/.done: \
	coco/preprocessed/train.h5 \
	coco/preprocessed/valid.h5

# ***** LaSOT ***** #

lasot-person: ; mkdir -p $@

LASOT_PERSON_DATA_URI := https://drive.google.com/uc?id=1yGANBnOx3bL52jVSWFCs1Mkwr0BRtaEp

.INTERMEDIATE: person.zip
person.zip:
	@echo ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	@echo + Please manually download $@ from
	@echo + $(LASOT_PERSON_DATA_URI)
	@echo + and place it in this directory if you see an error below.
	@echo ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	$(ACTIVATE) && gdown $(LASOT_PERSON_DATA_URI)

lasot-person/raw/.done: person.zip | lasot-person
	unzip $< -d $|
	mv $|/person $|/raw
	touch $@

lasot-person/preprocessed/train.h5 \
lasot-person/preprocessed/valid.h5: lasot-person/raw/.done
	$(PYTHON) -m skelevision.bin.data.prepare_lasot --data-dir $(abspath $(<D))
	touch $@

lasot-person/.done: lasot-person/preprocessed/train.h5 lasot-person/preprocessed/valid.h5
	touch $@
