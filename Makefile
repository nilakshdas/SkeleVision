SHELL := /bin/bash

.PHONY: all
all: venv

.PHONY: clean
clean:
	rm -rf lib
	rm -rf venv

# ***** lib ***** #

lib: ; mkdir -p $@

lib/adversarial-robustness-toolbox/.git: | lib
	git clone https://github.com/Trusted-AI/adversarial-robustness-toolbox.git $(@D)
	cd $(@D) && git checkout -b skelevision c4f4e242e89132c4b515e502e73a5c7d9811ace6

lib/armory/.git: | lib
	git clone https://github.com/twosixlabs/armory.git $(@D)
	cd $(@D) && git checkout -b skelevision 388edde7d85f96dac6a96c13854b955f1bb5c3c3

lib/pysot/.git: | lib
	git clone https://github.com/STVIR/pysot.git $(@D)
	cd $(@D) && git checkout -b skelevision 9b07c521fd370ba38d35f35f76b275156564a681

lib/.done: | \
		lib/adversarial-robustness-toolbox/.git \
		lib/armory/.git \
		lib/pysot/.git
	touch $@

# ***** venv ***** #

CONDA_ENV_NAME := skelevision
CONDA_URL ?= https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

CONDA_SH := $(abspath venv/etc/profile.d/conda.sh)
ACTIVATE := . $(CONDA_SH) && conda deactivate && conda activate $(CONDA_ENV_NAME)

CONDA_ENV_DIR := venv/envs/$(CONDA_ENV_NAME)

CONDA  := $(ACTIVATE) && conda
PIP    := $(ACTIVATE) && pip
PYTHON := $(ACTIVATE) && python
JQ     := $(CONDA_ENV_DIR)/bin/jq --indent 4 -r
YQ     := $(CONDA_ENV_DIR)/bin/yq

$(CONDA_ENV_DIR): environment.yml
	$(MAKE) clean
	wget $(CONDA_URL) -O miniconda.sh
	bash miniconda.sh -b -p venv
	rm miniconda.sh
	source $(CONDA_SH) && conda deactivate && conda env create -f $<

$(YQ): | $(CONDA_ENV_DIR)
	wget https://github.com/mikefarah/yq/releases/download/v4.18.1/yq_linux_386 -O $@
	chmod +x $@

$(CONDA_ENV_DIR)/.done: $(CONDA_ENV_DIR) $(YQ)
	touch $@

venv/.done: requirements.txt $(CONDA_ENV_DIR)/.done lib/.done
	$(PIP) install -r lib/pysot/requirements.txt
	$(PIP) install -r lib/armory/requirements.txt
	$(PIP) install -e lib/adversarial-robustness-toolbox
	$(PIP) install -e lib/armory
	$(PIP) install -e lib/pysot
	cd $(CONDA_ENV_DIR)/lib/python3.7/ \
		&& ln -sf $(abspath lib/pysot/pysot)
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	touch $@

venv: venv/.done

# ***** ARMORY ***** #

ARMORY_CONFIG := $(HOME)/.armory/config.json

$(ARMORY_CONFIG): | venv
	$(ACTIVATE) && yes "" | armory configure

.PHONY: armory_configure
armory_configure: $(ARMORY_CONFIG)
	cat $< | $(JQ) '.output_dir = "$(abspath experiments/adversarial)"' > $<.tmp
	mv $< $<.bak.$$(date +%Y%m%d%H%M%S)
	mv $<.tmp $<

# ***** Training Data ***** #

.PHONY: training_data
training_data: venv
	cd data && $(MAKE) $@

# ***** Model Training ***** #

.PHONY: pretrained_siamrpn
pretrained_siamrpn: venv
	cd experiments && $(MAKE) $@

finetune-% attack~%: venv pretrained_siamrpn training_data
	cd experiments && $(MAKE) $@
