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

JQ := $(CONDA_ENV_DIR)/bin/jq --indent 4 -r
YQ := $(CONDA_ENV_DIR)/bin/yq

.SECONDARY:

.PHONY: .phony-proxy
.phony-proxy:

# ***** Pre-trained SiamRPN Model ***** #

# This is "siamrpn_alex_dwxcorr" from https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md
PYSOT_PRETRAINED_ALEXNET_URI := https://drive.google.com/uc?id=1e51IL1UZ-5seum2yUYpf98l2lJGUTnhs

models/pretrained: ; mkdir -p $@

models/pretrained/siamrpn_alex_dwxcorr.pth: | models/pretrained
	$(ACTIVATE) && gdown $(PYSOT_PRETRAINED_ALEXNET_URI) -O $@

.PHONY: pretrained_siamrpn
pretrained_siamrpn: models/pretrained/siamrpn_alex_dwxcorr.pth

# ***** Fine-tuning ***** #

# This file can have device-specific config changes
models/train/%/tuning.yaml: configs/tuning/%.yaml
	test ! -f $@
	mkdir -p $(@D)
	cat $< \
		| $(YQ) '.TRAIN.NUM_WORKERS = 16' \
		| $(YQ) '.TRAINER.KWARGS.precision = 16' \
		| $(YQ) '.TRAINER.KWARGS.gpus = 4' > $@

models/train/%/.done: models/train/%/tuning.yaml pretrained_siamrpn
	cd $(ROOT_DIR)/data && $(MAKE) training_data

	$(PYTHON) -m skelevision.bin.train $(abspath $<) \
		|& tee -ia $(@D)/train.log; test $${PIPESTATUS[0]} -eq 0

	$(eval $@_CKPT_PATH := skelevision://experiments/$(@D)/best_valid_total_loss.ckpt)
	cat $< | $(YQ) '.MODEL_BUILDER.WEIGHTS_PATH = "$($@_CKPT_PATH)"' > $(@D)/model.yaml

	rm -f $(@D)/hparams.yaml
	touch $@

models/train/%/combined_heads.pth: models/train/%/.done
	$(PYTHON) -m skelevision.bin.combine_heads $(abspath $(@D))

finetune-%: .phony-proxy
	$(MAKE) models/train/$@/.done

#!!! Configurations

configs/tuning/pretrain-keypoint_head.yaml: configs/tuning/BASE-finetuning.yaml
	cat $< \
		| $(YQ) '.DATASET.NAMES = ["coco"]' \
		| $(YQ) '.RPN.TYPE = "NO_RPN"' \
		| $(YQ) '.RPN.KWARGS = {}' \
		| $(YQ) '.TRAIN.BASE_LR = 0.001' \
		| $(YQ) '.MODEL_BUILDER.FREEZE_BACKBONE = True' > $@

configs/tuning/pretrain-keypoint_head-deep.yaml: configs/tuning/pretrain-keypoint_head.yaml
	cat $< | $(YQ) '.KEYPOINT.KWARGS.conv_kwargs = [{"out_channels": 128}, {"out_channels": 128}, {"out_channels": 64}, {"out_channels": 64}]' > $@

configs/tuning/finetune-STL.yaml: configs/tuning/BASE-finetuning.yaml
	cat $< | $(YQ) '.TRAIN.KEYPOINT_WEIGHT = 0.0' > $@

configs/tuning/finetune-MTL-lambda_k_%.yaml: configs/tuning/BASE-finetuning.yaml
	cat $< | $(YQ) '.TRAIN.KEYPOINT_WEIGHT = $*' > $@

configs/tuning/finetune-MTL-pretrain_keypoint_head-lambda_k_%.yaml: \
		configs/tuning/finetune-MTL-lambda_k_%.yaml \
		| models/train/pretrain-keypoint_head/combined_heads.pth
	cat $< | $(YQ) '.MODEL_BUILDER.WEIGHTS_PATH = "skelevision://experiments/$|"' > $@

configs/tuning/finetune-MTL-deep-lambda_k_%.yaml: configs/tuning/finetune-MTL-lambda_k_%.yaml
	cat $< | $(YQ) '.KEYPOINT.KWARGS.conv_kwargs = [{"out_channels": 128}, {"out_channels": 128}, {"out_channels": 64}, {"out_channels": 64}]' > $@

configs/tuning/finetune-MTL-pretrain_keypoint_head-deep-lambda_k_%.yaml: \
		configs/tuning/finetune-MTL-deep-lambda_k_%.yaml \
		| models/train/pretrain-keypoint_head-deep/combined_heads.pth
	cat $< | $(YQ) '.MODEL_BUILDER.WEIGHTS_PATH = "skelevision://experiments/$|"' > $@

#!!! Models

# STL
models/train/finetune-STL/.done:

# MTL
models/train/finetune-MTL-lambda_k_0.2/.done:
models/train/finetune-MTL-lambda_k_0.4/.done:
models/train/finetune-MTL-lambda_k_0.6/.done:
models/train/finetune-MTL-lambda_k_0.8/.done:
models/train/finetune-MTL-lambda_k_1.0/.done:

#!!! Ablations

# MTL (deep)
models/train/finetune-MTL-deep-lambda_k_0.2/.done:
models/train/finetune-MTL-deep-lambda_k_1.0/.done:

# MTL (pretrained keypoint head)
models/train/finetune-MTL-pretrain_keypoint_head-lambda_k_0.2/.done:
models/train/finetune-MTL-pretrain_keypoint_head-lambda_k_1.0/.done:

# MTL (deep + pretrained keypoint head)
models/train/finetune-MTL-pretrain_keypoint_head-deep-lambda_k_0.2/.done:
models/train/finetune-MTL-pretrain_keypoint_head-deep-lambda_k_1.0/.done:

# ***** ARMORY ***** #

ARMORY := $(ACTIVATE) && armory

ARMORY_CONFIG := $(HOME)/.armory/config.json
ARMORY_OUTPUT_DIR := $(shell $(JQ) .output_dir $(ARMORY_CONFIG))

ifneq ($(abspath adversarial),$(ARMORY_OUTPUT_DIR))
$(error Please configure armory to use output_dir as "$(abspath adversarial)" or run `make armory_configure` from the root directory; current value of output_dir is "$(ARMORY_OUTPUT_DIR)")
endif

ARMORY_TARGET_MODEL ?= finetune-STL

adversarial/%.json: configs/armory/%.json
	$(JQ) '.sysconfig.output_dir = "$*.armory_run"' $< > $@

adversarial/%.armory_run: adversarial/%.json models/train/$(ARMORY_TARGET_MODEL)/.done .phony-proxy
	$(ARMORY) run --no-docker $< $(ARGS)

#!!! Configurations

# Example: configs/armory/carla-finetune-STL-eps_step_0.1-max_iter_100.json
configs/armory/carla-$(ARMORY_TARGET_MODEL)-%.json: configs/armory/BASE-carla.json
	$(eval $@_NS := $(subst -, ,$*))
	$(eval $@_NS_NUM := $(words $($@_NS)))
	@test $($@_NS_NUM) -eq 2

	$(eval $@_STP_STR := $(word 1,$($@_NS)))
	$(eval $@_ITR_STR := $(word 2,$($@_NS)))

	$(eval $@_STP := $(subst eps_step_,,$($@_STP_STR)))
	$(eval $@_ITR := $(subst max_iter_,,$($@_ITR_STR)))
	$(eval $@_CFG := skelevision://experiments/models/train/$(ARMORY_TARGET_MODEL)/model.yaml)

	cat $< \
		| $(JQ) '.scenario.export_samples = 20' \
		| $(JQ) '.attack.kwargs.step_size = $($@_STP)' \
		| $(JQ) '.attack.kwargs.max_iter = $($@_ITR)' \
		| $(JQ) '.model.model_kwargs.cfg_path = "$($@_CFG)"' > $@

configs/armory/otb-$(ARMORY_TARGET_MODEL)-%.json: configs/armory/BASE-otb.json
	$(eval $@_NS := $(subst -, ,$*))
	$(eval $@_NS_NUM := $(words $($@_NS)))
	@test $($@_NS_NUM) -eq 2

	$(eval $@_STP_STR := $(word 1,$($@_NS)))
	$(eval $@_ITR_STR := $(word 2,$($@_NS)))

	$(eval $@_STP := $(subst eps_step_,,$($@_STP_STR)))
	$(eval $@_ITR := $(subst max_iter_,,$($@_ITR_STR)))
	$(eval $@_CFG := skelevision://experiments/models/train/$(ARMORY_TARGET_MODEL)/model.yaml)

	cat $< \
		| $(JQ) '.scenario.export_samples = 38' \
		| $(JQ) '.attack.kwargs.step_size = $($@_STP)' \
		| $(JQ) '.attack.kwargs.max_iter = $($@_ITR)' \
		| $(JQ) '.model.model_kwargs.cfg_path = "$($@_CFG)"' > $@

#!!! Attack CLI

# Example: attack~carla~finetune-MTL-lambda_k_0.2~eps_step_0.1~max_iter_100
attack~%: .phony-proxy
	$(eval $@_NS := $(subst ~, ,$*))
	$(eval $@_NS_NUM := $(words $($@_NS)))
	@test $($@_NS_NUM) -eq 4

	$(eval $@_DATASET := $(word 1,$($@_NS)))
	$(eval $@_TRMODEL := $(word 2,$($@_NS)))
	$(eval $@_STP_STR := $(word 3,$($@_NS)))
	$(eval $@_ITR_STR := $(word 4,$($@_NS)))

	$(MAKE) ARMORY_TARGET_MODEL=$($@_TRMODEL) \
		adversarial/$($@_DATASET)-$($@_TRMODEL)-$($@_STP_STR)-$($@_ITR_STR).armory_run
