# SkeleVision

> [**SkeleVision: Towards Adversarial Resiliency of Person Tracking with Multi-Task Learning**](https://arxiv.org/abs/2204.00734)  
> Nilaksh Das, Sheng-Yun Peng, Duen Horng Chau

## Quickstart

```bash
$ make venv
$ make armory_configure
$ CUDA_VISIBLE_DEVICES=0,1,2,3 make \
    attack~carla~finetune-MTL-lambda_k_0.2~eps_step_0.2~max_iter_100
```

---

## Setup

```bash
$ make venv
$ make armory_configure
```

## Training Data

```bash
$ make training_data
```

> NOTE: We have automated the process for downloading the training data,
> but due to throttling limitations of Google Drive that are beyond our control,
> you MAY have to manually download the LaSOT dataset split (`person.zip`)
> from [this link](https://drive.google.com/uc?id=1yGANBnOx3bL52jVSWFCs1Mkwr0BRtaEp)
> and place it in the `data/` directory.

## Model Training

First, you will need to download the pre-trained SiamRPN weigths.

```bash
$ make pretrained_siamrpn
```

Next, running the following command will train the model
and save it to `experiments/models/train/<MODEL_NAME>`

```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3 make <MODEL_NAME>
```

### Model Names

The MTL models are suffixed with the MTL weight (`lambda_k`).

```
# STL
finetune-STL

# MTL
finetune-MTL-lambda_k_0.2
finetune-MTL-lambda_k_0.4
finetune-MTL-lambda_k_0.6
finetune-MTL-lambda_k_0.8
finetune-MTL-lambda_k_1.0

# MTL Ablation: deep
finetune-MTL-deep-lambda_k_0.2
finetune-MTL-deep-lambda_k_1.0

# MTL Ablation: pretrained keypoint head
finetune-MTL-pretrain_keypoint_head-lambda_k_0.2
finetune-MTL-pretrain_keypoint_head-lambda_k_1.0

# MTL Ablation: deep + pretrained keypoint head
finetune-MTL-pretrain_keypoint_head-deep-lambda_k_0.2
finetune-MTL-pretrain_keypoint_head-deep-lambda_k_1.0
```

## Adversarial Attack

Running the following command will
perform an adversarial attack
and save the results at:

`experiments/adversarial/<DATASET_NAME>-<MODEL_NAME>-eps_step_<STEP_SIZE>-max_iter_<NUM_ITERS>.armory_run_<TIMESTAMP>`

```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3 make \
    attack~<DATASET_NAME>~<MODEL_NAME>~eps_step_<STEP_SIZE>~max_iter_<NUM_ITERS>
```

### Dataset Names
- `carla`
- `otb`

### Attack examples

```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3 make \
    attack~carla~finetune-MTL-lambda_k_0.2~eps_step_0.2~max_iter_100

$ CUDA_VISIBLE_DEVICES=0,1,2,3 make \
    attack~carla~finetune-MTL-deep-lambda_k_1.0~eps_step_0.1~max_iter_50

$ CUDA_VISIBLE_DEVICES=0,1,2,3 make \
    attack~otb~finetune-MTL-lambda_k_0.4~eps_step_0.2~max_iter_20
```

> NOTE: If you have not downloaded the training data and the pre-trained model,
> or trained the model being attacked, running the above examples
> will do all of that before performing the adversarial attack.

## Citation

```
@misc{das2022skelevision,
      title={SkeleVision: Towards Adversarial Resiliency of Person Tracking with Multi-Task Learning}, 
      author={Nilaksh Das and Sheng-Yun Peng and Duen Horng Chau},
      year={2022},
      eprint={2204.00734},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
