<<<<<<< HEAD
# Controlled Knowledge Transfer for Generalized Few-Shot Object Detection
This is the code for "Controlled Knowledge Transfer for Generalized Few-Shot Object Detection"

![](imgs/overview.png)

This code is based on [MMFewshot](https://github.com/open-mmlab/mmfewshot), you can see the mmfew for more detail about the instructions.

![](imgs/nwpu_visual.png)
## Detection Results
![](imgs/dior_visual.png)

## Two-stage training framework

Following the original implementation, it consists of 3 steps:
- **Step1: Base training**
   - use all the images and annotations of base classes to train a base model.

- **Step2: Reshape the bbox head of base model**:
   - create a new bbox head for all classes fine-tuning (base classes + novel classes) using provided script.
   - the weights of base class in new bbox head directly use the original one as initialization.
   - the weights of novel class in new bbox head use random initialization.

- **Step3: Few shot fine-tuning**:
   - use the base model from step2 as model initialization and further fine tune the bbox head with few shot datasets.


### An example of DIOR split1 10-shot setting with 2 gpus

```bash
# step1: base training for voc split1-10shot-10shot
bash ./tools/detection/dist_train.sh \
    configs/detection/ETF/dior/split1-10shot-10shot/tfa_r101_fpn_dior-split1_base-training.py 2

# step2: reshape the bbox head of base model for few shot fine-tuning
python -m tools.detection.misc.initialize_bbox_head \
    --src1 work_dirs/ETF_r101_fpn_voc-split1_base-training/latest_arcfaceloss.pth \
    --method randinit \
    --save-dir work_dirs/ETF_r101_fpn_voc-split1_base-training

# step3(Model ETF): few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/ETF/dior/split1-10shot-10shot/ETF_r101_fpn_nwpuv2-split1_10shot-fine-tuning.py 2


# step3(Model ETF+Dis): few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/dis_loss/dior/split1-10shot-10shot/power4_dis_tfa_r101_fpn_dior-split1_3shot-fine-tuning.py 2


# step3(Model G-FSDet): few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/GFSDet/dior/split1-10shot-10shot/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_nwpu-split2_3shot-fine-tuning.py 2
```
**Note**:
- The default output path of the reshaped base model in step2 is set to `work_dirs/{BASE TRAINING CONFIG}/base_model_random_init_bbox_head.pth`.
  When the model is saved to different path, please update the argument `load_from` in step3 few shot fine-tune configs instead
  of using `resume_from`.
- To use pre-trained checkpoint, please set the `load_from` to the downloaded checkpoint path.

## Data preparation
We have provided  the few-shot annotations in 'data/few_shot_ann'. 

# Controlled Knowledge Transfer for Generalized Few-Shot Object Detection
>>>>>>> ckt/master
