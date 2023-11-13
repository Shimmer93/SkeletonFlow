# SkeletonFlow

## Data Preparation

### JHMDB Dataset

1. Register an account [here](http://jhmdb.is.tue.mpg.de/)
2. Download "Rename_Images.tar.gz"
3. Download the annotations from MMPose [here](https://download.openmmlab.com/mmpose/datasets/jhmdb_annotations.tar)
4. Unzip the zip files and arrange the file structure to be as follows:
```
jhmdb
    │-- annotations
    │   │-- Sub1_train.json
    │   |-- Sub1_test.json
    │   │-- Sub2_train.json
    │   |-- Sub2_test.json
    │   │-- Sub3_train.json
    │   |-- Sub3_test.json
    |-- Rename_Images
        │-- brush_hair
        │   │--April_09_brush_hair_u_nm_np1_ba_goo_0
        |   │   │--00001.png
        |   │   │--00002.png
        │-- catch
        │-- ...
```

## Dependency Setup
Create an new conda virtual environment
```
conda create -n af python=3.11.4 -y
conda activate af
```
Clone this repo and install required packages:
```
git clone https://github.com/Shimmer93/SkeletonFlow
cd SkeletonFlow/
pip install -r requirements.txt
```

## Training
```
python main.py -g 8 --n 1 -w 2 -b 32 -e 20 \
--data_dir [path_to_dataset] --pin_memory --wandb_project_name action_flow \
--model_ckpt_dir ./model_ckpt/$(date +'%Y%m%d_%H%M%S') -c cfg/sample.yaml
```

## Evaluation
```
python main.py -c cfg/dyformerv4_t.yaml \
--checkpoint_path [path_to_checkpoint] \
--data_dir [path_to_dataset] --test -g 1 -e 32
```

## Submission to Slurm
In `submit.slurm`:

```
#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --partition=gpu-share
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2

srun python main.py -g 8 -n 1 -w 2 -b 16 -e 10 --data_dir /scratch/PI/cqf/har_data/jhmdb --pin_memory --wandb_project_name test --cfg cfg/sample.yaml --model_ckpt_dir ./model_ckpt/$(date +'%Y%m%d_%H%M%S')
```