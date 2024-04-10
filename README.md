# Edge Private Graph Neural Networks with Singular Value Perturbation
This repository contains code for the PETS 2024 paper "Edge Private Graph Neural Networks with Singular Value Perturbation".

## Dependencies
We suggest using Ananconda for managing the environment. The complete list of required software pacakges are in environment.yml file.

To set up conda and create a conda environment based on environment.yml,
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
conda env create -n [env_name] --file environment.yml
```
The data for linkteller (Twitch dataset) can be download from the website https://snap.stanford.edu/data/twitch-social-networks.html. Please download the zip file from the website and extract it into the data folder and name the subfolder as linkteller-data. Also, rename the json file inside the DE subfolder as musae_DE_features.json.

## Repository content
This repository is heavily based on the code from [Kolluri et al. (2022)](https://github.com/aashishkolluri/lpgnet-prototype) and follows the same structure.

### Hardware Requirement
The codebase has been tested on the following environments.

#### local setup

```
- CPU: AMD EPYC 7502, 64 cores
- RAM: 504 GB
- Disk space: 3.5 T
- GPU: Nvidia Quadro RTX 5000, 16 GB per GPU
- GPU memory required: 8 GB
```

#### VM provided from PETS artifact site

```
- CPU: Xeon E5-2643, 4 cores
- RAM: 8GB
- Disk space: 40 GB
- GPU: none
```

### Software Requirement

The complete list of required software packages are in environment.yml file. Below are
the core libraries used in the codebase.

```
- Python >= 3.8
- PyTorch >= 2.0
- PyG >= 2.3.1
```

---

### Downloading Datasets

---

### Usage ###
```
usage: main.py [-h]
               --arch {mlp,mmlp,gcn}
               [--dataset {cora,citeseer,pubmed,facebook_page,twitch/ES,flickr,bipartite,chameleon}]
               [--test_dataset {twitch/RU,twitch/DE,twitch/FR,twitch/ENGB,twitch/PTBR}]
               [--hidden_size HIDDEN_SIZE]
               [--num_hidden NUM_HIDDEN]
               [--nl NL]
               [--w_dp]
               [--eps EPS]
               [--outdir OUTDIR]
               [--num_seeds NUM_SEEDS] 
               [--sample_seed SAMPLE_SEED]
               [--svd]
               [--rank]
               {train,attack} ...
```
### Arguments ###
`--dataset` (Default: <Dataset.Cora: 'cora'>)
cora|citeseer|pubmed...

`--arch` (Default: <Architecture.MMLP: 'mmlp'>)
Type of architecture to train: mmlp|gcn|mlp

`--nl` (Default: -1)
Only use for MMLP, Number of stacked models, default=-1

`--eps` (Default: 0.0)
The privacy budget. If 0, then do not DP train the arch

`--w_dp`
Run with DP guarantees - if eps=0.0 it throws a warning

`--svd` (Default: False)
Enable Singular Value Decomposition used in Eclipse.

`--rank` (Default: 20)
Choose the rank for Singular Value Decomposition used in Eclipse.

`--hidden_size` (Default: 16)
Size of the hidden layers

`--num_hidden` (Default: 2)
Number of hidden layers

`--outdir` (Default: ../results)
Directory to save the models and results

`--test_dataset` (Default: None)
Test on this dataset, used for Twitch

`--sample_type` (Default: balanced)
Determines how we sample edges for attack.

`--attack_mode` (Default: efficient)
Choose baseline for running LPA and efficient for LinkTeller.

---

### Quick Start: Training and Attacking single models

#### Output directory
We arrange the output in the following way. During running, all intermediate results will
be saved in **results/**. Depending on whether the task is model training or attack, 
results will be saved in **results/train** or **results/attack**.

```
results/
|----train/: model checkpoints (.pth) and metrics (e.g., F1 score, .pkl) after training
|         |
|         |----experiment1/: results generated after training
|         |         |
|         |         |---- model/: model checkpoints (.pth)
|         |         |
|         |         |---- *.pkl: metrics (F1 score)
|         |
|         |-----todos_experiment1/: all tasks (that specify models, datasets, epsilon, …)
|                   |
|                   |----done/: completed training tasks
|                   |
|                   |----working/: pending training tasks
|
| 
|----attack/:
           |
           |----todos_experiment1/: all tasks (that specify models, datasets, epsilon, …)
                    |
                    |----done/: completed attack tasks
                    |
                    |----working/: pending attack tasks

```

---

#### Run training for a single model and dataset with DP
```
python main.py --dataset [Dataset] --arch [mmlp|gcn|mlp] --nl [# stack layers for mmlp] --w_dp --eps [Eps] --svd --rank 20 --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] train --lr [Lr] --dropout [Dropout]
```
- Here is an example to train GCN with Eclipse of rank 20
```
python main.py --dataset cora --arch gcn --w_dp --eps 4.0 --svd --rank 20 --sample_seed 42 --hidden_size 256 --num_hidden 2 train --lr 0.01 --dropout 0.2
```
You can also run for multiple seeds using the --num_seeds option. The results are stored in the folder defined in globals.py or the directory specified using the --outdir option. The trained models are stored in the args.outdir/models directory.
#### Run the attacks on a single trained model ####
To run attack on a trained model, we need all the options used for training that model and a few options in addition such as the attack_mode and sample_type (samples for evaluation).
```
python main.py --dataset [Dataset] --arch [mmlp|gcn|mlp] --nl [# stack layers for mmlp] --w_dp --eps [Eps] --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] --outdir [Outdir] **attack** --lr [Lr] --dropout [Dropout] --attack_mode [bbaseline (lpa) | efficient (linkteller)] --sample_type [balanced | unbalanced]
```
- Here is an example to attack a GCN model trained with Eclipse of rank 20 and stored in ../results/models/
```
python main.py --dataset cora --arch gcn --w_dp --eps 4.0 --svd --rank 20 --sample_seed 42 --hidden_size 256 --num_hidden 2 --outdir ../results attack --lr 0.01 --dropout 0.2  --attack_mode baseline --sample_type balanced
```
The attack results are stored in the directory with name eval_[dataset] which is placed in the current directory.

### Reproducing the results ###
The procedure to reproduce the results follows the steps described in [Kolluri et al. (2022)](https://github.com/aashishkolluri/lpgnet-prototype). We show the commands to include Eclipse in the script.

#### Training the models ####
To train on the best hyperparameters for transductive, 
```
python run_exp.py --rank 20  --num_seeds 5 --command train --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl
```
To train on the best hyperparameters for inductive (Twitch dataset), 
```
python run_exp.py --rank 20  --num_seeds 5 --num_epochs 200 --command train --inductive --datasets TwitchES --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl
```
The [results-dir] and [todos] can be any directory paths where you want to save the results and cache the todo/finished tasks respectively.

#### Attack the trained models ####
To attack the trained models for transductive, 
```
python run_exp.py --rank 20  --num_seeds 5 --command attack --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl 
```
To attack the trained models for inductive (Twitch dataset), 
```
python run_exp.py --rank 20  --num_seeds 5 --num_epochs 200 --command attack --datasets TwitchES --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl 
```
For attacks the --outdir option is used to provide the path to the trained models which is the same as the corresponding path used in training the models. The attack commands save the results in the current directory with "eval_" as a prefix.