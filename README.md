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

## Repository content
This repository is heavily based on the code from [Kolluri et al. (2022)](https://github.com/aashishkolluri/lpgnet-prototype) and follows the same structure. All python scripts are included in src/ directory.

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

Note: this setup was only tested on Cora, Citeseer, Chameleon dataset 
due to resource constraints in the provided VMs.

In general, the code can be executed on CPU or GPU. The CPU memory should be larger than 8GB to avoid Out-of-Memory error when executing on Pubmed and facebook page dataset. For all other datasets, 8GB CPU memory is sufficient. The GPUs can be rented through GPU cloud service providers, such as AWS, Lambda Labs, vast.ai, etc. The GPU memory should also be at least 8GB.

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

The data for linkteller (Twitch dataset) can be download and prepared using

```
bash get_twitch.sh
```
---

### Estimated Time and Storage Consumption
Depending on the dataset size, the evalutation for one specific choice of (dataset, random seed, model, epsilon value) will take at most 5 minutes and consume at most 4GB space on the disk. The total space to reserve on disk to accomondate all datasets and intermediate results should be 8GB.

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


## Quick Start: Training and Attacking single models

### Output directory
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

### Run training for a single model and dataset with DP
```
python main.py --dataset [Dataset] --arch [mmlp|gcn|mlp] --nl [# stack layers for mmlp] --w_dp --eps [Eps] --svd --rank 20 --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] train --lr [Lr] --dropout [Dropout]
```
- Here is an example to train GCN with Eclipse of rank 20
```
python main.py --dataset cora --arch gcn --w_dp --eps 4.0 --svd --rank 20 --sample_seed 42 --hidden_size 256 --num_hidden 2 train --lr 0.01 --dropout 0.2
```
You can also run for multiple seeds using the --num_seeds option. The results are stored in the folder defined in globals.py or the directory specified using the --outdir option. The trained models are stored in the args.outdir/models directory.
### Run the attacks on a single trained model
To run attack on a trained model, we need all the options used for training that model and a few options in addition such as the attack_mode and sample_type (samples for evaluation).
```
python main.py --dataset [Dataset] --arch [mmlp|gcn|mlp] --nl [# stack layers for mmlp] --w_dp --eps [Eps] --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] --outdir [Outdir] **attack** --lr [Lr] --dropout [Dropout] --attack_mode [bbaseline (lpa) | efficient (linkteller)] --sample_type [balanced | unbalanced]
```
- Here is an example to attack a GCN model trained with Eclipse of rank 20 and stored in ../results/models/
```
python main.py --dataset cora --arch gcn --w_dp --eps 4.0 --svd --rank 20 --sample_seed 42 --hidden_size 256 --num_hidden 2 --outdir ../results attack --lr 0.01 --dropout 0.2  --attack_mode baseline --sample_type balanced
```
The attack results are stored in the directory with name eval_[dataset] which is placed in the current directory.


## Reproducing the results
The procedure to reproduce the results follows the steps described in [Kolluri et al. (2022)](https://github.com/aashishkolluri/lpgnet-prototype). We show the commands to include Eclipse in the script.

### Training Models
---
#### Hyperparameter Search ####
To search for the best hyperparameters for transductive datasets,
```
python run_exp.py --num_seeds 5 --command train --outdir ../data-hyperparams --hyperparameters --todos_dir [todos]
```
To search for the best hyperparameters for inductive dataset (Twitch dataset),
```
python run_exp.py --num_seeds 5 --num_epochs 200 --command train --outdir ../data-hyperparams-inductive --hyperparameters --todos_dir [todos] --inductive
```
To parse the best configuration for (dataset, architecture)
```
python run_exp.py --parse_config_dir [data-hyperparameters-dir]
```
This creates a file `best_config.pkl` in the local directory which contains the best hyperparameters.

You can also skip the above search by directly using the `best_config.pkl` file provided in the artifact.

#### Generating to-do tasks for training ####
Before training for transductive starts, generate the complete set of to-do tasks,
```
python run_exp.py --rank 20  --num_seeds 5 --command train --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl --only_create_todos
```
Before training for inductive starts, generate the complete set of to-do tasks,
```
python run_exp.py --rank 20  --num_seeds 5 --num_epochs 200 --command train --inductive --datasets TwitchES --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl --only_create_todos
```
#### Training the models ####
To train on the best hyperparameters for transductive, 
```
python run_exp.py --rank 20  --num_seeds 5 --command train --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl
```
To train on the best hyperparameters for inductive (Twitch dataset), 
```
python run_exp.py --rank 20  --num_seeds 5 --num_epochs 200 --command train --inductive --datasets TwitchES --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl
```
You can change the rank to keep after Singular Value Decomposition by setting a different value for `--rank`. We used rank of 20 to produce the results in the paper.

The [results-dir] and [todos] can be any directory paths where you want to save the results and cache the todo/finished tasks respectively.

#### Parse training results ####
To parse results to get utility scores, provide path to the results directory used during the training.

For transductive,
```
python parser_ash.py --results_dir [results-dir]
```
For inductive,
```
python parser_ash_ind_utility.py --results_dir [results-dir]
```
The parsed results will be output in the results folder.

The expected result is stored in the `expected_results` folder.

The training experiment for transductive in our paper runs for 5 datasets(Cora, Citeseer, Pubmed, Facebook page, Chameleon) x 5 random seeds x 4 model architectures(GCN,MLP,LPGNet,Eclipse) x 20 epsilon values ([0.1,0.2,...,0.9], [0,1,2,...,10]) = 1525 tasks (MLP only uses one epsilon value since no adjacency matrix is involved in MLP training). These task runs in approximately 128 hours on one machine and consumes 8GB on disk. You can run the command simultaneously in different terminals to expediate the execution of the job.

#### Quick Verification
To quickly verify that the training experiment can be executed correctly, we provide run_exp_demo.py to run the training experiment on Cora dataset and 3 epsilon values([0,1,10]) only. 

First, generate to-do tasks for this quick experiment
```
python run_exp_demo.py --rank 20  --num_seeds 5 --command train --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl --only_create_todos
```
Then, run the tasks 
```
python run_exp_demo.py --rank 20  --num_seeds 5 --command train --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl
```
There will be 10 to-do tasks generated in the run_exp_demo.py experiment, but the actual run will be 50 tasks, since each task will be executed for 5 times with 5 different random seeds. These tasks are expected to run in about 250 minutes (~4.2 hours) on one machine. After parsing the results following the instruction above, the results should be similar to the results provided in expected_results/result_cora.csv (lines corresponding to epsilon values 0, 1 and 10). 

We reported the mean values in the results for plotting the figures in our paper, and the models and column names correspond as follows:
```
gcn_mean -> GCN
gcn_rank20_mean -> Eclipse
mlp_mean -> MLP
mmlp_nl2_mean -> LPGNet
```
Note that since MLP training does not involve adjacency matrix, and the epsilon values are applied to add DP noise to adjacency matrix only, our code automatically assigns value of -1 to MLP experiments with non-zero epsilon values.

### Attacking the Trained Models
---

Note: DO NOT delete the results-dir obtained during training the models, as the trained model information is saved in results-dir and is needed for loading the model and running the attack on the trained models.
#### Generating to-do tasks for attack ####
Before attacks for transductive start, generate the complete set of to-do tasks,
```
python run_exp.py --rank 20  --num_seeds 5 --command attack --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl --only_create_todos
```
Before attacks for inductive start, generate the complete set of to-do tasks,
```
python run_exp.py --rank 20  --num_seeds 5 --num_epochs 200 --command attack --datasets TwitchES --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl --only_create_todos
```
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

#### Parse attack results ####
To parse results to get attack AUC (Area Under Curve) scores, provide the path to the directory with saved models ( "eval_" as prefix in src/ directory). Note that this script is used for both transductive and inductive.
```
python parser_ash_trans_attack.py --results_dir [results-dir]
```
The parsed results will be output in the results folder.

The expected result is stored in the expected_results folder.

Similar runtime (1525 tasks, ~128 hours on one machine) is expected for the attack experiment in our paper.

#### Quick Verification
Using the trained models from the quick verification experiment above, we can quicky verify if the attack experiement can be executed correctly on Cora dataset and 3 epsilon values.

First, generate to-do tasks for attacking the models
```
python run_exp_demo.py --rank 20  --num_seeds 5 --command attack --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl --only_create_todos
```
Then, attack the trained models
```
python run_exp_demo.py --rank 20  --num_seeds 5 --command attack --outdir [results-dir] --todos_dir [todos] --best_config_file best_config.pkl 
```
There are 10 attack tasks generated in the run_exp_demo.py experiment, but the actual run will be 100 tasks, because each task will be executed for 5 times with 5 different random seeds, and each random seed will run for 1 LPA attack and 1 Linkteller attack. These tasks are expected to run in about 500 minutes (~8.4 hours) on one machine. After parsing the results following the instruction above, the results should be similar to the results provided in expected_results/result_attack_cora_baseline_balanced.csv (lines corresponding to epsilon values 0, 1 and 10) for LPA attack, and expected_results/result_attack_cora_efficient_balanced.csv (lines corresponding to epsilon values 0, 1 and 10) for Linkteller attack.

Note that for MLP model with non-zero epsilon values, -1 will be automatically assigned.
