Overview This repository contains code for a simulation framework that implements multi-agent deep reinforcement learning for task offloading and resource allocation.

Repository Structure 

### env_models/

– base_env.py: Contains the base environment class. 

– noma_environment.py: Inherits from base_env, handling NOMA-related features. 

– mec_environment.py: Inherits from noma_environment, extending MEC logic. 

### general_configs/

– bs_config.json: Configures the base station parameters (num_sc, sc_capacity, es_capacity).

– ue_config.json: Configures the user equipment (task_deadline, task_arrival_probability, task_type_probability).

– env_config.json: Configures the environment (coherence_time, n_sbs). 

### ppo/ 

– ppo_config.json and ppo_offloading_config.json: Hyperparameters for the proposed mechanism

### isac/ 

– isac_config.json isac_offloading_config.json: Hyperparameters for the MAISAC benchmark. 

### duel_ddqn/ 

– duel_ddqn_config.json adn duel_ddqn_offloading_config.json: Hyperparameters for the MADQN benchmark. 

### maddpg/ 

– maddpg_config.json and maddpg_offloading_config.json: Hyperparameters for the MADDPG (BS as an Agent) benchmark.

### maddpg_device/

– maddpg_device_config.json and maddpg_device_offloading_config.json: Hyperparameters for the MADDPG (ID as an Agent) benchmark. 

### aa_test_ppo/ 

– training/: Scripts (ppo_1.py through ppo_10.py) for training the proposed mechanism. 

– testing/: Scripts (ppo_testing_1.py through ppo_testing_10.py) for testing the trained models. 

### saved_models/: Directory where trained models are stored (by default). 

### sim_results/: Contains MATLAB loaders and figures for analyzing training/testing outcomes.

### Important Configuration Notes 

### bs_config.json 

– num_sc: Number of subchannels. 

– sc_capacity: Subchannel capacity. 

– es_capacity: Number of tasks an edge server can process simultaneously. 

### ue_config.json 

– task_deadline: Deadline (in time slots) for tasks.

– task_arrival_probability: Probability that a new task arrives in each time slot. 

– task_type_probability: Probability that a task is delay-sensitive. 

### env_config.json 

– coherence_time: Channel coherence time (e.g., 0.1 or 1 second).

– n_sbs: Number of base stations.

# How to Run Step 

## Step 1: Naming the Run

Before training, set a name in the corresponding algorithm’s config file.

Example: proposed_sbs_2, isac_sbs_2, madqn_sbs_2, etc.
(for chosen names, please refer to matlab files)

## Step 2: Hyperparameter Tuning

For each mechanism, tune hyperparameters in the respective config files (e.g., ppo_config.json, ppo_offloading_config.json).

### Important hyperparameters: layer_N, batch_size, lr, hidden_size, actor_lr, critic_lr.

### Benchmarks usually need more care to adjust to environment changes.

## Step 3: Training

For the proposed mechanism, run the scripts in aa_test_ppo/training/ (ppo_1.py through ppo_10.py).

The chosen run name (e.g., proposed_sbs_1) will create a directory in saved_models/ to store trained models.

### Example filenames:
noma_proposed_sbs_12_actor.pt

noma_proposed_sbs_12_critic.pt

offloading_proposed_sbs_12_actor.pt

offloading_proposed_sbs_12_critic.pt

## Step 4: Testing

Once a model shows good performance in training (say, ppo_2.py), edit ppo_config.json and ppo_offloading_config.json: chosen_env_idx: "2"

Then run aa_test_ppo/testing/ scripts (ppo_testing_1.py through ppo_testing_10.py).

## Step 5: Results Visualization

Open MATLAB and navigate to sim_results/.

Right-click the loaders/ folder → “Add to Path” → “Selected Folders and Subfolders”.

For training plots: open figures/single_train.m, set algo_name = "proposed_sbs_2", then run.

For testing plots: open figures/single_test.m, set algo_name = "proposed_sbs_2_best", then run.

## Model Saving

Due to numerous simulation configurations, we did not saved the pretrained models, so you need to perform training and testing.

Notes on Code Clarity

Some files include ChatGPT-generated explanations to enhance readability. Comments have been added for better maintainability and understanding of the logic.
