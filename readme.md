# Graph2Mat4aBN

Using Graph2Mat to predict the hamiltonian of amorphous Boron Nitride. Master thesis at ICN2.

### Autor: Ángel Lapeña López

### Supervisors: Andrei Voicu Tomut, Thomas Jean-François Galvani, Stephan Roche

# Set up

Currently, we are using a version of Graph2Mat that is not officially released yet. Thus, you **cannot** do `pip install graph2mat`. Instead, you have to install separately the repo of Graph2Mat, where the last version we need for this project is available, in your computer and then install it manually from there. 

You need Linux or MacOS. WINDOWS NOT SUPPORTED (this is because SISL library is not easy to install on windows).

Instructions:

### 1. Clone Graph2Mat4aBN's repo and create virtual enviroment
For example:
```
cd GitHub
git clone https://github.com/alapena/graph2mat4abn
cd graph2mat4abn
```

Now, create and activate a virtual enviroment with python 3.9. It's preferred to use conda (we did not test others).
```
conda create -n g2m4ab python=3.9
conda activate g2m4ab
```

### 2. Install graph2mat4abn

Execute
```
pip install -e .
```
The `-e` flag is for you to install the packages in an editable manner (easier to edit for devs). You are ready to start using the package!

### 3. Install the dataset

1. Copy the dataset of Jaime (SHARE_OUTPUTS_X_ATOMS) inside a `dataset` folder.

2. Execute the `eda` notebook (exploratory_data_analysis) to exclude the carbons (I'm sorry it's not a clean notebook).

### 2. Install Graph2Mat from github.

Clone Graph2Mat's repo, e.g,
```
cd ..
git clone https://github.com/BIG-MAP/graph2mat
cd graph2mat
pip install .
```

Now you have installed the latest version of Graph2Mat. Now you need to install the other dependencies.

<!-- ### 3. Install torch
Now you have to install torch. This is done separately because for some reason CUDA support fails if you try to install it through the dependencies. Also you need torch version 2.5 or lower.
```
pip3 install torch==2.5.0 torchvision torchaudio
``` -->


# Repo structure

In order to use the repo, you need to copy all the structures (each of the `SHARE_OUTPUTS_X_ATOMS`...) into a folder called `dataset`.

The folder `scripts` contains exampels of scripts to train models, to perform inference and to plot model's predictions.

The `config.yaml` file is very important. All the parameters that caracterize a training are set here. For now, the most important is the `results_dir` parameter. Set it to `results/first_try` and go on.

# The `config.yaml` file  
This file sets the configuration of each training. It has the following parameters:

- **`debug_mode`**: setting this to `true` accelerates some calculations, skips heavy steps, and is mainly used to test or debug scripts.  
- **`device`**: selects the hardware device to run on. `"cpu"` forces CPU training, `"cuda:0"` uses the first GPU.  
- **`results_dir`**: directory where training results, logs, and checkpoints will be stored.  
- **`trained_model_path`**: path to a pre-trained model checkpoint. If `null`, training starts from scratch.  

### Dataset parameters
- **`train_split_ratio`**: fraction of the dataset used for training (the rest is used for validation).  
- **`stratify`**: ensures class proportions are preserved when splitting. 
- **`seed`**: random seed for reproducibility in dataset splitting and shuffling.  
- **`num_unique_z`**: number of unique atomic numbers (atom species) used to speed up basis calculations.  
- **`use_only`**: list of dataset subfolder suffixes to include (e.g., only using `"2_ATOMS"` data). Must match the naming format of the dataset folder (SHARE_OUTPUT_**X_ATOMS**).
- **`exclude_carbons`**: if `true`, removes carbon-containing samples from the dataset (actually, it uses instead the dataset previously generated with no carbons included). 
- **`custom_dataset`**: set to `true` if loading a custom dataset format instead of all of it. You must code it manually in the train.py.  
- **`extra_custom_validation`**: adds an extra validation custom split. You must code it manually in the train.py.  

### Orbitals
- **`orbitals`**: dictionary mapping quantum shell index (`1`–`8`) to the number of orbitals considered per shell.  

### Environment representation

This are the MACE parameters.

- **`num_interactions`**: number of message-passing layers in the network.  
- **`correlation`**: order of message correlation, controlling higher-order interactions.  
- **`max_ell`**: spherical harmonics expansion order (angular resolution).  
- **`num_channels`**: size of hidden channels in the network (controls model capacity).  
- **`max_L`**: maximum symmetry order for messages, must match irreps structure.  
- **`r_max`**: cutoff distance (in Å) for local environment interactions.  
- **`num_elements`**: number of distinct chemical elements in the dataset.  
- **`atomic_energies`**: per-element reference energies (for normalization). Must match `atomic_numbers` shape.  
- **`avg_num_neighbors`**: expected average number of neighboring atoms, used for normalization.  
- **`atomic_numbers`**: list of atomic numbers representing the chemical species in the dataset (e.g., `[5, 7]` = Boron, Nitrogen).  
- **`num_bessel`**: number of radial basis functions (Bessel functions).  
- **`num_polynomial_cutoff`**: smoothness parameter for the radial cutoff function.  
- **`hidden_irreps`**: irreducible representation (irrep) structure of hidden layers, describing equivariant feature spaces.  
- **`MLP_irreps`**: irreps used in the readout MLP before prediction. Controls dimensionality.  
- **`gate`**: non-linear activation function used in the readout layer (e.g., `silu`).  

### Model parameters
- **`readout_per_interaction`**: if `true`, performs readout after each interaction block. Otherwise, only at the end.  
- **`preprocessing_nodes`**: module used for node preprocessing (e.g., `E3nnInteraction`).  
- **`preprocessing_edges`**: module used for edge preprocessing (e.g., `E3nnEdgeMessageBlock`).  
- **`node_operation`**: defines the operation block applied at nodes (e.g., `E3nnSimpleNodeBlock`).  
- **`edge_operation`**: defines the operation block applied on edges (e.g., `E3nnSimpleEdgeBlock`).  

### Optimizer
- **`lr`**: initial learning rate used by the optimizer (epoch 0).  
- **`initial_lr`**: explicitly sets the initial learning rate when resuming training from a checkpoint.  

### Scheduler
- **`type`**: learning rate scheduler type (e.g., `ReduceLROnPlateau`).  
- **`step`**: metric used to decide scheduler steps (`train_loss`, `val_loss`, or `None`).  
- **`kwargs`**: configuration arguments for the scheduler (e.g., `mode`, `factor`, `patience`, `min_lr`).  

### Trainer
- **`batch_size`**: number of samples per training batch.  
- **`checkpoint_freq`**: frequency (in steps) to save checkpoints.  
- **`live_plot`**: enables real-time loss/metric plotting.  
- **`live_plot_freq`**: frequency (in epochs) to update live plots.  
- **`live_plot_matrix`**: enables live plotting of predicted matrices.  
- **`live_plot_matrix_freq`**: frequency to update matrix plots.  
- **`live_plot_matrices_num`**: number of matrices to visualize live.  
- **`num_epochs`**: maximum number of training epochs.  
- **`loss_function`**: loss function used (`elementwise_mse`, `block_type_mse`, `block_type_mse_threshold`).  
- **`keep_in_memory`**: if `true`, loads dataset into memory instead of reading from disk.  
- **`rotating_pool`**: enables rotating batch sampling from a subset pool.  
- **`rotating_pool_size`**: size of the rotating dataset pool.  
- **`matrix`**: type of matrix to predict (`hamiltonian`, `tim`, or `overlap`).  


# The train.py file

In order to train a model using graph2mat, you need to follow the following steps:
1. Get the paths to all structures
2. Create the BasisTableWithEdges object (made from a list of PointBasis objects)
3. Create the MatrixDataProcessor object. This is the graph <--> matrix internal "translator"
4. Generate the dataset (`TorchBasisMatrixDataset`) using the structure (sisl geometry) and target hamiltonians (sparse matrices). ``BasisConfiguration.from_matrix()`` helps with this. This feature is not yet released in the official version of graph2mat; that's why we need to install the GitHub version.
5. We then use the ``InMemoryData`` to keep all the dataset loaded in memory. You can either use ``RotatingPoolData``. Or none of them. Tune this in ``config.yaml``.
6. Initialize the models and the trainer.

Usually, initializing MACE is expensive (1-2 minutes). If you use large structures (>~64), the dataset generation is also expensive (~2 s per structure).

# The results folder

One you execute your first training with the default config.yaml,

```
python3 scripts/train.py
```

the folder `results/test/` will be created. In this folder, the following results will be saved:
- The model saved at each epoch specified by `checkpoint_freq` (`config.yaml`)
- The dataset used (train_dataset.txt, val_dataset.txt)
- The ``config.yaml`` used for this training.
- A plot `memory_usage.png` to monitor the CUDA RAM usage
- An interactive plot `plot_loss.html` of the loss curves
- The models saved at the epochs where they had the lowest training or validation losses (``train_best_model.tar``, ``val_best_model.tar``)
- In case the model reaches a minimum on nodes or edges but not on global loss, it also saves ``train_edge_best_model.tar``, ``train_node_best_model.tar``, ``val_edge_best_model.tar`` or ``val_node_best_model.tar``.
