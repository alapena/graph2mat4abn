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

Now, create the virtual enviroment. Then, activate it.
```
python -m venv .venv
source .venv/bin/activate
```

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

### 3. Install Graph2Mat4aBN's dependencies

```
cd ..
cd graph2mat4abn
pip install -e .
```

The `-e` flag is for you to install the packages in an editable manner (easier to edit for devs). You are ready to start using the package!

# Repo structure

In order to use the repo, you need to copy all the structures (each of the `SHARE_OUTPUTS_X_ATOMS`...) into a folder called `dataset`.

The folder `scripts` contains exampels of scripts to train models, to perform inference and to plot model's predictions.

The `config.yaml` file is very important. All the parameters that caracterize a training are set here. For now, the most important is the `results_dir` parameter. Set it to `results/first_try` and go on.