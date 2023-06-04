# MAEC - L04-Music-Generation
Repository for Music and Acoustics Engineering Capstone 2022/23 Course

## Environment

### Setup
1. Download and install [PyCharm IDE](https://www.jetbrains.com/pycharm/download/#section=linux).
2. Install [Conda](https://conda.io/projects/conda/en/stable/user-guide/install/index.html)
3. Install [Cuda Toolkit](https://docs.nvidia.com/cuda/) if you want GPU support
   - [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
   - [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
   - [macOS](https://developer.nvidia.com/nvidia-cuda-toolkit-11_7_0-developer-tools-mac-hosts)
4. Create a new PyCharm project from this GIT repository.
    - From PyCharm's Welcome Screen &rarr; Get from VCS &rarr; Paste the link to this repository
5. Create the environment from the `environment.yml` file:
   - `. ./setup-environment.sh`
6. Activate the new environment:
   - `conda activate ./venv`
7. Configure the new environment as Python interpreter in PyCharm:
   - Settings &rarr; Project Interpreter &rarr; Add Interpreter &rarr; Add Local Interpreter &rarr; Conda Environment 
   &rarr; &rarr; Use Existing Environment

### Update
If you need a new package while developing, just update the content of the environment.yml file and run:
```shell script
conda env update --prefix ./venv --file environment.yml  --prune
```
More info on this process can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#updating-an-environment).

## Scripts

Create your `/config/config.ini` file from `/config/config.ini.sample`.
All scripts configurations are specified in this file.

If you want to launch a script in a terminal and not with PyCharm's run configurations, remember to add the project's 
root directory to the path.

```shell script
cd /path/to/project/root && export PYTHONPATH=.
```

### Dataset Creation

```shell script
python ./scripts/create_dataset.py
```

### Data Preprocessing

```shell script
python ./scripts/preprocess_dataset.py --alsologtostderr
```

### Model training

```shell script
python ./scripts/train_model.py
```
