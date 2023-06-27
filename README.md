# MAEC - L04-Music-Generation
Repository for Music and Acoustics Engineering Capstone 2022/23 Course

## Abstract

[TODO]

## Environment

### Setup
1. Download and install [PyCharm IDE](https://www.jetbrains.com/pycharm/download/#section=linux).
2. Install [Conda](https://conda.io/projects/conda/en/stable/user-guide/install/index.html)
3. If running on Linux, install Magenta dependencies. Ubuntu/Debian based dìstros:
   - `sudo apt-get install build-essential libasound2-dev libjack-dev portaudio19-dev`
4. Create a new PyCharm project from this GIT repository.
    - From PyCharm's Welcome Screen &rarr; Get from VCS &rarr; Paste the link to this repository
5. Create the environment from the `environment.yml` file:
   - `. ./setup-environment.sh`
6. Configure the new environment as Python interpreter in PyCharm:
   - Settings &rarr; Project Interpreter &rarr; Add Interpreter &rarr; Add Local Interpreter &rarr; Conda Environment 
   &rarr; &rarr; Use Existing Environment
7. Download Magenta checkpoints that you need from [here](https://github.com/magenta/magenta/blob/main/magenta/models/music_vae/README.md#generate-script-w-pre-trained-models)

### Update
If you need a new package while developing, just update the content of the environment.yml file and run:
```shell script
conda env update --prefix ./venv --file environment.yml  --prune
```
More info on this process can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#updating-an-environment).

## Config

[TODO]

## Scripts

Create your `/config/config.ini` file from `/config/config.ini.sample`.
All scripts configurations are specified in this file.

If you want to launch a script in a terminal and not with PyCharm's run configurations, remember to add the project's 
root directory to the path.

```shell script
cd /path/to/project/root && export PYTHONPATH=.
```

### Sampling

[TODO]

### Complexities

[TODO]

### Evaluation

