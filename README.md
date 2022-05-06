# DIP2022_PKU
Class project of PKU Digital Image Processing 2022.

## Directory

- HistEq.py: HistEq Algorithm
- Morph.py: Morph Algorithm
- Sharpen.py: Sharpen Algorithm
- run.py: launch scripts for tasks
- app.py: scripts for demo app website
- utils.py: utils func including cvtColor, resize.
- dataset: directory containing all results of three tasks
    - HistEq
    - Morph
    - Sharpen
- results: directory containing input images of three tasks
    - HistEq
    - Morph
    - Sharpen
- scripts: directory containing all scripts to run task
- templates: html templates for website



## Environment Setup
```shell
conda create -n img2022 python=3.8
conda activate img2022
cd DIP2022_PKU
pip install -r requirements.txt
```

## Usage

### Run Source Code
To run single task, please run following scripts:
```shell
sh scripts/run_histeq.sh
sh scripts/run_morph.sh
sh scripts/run_sharpen.sh
```
In ```scripts/run_histeq.sh```, ```scripts/run_morph.sh```, ```scripts/run_sharpen.sh```, user can specify ```IMG_NAME``` and ```IMG_TYPE``` to choose an image to be processed. Make sure that the image exist in ```./dataset/TASK_NAME/```. 
For example, to run ```scripts/run_histeq.sh```, user need to make sure that the image exist in ```./dataset/HistEq/```. 

If user wants to process all images in dataset for one task, please set ```IMG_NAME="all"```. 

To run all tasks with all images, please run following scripts:
```shell
sh scripts/run_all.sh
```

### Run Demo
To run demo app in website, please run following scripts:
```shell
sh run_demo.sh
```