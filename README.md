# Hempel et al. Replication/Extension Project

## Project Overview
This repository contains the workflow for replicating and extending the analyses from Hempel et al. It includes data preprocessing, model replication, and exploratory extensions.


## Replication Instructions
Note the instruction are desinged for replicaiton on a PC. This was trained on a HPC cluster do you can utalize the slurm files to run the jobs as well but you have to change them to configure to your specific HPC.
### 1. Obtain Dataset
The dataset required for this project is available at the following OneDrive link:  
[Download Dataset](https://1drv.ms/u/c/3b10ceb52a1a5357/IQAEsw_gKL0mQoc4Pf_lEjY1AXF1ssJOb8UMiVpPra1l1io?e=TZY5vy)
Download the dataset onto your local machine. It is in the form of a zip file. Typically it will be downloaded to your Downloads directory.

### 2. Clone the repository 
To clone the repo, navigate to a directory where you want to place it and execute the command:
```bash
git clone https://github.com/jakepatock/comp_biomed_final_project
``` 
Now that you have the repo navigate into the repo with command:
```bash
cd comp_biomed_final_project
```
Finally, move the mimic-iv-3.1.zip file from the local location it was downloaded to on your machine, to the "dataset" folder in the repo we just cloned.
The dataset folder should not be empty anymore as it contains the zip file called "mimic-iv-3.1.zip".
Extract the zip file utalizing any method you would like. If you are on Linux or macOS ensure you are in the repo's root directory and use
```bash
unzip dataset/mimic-iv-3.1.zip -d dataset/
```
This will results in a directory called mimic-iv-3.1 being created in the dataset directory. This concludes the setting up of the repository. 

### 3. Configure Environment
To ensure compatibility with the dependencies in this project we will install the packages found in the requirements.txt in the root directory. To create this environment, create a new Python environment with Anaconda or any method you prefer. The example below uses Anaconda to initialize a fresh Python 3.11 environment and install the requirments file.

```bash
# 1. Create a new conda environment with Python 3.11
conda create -n comp_biomed python=3.11 -y

# 2. Activate the new environment
conda activate comp_biomed

# 3. Install all packages from requirements.txt
pip install -r requirements.txt
```


### 4. Data Preparation
To complete the preprocess in this dataset we will run the python file called preprocessing_script.py found in the scripts directory. This script takes no arguments. I will generate the figures replicated from Hempel et al. From the repo's root directory run command:
```bash
python scripts/preprocessing_script.py
```
This will take around 40 minutes on a pc and will save the preprocessed dataset in the dataset directory in a file called "mean_dataset.csv"

### 5. Modeling workflows
This step is the modeling workflow. Figure 3 in the report shows the workflow this script executes. 
First, we will address the classification task of prediction if the LOS is less than or greater than 4 days.
Execute this command to rerun this pipeline:
```bash
python scripts/cls_hyperpara_optim.py
```
This workflow will generate files found in the results directory. It will make a json file of the best hyperparameters found for each model in results\best_cls_hyperparams.json and test results in results\cls_results.csv.

Second, we will address the short regression task of predicting the continous LOS the subset dataset of 1 <= LOS < 4. To run this pipeline we will use the script called reg_hyperpara_optim.py found in the scripts folder. This file takes on argument called --reg_dataset. If you want do execute the short regression task execute the command:
```bash
python scripts/reg_hyperpara_optim.py --reg_dataset short
``` 

Thirdly, to run simply run the long regresion task of the full dataset (1 <= LOS < 21) change the --reg_dataset to long. Execute this command:
```bash
python scripts/reg_hyperpara_optim.py --reg_dataset long
``` 
Both scripts will save their optimzied hyperparameters to json files named best_short_reg_hyperparams.json and best_long_reg_hyperparams.json in the results directory. They will also save the test results to short_reg_results.csv and long_reg_results.csv also in the test directory.

Finally, we will run the stepwise workflow. This file takes no arguments and will generate the confusion matricies for the classificaitons models as well. To run this workflow execute:
```bash
python scripts/stepwise_modeling.py
```
This workflow will generate the 4 different sets of stepwise regression optimal hyperparameters using the 
