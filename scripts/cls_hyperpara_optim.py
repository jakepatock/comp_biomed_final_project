'''
Docstring for scripts.cls_hyperpara_optim:
This file reads in the preprocessed dataset, 
'''

# Standard library
import ast
import json
import logging
import os
import pickle
import sys
from functools import partial
from pathlib import Path

# Third-party: Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Third-party: Machine learning - preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Third-party: Machine learning - model selection
from sklearn.model_selection import StratifiedKFold, train_test_split

# Third-party: Machine learning - models
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

# Third-party: Machine learning - metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from imblearn.metrics import geometric_mean_score

# Third-party: Hyperparameter optimization
import optuna

# Logging Config Function 
def setup_logger(saving_dir):
    # Configuring logger
    log_file_path = os.path.join(saving_dir, 'training_history.log')

    # Remove the existing log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    # Set up basic logging to file
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )

    # Set up logging to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(console_handler)

# Maping a vector of numbers to multi-hot count vector 
def to_multihot(chapters, size=23):
    # Init size 
    vec = np.zeros(size, dtype=int)

    # Counting how many of each int are in the input 
    for idx in range(size):
        vec[idx] = chapters.count(idx)
    return vec

# Combining rare columns with highly imbalcned features to one
def merge_rare_onehots(df, rare_dict):
    for new_col, rare_list in rare_dict.items():
        df[new_col] = df[rare_list].max(axis=1)
        df.drop(columns=rare_list, inplace=True)
    return df

# Preprocessing pipeline 
def preprocess_dataset(dataset):
    # Label Encoder for M and F 
    label_encoder = LabelEncoder()
    # Encoding Labels 
    dataset['gender'] = label_encoder.fit_transform(dataset['gender'])

    # Reading in the string to python list
    dataset['diagnoses_chapters'] = dataset['diagnoses_chapters'].apply(ast.literal_eval)
    # Converting List of ICD-10 Chapters to multihot counts 
    dataset['diagnoses_chapters'] = dataset['diagnoses_chapters'].apply(to_multihot)

    # Adding ICD_chapters to all ICD features 
    chapter_column_names = [f'ICD_chapter_{i}' for i in range(23)]
    # Converting to numpy array, then converting back to df with each ICD in its own column
    chapters_df = pd.DataFrame(np.vstack(dataset['diagnoses_chapters'].tolist()), columns=chapter_column_names)

    # Cating the new ICD counts with the old df and dropping the oringial features 
    dataset = pd.concat([dataset, chapters_df], axis=1).drop(columns=['diagnoses_chapters'])

    # Mapping first_careunit to onehot and cating 
    careunit_dummies = pd.get_dummies(dataset['first_careunit'], prefix='first_careunit').astype(int)
    dataset = pd.concat([dataset, careunit_dummies], axis=1).drop(columns=['first_careunit'])

    # Mapping admission_type to onehot and cating 
    admission_type_dummies = pd.get_dummies(dataset['admission_type'], prefix='admission_type').astype(int)
    dataset = pd.concat([dataset, admission_type_dummies], axis=1).drop(columns=['admission_type'])

    # Mapping admission_location to onehot and cating 
    admission_location_dummies = pd.get_dummies(dataset['admission_location'], prefix='admission_location').astype(int)
    dataset = pd.concat([dataset, admission_location_dummies], axis=1).drop(columns=['admission_location'])

    # Getting CLS labels for this 
    classification_labels = (dataset['los'] > 4).astype(int)

    # Getting the reg labels 
    reg_labels = dataset.pop('los')

    # Cleaning rare columns to maintain stable training 
    rare_cols = [c for c in dataset.columns if dataset[c].mean() < 0.005]

    # Grabing name of rare columns 
    ICD_chapter_names = list(filter(lambda x: x.startswith("ICD_"), rare_cols))
    first_careunit_names = list(filter(lambda x: x.startswith("first_careunit_"), rare_cols))
    admission_type_names = list(filter(lambda x: x.startswith("admission_type_"), rare_cols))
    admission_location_names = list(filter(lambda x: x.startswith("admission_location_"), rare_cols))

    rare_groups = {
        'ICD_chapter_OTHER': ICD_chapter_names,
        'first_careunit_OTHER': first_careunit_names,
        'admission_type_OTHER': admission_type_names,
        'admission_location_OTHER': admission_location_names
        }
    # Aggregating rare columns 
    dataset = merge_rare_onehots(dataset, rare_groups)

    return dataset, classification_labels, reg_labels 

# Classification Cross valdiadtion function
def cls_cross_val(model, input_features, labels, n_folds):
    # Init the folds
    cv_folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Init list for fold scores 
    fold_scores = np.zeros((n_folds, 6))

    # For each fold 
    for idx, (train_idx, val_idx) in enumerate(cv_folds.split(input_features, labels)):
        # Parition to val and train 
        train_input, val_input = input_features[train_idx], input_features[val_idx]
        train_label, val_label = labels[train_idx], labels[val_idx]

        # Isolation Forest 
        iso = IsolationForest(contamination="auto", random_state=seed, n_estimators=100)
        train_outlier_mask = iso.fit_predict(train_input)
        keep_mask = train_outlier_mask == 1
        train_input = train_input[keep_mask]
        train_label = train_label[keep_mask]

        # Init standard scaler 
        std_scaler = StandardScaler()
        # Scaling features using training parition as mu and sig
        train_input = std_scaler.fit_transform(train_input)
        val_input = std_scaler.transform(val_input)

        # Fit the models on the dataset 
        model.fit(train_input, train_label)
        
        # Predict the labels 
        val_preds = model.predict(val_input)
        # Predicting 
        try:
            val_probs = model.predict_proba(val_input)[:, 1]
        except AttributeError:
            val_probs = model.decision_function(val_input)

        # Calculating Evaluation Metrics 
        fold_accuracy = accuracy_score(val_label, val_preds)
        fold_f1 = f1_score(val_label, val_preds)
        fold_balanced_accuracy = balanced_accuracy_score(val_label, val_preds)
        fold_mcc = matthews_corrcoef(val_label, val_preds)
        fold_gmean = geometric_mean_score(val_label, val_preds)
        fold_aucroc = roc_auc_score(val_label, val_probs)
        

        fold_scores[idx] = np.array([fold_accuracy, fold_f1, fold_balanced_accuracy, fold_mcc, fold_gmean, fold_aucroc])

    # Summarizing Scores over folds 
    mean_metrics = np.mean(fold_scores, axis=0)

    return model, {'accuracy': mean_metrics[0], 'f1': mean_metrics[1], 'balanced_accuracy': mean_metrics[2], 'mcc': mean_metrics[3], 'g_mean': mean_metrics[4], 'aucroc': mean_metrics[5]}

# Final model production
def train_test_model(model, train_features, test_features, train_labels, test_labels):
    # Isolation Forest 
    iso = IsolationForest(contamination="auto", random_state=seed, n_estimators=100)
    train_outlier_mask = iso.fit_predict(train_features)
    keep_mask = train_outlier_mask == 1
    train_features = train_features[keep_mask]
    train_labels = train_labels[keep_mask]

    # Init standard scaler 
    std_scaler = StandardScaler()
    # Scaling features using training parition as mu and sig
    train_input = std_scaler.fit_transform(train_features)
    test_input = std_scaler.transform(test_features)

    # Fit the models on the dataset 
    model.fit(train_input, train_labels)
    
    # Predict the labels 
    test_preds = model.predict(test_input)
    
    # Predicting 
    try:
        test_probs = model.predict_proba(test_input)[:, 1]
    except AttributeError:
        test_probs = model.decision_function(test_input)

    # Calculating Evaluation Metrics 
    accuracy = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds)
    balanced_accuracy = balanced_accuracy_score(test_labels, test_preds)
    mcc = matthews_corrcoef(test_labels, test_preds)
    gmean = geometric_mean_score(test_labels, test_preds)
    aucroc = roc_auc_score(test_labels, test_probs)

    return model, {'accuracy': accuracy, 'f1': f1, 'balanced_accuracy': balanced_accuracy, 'mcc': mcc, 'g_mean': gmean, 'aucroc': aucroc}

# Logistic Regression Objective Function 
def logistic_objective(trial, input_features, labels):
    # Solver
    solver = trial.suggest_categorical("solver", ["newton-cholesky", "lbfgs", "saga"])
    
    # Penalty and l1_ratio
    if solver in ["newton-cholesky", "lbfgs"]:
        penalty = 'l2'
        l1_ratio = None
    elif solver == "saga":
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        else:
            l1_ratio = None

    # Other Hyperparameters
    tol = trial.suggest_float("tol", 1e-6, 1e-2, log=True)
    C = trial.suggest_float("C", 1e-3, 1e9, log=True)
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    max_iter = 1000
    warm_start = trial.suggest_categorical("warm_start", [False, True])

    # Model initialization
    model = LogisticRegression(
        penalty=penalty,
        solver=solver,
        tol=tol,
        C=C,
        fit_intercept=fit_intercept,
        class_weight=class_weight,
        max_iter=max_iter,
        warm_start=warm_start,
        l1_ratio=l1_ratio,
        random_state=seed,
        n_jobs=-1
    )

    # Cross Val 
    _, metrics = cls_cross_val(model, input_features, labels, n_folds=3)
    
    return metrics["aucroc"]

# SVC Objective function 
def svc_objective(trial, input_features, labels):
    # Hyperparameters to tune
    C = trial.suggest_float("C", 1e-3, 5, log=True)
    kernel = "rbf"
    tol = trial.suggest_float("tol", 1e-4, 1e-1, log=True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    gamma = "scale"
    
    # Initialize the model
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        tol=tol,
        class_weight=class_weight,
        probability=False,
        random_state=seed
    )

    # Cross-validation
    _, metrics = cls_cross_val(model, input_features, labels, n_folds=3)

    # Return metric to optimize
    return metrics["aucroc"]

# Random Forest objective function 
def rf_objective(trial, input_features, labels):
    # Hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=50)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    max_depth = trial.suggest_int("max_depth", 2, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])


    # Initialize the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight=class_weight,
        criterion=criterion,
        random_state=seed,
        n_jobs=-1
    )

    # Cross-validation
    _, metrics = cls_cross_val(model, input_features, labels, n_folds=3)

    # Return metric to optimize
    return metrics["aucroc"]

# XGBoost objective function
def xgb_objective(trial, input_features, labels):
    # Core hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=50)
    max_depth = trial.suggest_int("max_depth", 2, 50)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)

    # Regularization
    reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
    min_child_weight = trial.suggest_float("min_child_weight", 1, 10.0, log=True)
    gamma = trial.suggest_float("gamma", 0, 1.0)

    # Subsampling
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

    # Class imbalance
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    # Model Init
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        class_weight=class_weight,

        tree_method="hist",
        device="cuda",
        n_jobs=-1,

        random_state=seed,
        verbosity=0)

    # Cross-validation
    _, metrics = cls_cross_val(model, input_features, labels, n_folds=3)

    # Return metric to optimize (AUC, Accuracy, etc.)
    return metrics["aucroc"]


# Init seed
global seed
seed = 0

# Configuring repo paths 
repo_dir = Path(__file__).parent.parent
dataset_dir = os.path.join(repo_dir, 'improved_dataset')
results_dir = os.path.join(repo_dir, 'results')
# Set up Logging 
setup_logger(repo_dir)

# Reading in dataset 
dataset = pd.read_csv(os.path.join(dataset_dir, 'mean_dataset.csv'))
dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

# Maping dataset to ML features 
dataset, cls_labels, reg_labels  = preprocess_dataset(dataset)

# Spliting to train and test 
train_input_df, test_input_df, train_cls_df, test_cls_df, train_reg_df, test_reg_df = train_test_split(dataset, 
                                                                                                        cls_labels, 
                                                                                                        reg_labels, 
                                                                                                        test_size=0.2,
                                                                                                        random_state=seed,
                                                                                                        stratify=cls_labels.values)

# Num trails for TPE hyperparameter optimization
n_trials = 128

# Optimize logistic regression 
logistic_objective = partial(logistic_objective, input_features=train_input_df.values, labels=train_cls_df.values)
logistic_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
logistic_study.optimize(logistic_objective, n_trials=n_trials)

# Optimize SVC
svc_objective = partial(svc_objective, input_features=train_input_df.values, labels=train_cls_df.values)
svc_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
svc_study.optimize(svc_objective, n_trials=n_trials)

# Optimize RF
rf_objective = partial(rf_objective, input_features=train_input_df.values, labels=train_cls_df.values)
rf_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
rf_study.optimize(rf_objective, n_trials=n_trials)

# Optimize XGB 
xgb_objective = partial(xgb_objective, input_features=train_input_df.values, labels=train_cls_df.values)
xbg_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
xbg_study.optimize(xgb_objective, n_trials=n_trials)

logging.info(f"")

# Logistic Regression
best_logistic_trial = logistic_study.best_trial
logging.info(f"Best Logistic AUC: {best_logistic_trial.value}")
logging.info(f"Best Logistic hyperparameters: {best_logistic_trial.params}")

# SVC
best_svc_trial = svc_study.best_trial
logging.info(f"Best SVC AUC: {best_svc_trial.value}")
logging.info(f"Best SVC hyperparameters: {best_svc_trial.params}")

# Random Forest
best_rf_trial = rf_study.best_trial
logging.info(f"Best RF AUC: {best_rf_trial.value}")
logging.info(f"Best RF hyperparameters: {best_rf_trial.params}")

# XGBoost
best_xgb_trial = xbg_study.best_trial
logging.info(f"Best XGB AUC: {best_xgb_trial.value}")
logging.info(f"Best XGB hyperparameters: {best_xgb_trial.params}")

# Packing optimal hyerparameters 
best_paras_dict = {'logistic': best_logistic_trial.params, 'svc': best_svc_trial.params, 'rf': best_rf_trial.params, 'xgb': best_xgb_trial.params}

# Saving them
with open(os.path.join(results_dir, "best_cls_hyperparams.json"), "w") as f:
    json.dump(best_paras_dict, f, indent=4)

# Loading them back
with open(os.path.join(results_dir, "best_cls_hyperparams.json"), "r") as f:
    best_paras_dict = json.load(f)

# Init best models 
# Logistic Regression
best_logistic_model = LogisticRegression(**best_paras_dict['logistic'], max_iter=1000, random_state=seed, n_jobs=-1)

# Logistic Regression
best_svc_model = SVC(**best_paras_dict['svc'], kernel='rbf', gamma="scale",  probability=False, random_state=seed)

# Random Forest
best_rf_model = RandomForestClassifier(**best_paras_dict['rf'], random_state=seed, n_jobs=-1)

# XGBoost
best_xgb_model = xgb.XGBClassifier(**best_paras_dict['xgb'], tree_method="hist", device="cuda", random_state=seed, verbosity=0)

# Training all models using optimal hyperparameters on the train parition and evaluating on the test parition 
best_logistic_model, logistic_results = train_test_model(best_logistic_model, train_input_df.values, test_input_df.values, train_cls_df.values, test_cls_df.values)
best_svc_model, svc_results = train_test_model(best_svc_model, train_input_df.values, test_input_df.values, train_cls_df.values, test_cls_df.values)
best_rf_model, rf_results = train_test_model(best_rf_model, train_input_df.values, test_input_df.values, train_cls_df.values, test_cls_df.values)
best_xgb_model, xbg_results = train_test_model(best_xgb_model, train_input_df.values, test_input_df.values, train_cls_df.values, test_cls_df.values)

# Combine results into a list of dicts with model names
cls_results_list = [
    {"model": "LogisticRegression", **logistic_results},
    {"model": "SVC", **svc_results},
    {"model": "RandomForest", **rf_results},
    {"model": "XGBoost", **xbg_results}
]

# Convert to DataFrame
results_df = pd.DataFrame(cls_results_list)

# Saving final test results 
results_df.to_csv(os.path.join(results_dir, "cls_results.csv"), index=False)
