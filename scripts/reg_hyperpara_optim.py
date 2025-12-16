# Standard library
import argparse
import ast
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path

# Third-party: Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Third-party: Machine learning - preprocessing
from sklearn.preprocessing import LabelEncoder, RobustScaler

# Third-party: Machine learning - model selection
from sklearn.model_selection import KFold, train_test_split

# Third-party: Machine learning - models
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
import xgboost as xgb

# Third-party: Machine learning - metrics
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error
)
from imblearn.metrics import geometric_mean_score

# Third-party: Hyperparameter optimization
import optuna

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

def to_multihot(chapters, size=23):
    vec = np.zeros(size, dtype=int)

    for idx in range(size):
        vec[idx] = chapters.count(idx)
    return vec

def merge_rare_onehots(df, rare_dict):
    for new_col, rare_list in rare_dict.items():
        df[new_col] = df[rare_list].max(axis=1)
        df.drop(columns=rare_list, inplace=True)
    return df

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

def reg_cross_val(model, input_features, reg_labels, n_folds):
    # Init the folds
    cv_folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Init list for fold scores 
    fold_scores = np.zeros((n_folds, 4))

    # For each fold 
    for idx, (train_idx, val_idx) in enumerate(cv_folds.split(input_features)):
        # Parition to val and train 
        train_input, val_input = input_features[train_idx], input_features[val_idx]
        train_label, val_label = reg_labels[train_idx], reg_labels[val_idx]

        # Isolation Forest 
        iso = IsolationForest(contamination="auto", random_state=seed, n_estimators=100)
        train_outlier_mask = iso.fit_predict(train_input)
        keep_mask = train_outlier_mask == 1
        train_input = train_input[keep_mask]
        train_label = train_label[keep_mask]

        # Init standard scaler 
        std_scaler = RobustScaler()
        # Scaling features using training parition as mu and sig
        scaled_train_input = std_scaler.fit_transform(train_input)
        scaled_val_input = std_scaler.transform(val_input)

        # Label Standarization
        label_scaler = RobustScaler()
        scaled_train_label = label_scaler.fit_transform(train_label.reshape(-1, 1)).flatten()

        # Fit the models on the dataset 
        model.fit(scaled_train_input, scaled_train_label)
        
        # Predict the labels 
        scaled_val_preds = model.predict(scaled_val_input)

        unscaled_val_preds = label_scaler.inverse_transform(scaled_val_preds.reshape(-1, 1)).flatten()

        # Calculating Evaluation Metrics 
        rmse = root_mean_squared_error(val_label, unscaled_val_preds)
        mae = mean_absolute_error(val_label, unscaled_val_preds)
        mape = mean_absolute_percentage_error(val_label, unscaled_val_preds)
        r2 = r2_score(val_label, unscaled_val_preds)

        fold_scores[idx] = np.array([rmse, mae, mape, r2])

    mean_metrics = np.mean(fold_scores, axis=0)

    return model, {'rmse': mean_metrics[0], 'mae': mean_metrics[1], 'mape': mean_metrics[2], 'r2': mean_metrics[3]}

def train_test_model(model, train_features, test_features, train_labels, test_labels):
    # Isolation Forest 
    iso = IsolationForest(contamination="auto", random_state=seed, n_estimators=100)
    train_outlier_mask = iso.fit_predict(train_features)
    keep_mask = train_outlier_mask == 1
    train_features = train_features[keep_mask]
    train_labels = train_labels[keep_mask]

    # Init standard scaler 
    std_scaler = RobustScaler()
    # Scaling features using training parition as mu and sig
    train_input = std_scaler.fit_transform(train_features)
    test_input = std_scaler.transform(test_features)

    # Scaling labels 
    label_scaler = RobustScaler()
    scaled_train_labels = label_scaler.fit_transform(train_labels.reshape(-1, 1)).flatten()
    
    # Fit the models on the dataset 
    model.fit(train_input, scaled_train_labels)
    
    # Predict the labels 
    scaled_test_preds = model.predict(test_input)
    
    unscaled_test_preds = label_scaler.inverse_transform(scaled_test_preds.reshape(-1, 1)).flatten()

    # Calculating Evaluation Metrics 
    rmse = root_mean_squared_error(test_labels, unscaled_test_preds)
    mae = mean_absolute_error(test_labels, unscaled_test_preds)
    mape = mean_absolute_percentage_error(test_labels, unscaled_test_preds)
    r2 = r2_score(test_labels, unscaled_test_preds)

    return model, {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}

def linear_objective(trial, input_features, reg_labels):
    # Choose model type
    model_type = trial.suggest_categorical("model_type", ["Ridge", "Lasso", "ElasticNet"])
    
    # Regularization parameter
    alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
    
    # ElasticNet l1_ratio
    l1_ratio = None
    if model_type == "ElasticNet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    
    # Fit intercept
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    
    # Model initialization
    if model_type == "Ridge":
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=seed)
    elif model_type == "Lasso":
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, random_state=seed, max_iter=10000)
    else:  # ElasticNet
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, random_state=seed, max_iter=10000)
    
    # Cross-validation
    _, metrics = reg_cross_val(model, input_features, reg_labels, n_folds=5)
    
    return metrics["rmse"]

def svr_objective(trial, input_features, reg_labels):
    # Hyperparameters to tune
    C = trial.suggest_float("C", 1e-3, 5, log=True)
    kernel = "rbf"
    tol = trial.suggest_float("tol", 1e-4, 1e-1, log=True)
    epsilon = trial.suggest_float("epsilon", 1e-4, 1, log=True)
    gamma = "scale"
    
    # Initialize the model
    model = SVR(
        C=C,
        kernel=kernel,
        gamma=gamma,
        epsilon=epsilon,
        tol=tol,
    )

    # Cross-validation
    _, metrics = reg_cross_val(model, input_features, reg_labels, n_folds=5)

    # Return metric to optimize
    return metrics["rmse"]

def rf_objective(trial, input_features, reg_labels):
    # Hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=50)
    criterion = "squared_error"
    max_depth = trial.suggest_int("max_depth", 3, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    bootstrap = True


    # Initialize the model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        random_state=seed,
        n_jobs=-1
    )

    # Cross-validation
    _, metrics = reg_cross_val(model, input_features, reg_labels, n_folds=5)

    # Return metric to optimize
    return metrics["rmse"]

def xgb_objective(trial, input_features, reg_labels):
# Core hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 50)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)

    # Regularization
    reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
    min_child_weight = trial.suggest_float("min_child_weight", 1, 10.0, log=True)
    gamma = trial.suggest_float("gamma", 0, 1.0)

    # Subsampling
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

    # Model Init
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,

        tree_method="hist",
        device="cuda",
        n_jobs=-1,

        random_state=seed,
        verbosity=0
    )

    # Cross-validation
    _, metrics = reg_cross_val(model, input_features, reg_labels, n_folds=5)

    # Return metric to optimize (AUC, Accuracy, etc.)
    return metrics["rmse"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script example")
    parser.add_argument("--reg_dataset", type=str, required=True)
    args = parser.parse_args()

    # init seed
    global seed
    seed = 0

    repo_dir = Path(__file__).parent.parent
    dataset_dir = os.path.join(repo_dir, 'improved_dataset')
    results_dir = os.path.join(repo_dir, 'results')
    setup_logger(repo_dir)

    dataset = pd.read_csv(os.path.join(dataset_dir, 'mean_dataset.csv'))
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    dataset, cls_labels, reg_labels  = preprocess_dataset(dataset)

    if args.reg_dataset == 'short':
        dataset = dataset[cls_labels == 0]
        reg_labels = reg_labels[cls_labels == 0]
        cls_labels = cls_labels[cls_labels == 0]
    else:
        pass

    train_input_df, test_input_df, train_cls_df, test_cls_df, train_reg_df, test_reg_df = train_test_split(dataset, 
                                                                                                            cls_labels, 
                                                                                                            reg_labels, 
                                                                                                            test_size=0.2,
                                                                                                            random_state=seed,
                                                                                                            stratify=cls_labels.values)

    n_trials = 128

    linear_objective = partial(linear_objective, input_features=train_input_df.values, reg_labels=train_reg_df.values)
    linear_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    linear_study.optimize(linear_objective, n_trials=n_trials)

    svr_objective = partial(svr_objective, input_features=train_input_df.values, reg_labels=train_reg_df.values)
    svr_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    svr_study.optimize(svr_objective, n_trials=n_trials)

    rf_objective = partial(rf_objective, input_features=train_input_df.values, reg_labels=train_reg_df.values)
    rf_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    rf_study.optimize(rf_objective, n_trials=n_trials)

    xgb_objective = partial(xgb_objective, input_features=train_input_df.values, reg_labels=train_reg_df.values)
    xbg_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    xbg_study.optimize(xgb_objective, n_trials=n_trials)

    logging.info(f"")

    # Linear Regression
    best_linear_trial = linear_study.best_trial
    logging.info(f"Best Linear RMSE: {best_linear_trial.value}")
    logging.info(f"Best Linear hyperparameters: {best_linear_trial.params}")

    # SVR
    best_svr_trial = svr_study.best_trial
    logging.info(f"Best SVR RMSE: {best_svr_trial.value}")
    logging.info(f"Best SVR hyperparameters: {best_svr_trial.params}")

    # Random Forest
    best_rf_trial = rf_study.best_trial
    logging.info(f"Best RF RMSE: {best_rf_trial.value}")
    logging.info(f"Best RF hyperparameters: {best_rf_trial.params}")

    # XGBoost
    best_xgb_trial = xbg_study.best_trial
    logging.info(f"Best XGB RMSE: {best_xgb_trial.value}")
    logging.info(f"Best XGB hyperparameters: {best_xgb_trial.params}")

    best_paras_dict = {'linear': best_linear_trial.params, 'svr': best_svr_trial.params, 'rf': best_rf_trial.params, 'xgb': best_xgb_trial.params}

    with open(os.path.join(results_dir, f"best_{args.reg_dataset}_reg_hyperparams.json"), "w") as f:
        json.dump(best_paras_dict, f, indent=4)

    with open(os.path.join(results_dir, f"best_{args.reg_dataset}_reg_hyperparams.json"), "r") as f:
        best_paras_dict = json.load(f)

    # Init best models 
    # Linear Regression
    if best_paras_dict['linear']['model_type'] == "Ridge":
        best_linear_model = Ridge(alpha=best_paras_dict['linear']['alpha'], fit_intercept=best_paras_dict['linear']['fit_intercept'], random_state=seed)
    elif best_paras_dict['linear']['model_type'] == "Lasso":
        best_linear_model = Lasso(alpha=best_paras_dict['linear']['alpha'], fit_intercept=best_paras_dict['linear']['fit_intercept'], random_state=seed, max_iter=10000)
    else:  # ElasticNet
        best_linear_model = ElasticNet(alpha=best_paras_dict['linear']['alpha'], l1_ratio=best_paras_dict['linear']['l1_ratio'], fit_intercept=best_paras_dict['linear']['fit_intercept'], random_state=seed, max_iter=10000)

    # Linear Regression
    best_svr_model = SVR(**best_paras_dict['svr'], kernel='rbf', gamma="scale")

    # Random Forest
    best_rf_model = RandomForestRegressor(**best_paras_dict['rf'], random_state=seed, n_jobs=-1)

    # XGBoost
    best_xgb_model = xgb.XGBRegressor(**best_paras_dict['xgb'], tree_method="hist", device="cuda", random_state=seed, verbosity=0)

    best_linear_model, linear_results = train_test_model(best_linear_model, train_input_df.values, test_input_df.values, train_reg_df.values, test_reg_df.values)
    best_svr_model, svr_results = train_test_model(best_svr_model, train_input_df.values, test_input_df.values, train_reg_df.values, test_reg_df.values)
    best_rf_model, rf_results = train_test_model(best_rf_model, train_input_df.values, test_input_df.values, train_reg_df.values, test_reg_df.values)
    best_xgb_model, xbg_results = train_test_model(best_xgb_model, train_input_df.values, test_input_df.values, train_reg_df.values, test_reg_df.values)

    # Combine results into a list of dicts with model names
    cls_results_list = [
        {"model": "Linear", **linear_results},
        {"model": "SVR", **svr_results},
        {"model": "RandomForest", **rf_results},
        {"model": "XGBoost", **xbg_results}
    ]

    # Convert to DataFrame
    results_df = pd.DataFrame(cls_results_list)

    results_df.to_csv(os.path.join(results_dir, f"{args.reg_dataset}_reg_results.csv"), index=False)
