import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, train_test_split, cross_val_predict, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.tree import *
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import *
from sklearn.preprocessing import OrdinalEncoder
import optuna
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

import warnings

# Игнорировать все предупреждения
warnings.filterwarnings("ignore")
def new_features(data):
    # Bee Density Ratios:
    data['Honeybee_Bumble_Ratio'] = data['honeybee'] / (data['bumbles'] + 1e-5)
    data['Honeybee_Andrena_Ratio'] = data['honeybee'] / (data['andrena'] + 1e-5)
    data['Total_Bee_Density'] = data['honeybee'] + data['bumbles'] + data['andrena'] + data['osmia']

    # Weather Features:
    data['Rain_Density'] = data['RainingDays'] / (data['AverageRainingDays'] + 1e-5)
    data['Rain_Bee_Ratio'] = data['RainingDays'] / (data['Total_Bee_Density'] + 1e-5)

    #Pollination Effectiveness:
    data['Fruit_Bee_Ratio'] = data['fruitmass'] / (data['Total_Bee_Density'] + 1e-5)
    data['Seed_Bee_Ratio'] = data['seeds'] / (data['Total_Bee_Density'] + 1e-5)

    #Interactions with Clone Size:
    data['Clone_Bee_Ratio'] = data['clonesize'] / (data['Total_Bee_Density'] + 1e-5)
    data['Clone_Rain_Ratio'] = data['clonesize'] / (data['RainingDays'] + 1e-5)

    #Fruit-to-Seed Ratios:
    data['FruitMass_per_Seed'] = data['fruitmass'] / (data['seeds'] + 1e-5)
    data['Seed_per_FruitSet'] = data['seeds'] / (data['fruitset'] + 1e-5)

    #Rain-Adjusted Pollination Metrics:
    data['Adjusted_FruitSet'] = data['fruitset'] / (data['RainingDays'] + 1e-5)
    data['Adjusted_FruitMass'] = data['fruitmass'] / (data['RainingDays'] + 1e-5)

    #Clonesize Interactions:
    data['Clone_FruitMass_Interaction'] = data['clonesize'] * data['fruitmass']
    data['Clone_FruitSet_Interaction'] = data['clonesize'] * data['fruitset']

    #Fruit Efficiency Metrics:
    data['FruitSet_per_BeeDensity'] = data['fruitset'] / (data['honeybee'] + data['bumbles'] + data['andrena'] + data['osmia'] + 1e-5)
    data['FruitMass_per_BeeDensity'] = data['fruitmass'] / (data['honeybee'] + data['bumbles'] + data['andrena'] + data['osmia'] + 1e-5)

    #Temperature-Rain Interactions:
    data['Rain_Adjusted_MaxUpperT'] = data['MaxOfUpperTRange'] * data['RainingDays']
    data['Rain_Adjusted_MinLowerT'] = data['MinOfLowerTRange'] * data['RainingDays']

    #Bee Density Impact on Seeds:
    data['SeedDensity_per_Bee'] = data['seeds'] / (data['honeybee'] + data['bumbles'] + data['andrena'] + data['osmia'] + 1e-5)
    return data

def add_useful_features(data):
    data['Adjusted_FruitMass'] = data['fruitmass'] / (data['RainingDays'] + 1e-5)
    data['Clone_FruitMass_Interaction'] = data['clonesize'] * data['fruitmass']
    return data

def useful_features(data):
    data = add_useful_features(data)
    columns = ['fruitset', 'seeds', 'Clone_FruitMass_Interaction', 'Adjusted_FruitMass',
               'RainingDays', 'MinOfUpperTRange', 'fruitmass', 'MinOfLowerTRange', 'clonesize']
    if 'yield' in data.columns:
        columns.append('yield')
    return data[columns]

def stacking_model_function(X, y, test, seed=1):
    # Define the Huber Regressor pipeline with MinMaxScaler
    huber_pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('regressor', HuberRegressor(
            epsilon=1.1175188294038665,
            alpha=0.00012770419443760964
        ))
    ])

    # Initialize the RandomForestRegressor with specified parameters
    random_forest = RandomForestRegressor(
        random_state=seed,
        n_jobs=-1,
        n_estimators=180,
        max_depth=9,
        min_samples_split=2,
        min_samples_leaf=4,
        criterion='absolute_error'
    )

    # Define the Ridge pipeline with StandardScaler
    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.9777123273142374))
    ])

    # Define the Stacking Regressor using Ridge pipeline as the final estimator
    stacking_model = StackingRegressor(
        estimators=[
            ('random_forest', random_forest),
            ('huber', huber_pipeline)  # Huber pipeline with MinMax scaling
        ],
        n_jobs=-1,
        final_estimator=ridge_pipeline  # Ridge pipeline with Standard scaling
    )

    # Fit the stacking model on the full training data
    stacking_model.fit(X, y)

    # Make predictions on test data
    y_pred_test = stacking_model.predict(test)

    return y_pred_test

def check_nuns_nulls_infs(data):
    # Check for NaN values in each column
    nan_counts = data.isna().sum()
    print("NaN counts for each column:")
    print(nan_counts[nan_counts > 0])  # Display columns with NaN values only

    # Check for zero values in each column
    zero_counts = (data == 0).sum()
    print("\nZero counts for each column:")
    print(zero_counts[zero_counts > 0])  # Display columns with zero values only

    # Check for infinite values in each column
    inf_counts = np.isinf(data).sum()
    print("\nInfinite counts for each column:")
    print(inf_counts[inf_counts > 0])  # Display columns with infinite values only
