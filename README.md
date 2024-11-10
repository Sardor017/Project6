
# Wild Blueberry Yield Prediction

This project predicts wild blueberry yield using various features related to bee density, weather conditions, and pollination effectiveness. The objective is to explore the relationship between these factors and yield to develop a robust model capable of making accurate predictions.

## Project Structure

- **Project_6_moduls.py**: Contains utility functions and model pipelines for feature engineering, data cleaning, and model training.
- **Project_6_Visualization_and_Analyzing.ipynb**: Jupyter notebook for data visualization and exploratory data analysis.
- **Project_6_main.ipynb**: Jupyter notebook with the main pipeline for training and testing the model.
- **train_reg.csv**: Training dataset containing features and the target variable (`yield`) for model training.
- **test_reg.csv**: Test dataset for generating predictions using the trained model.

## Features

### Key Functions in `Project_6_moduls.py`

- **new_features**: Adds various engineered features related to bee density, rain density, pollination effectiveness, and temperature adjustments.
- **useful_features**: Extracts a subset of features for streamlined analysis, specifically:
  - `fruitset`, `seeds`, `Clone_FruitMass_Interaction`, `Adjusted_FruitMass`, `RainingDays`, `MinOfUpperTRange`, `fruitmass`, `MinOfLowerTRange`, `clonesize`
- **stacking_model_function**: Builds a stacking regression model using Huber Regressor, Random Forest, and Ridge as estimators.
- **check_nuns_nulls_infs**: Checks for NaNs, zero values, and infinite values to ensure data integrity.

## Getting Started

1. Clone the repository.
   ```bash
   git clone https://github.com/Sardor017/Project6.git
   ```

2. Run `Project_6_Visualization_and_Analyzing.ipynb` for data exploration and visualization.

3. Run `Project_6_main.ipynb` to train the model and generate predictions on the test data.

## Model Pipeline

The project employs a stacking regressor model combining:
- **Huber Regressor**: Robust to outliers with MinMax scaling.
- **Random Forest Regressor**: Ensemble technique for variance reduction.
- **Ridge Regressor**: Final estimator for stacking to improve model stability.

## Feature Engineering

The project uses the following key features for prediction: 
- `fruitset`, `seeds`, `Clone_FruitMass_Interaction`, `Adjusted_FruitMass`, `RainingDays`, `MinOfUpperTRange`, `fruitmass`, `MinOfLowerTRange`, and `clonesize`

## Usage

- Load the data from `train_reg.csv` and `test_reg.csv`.
- Run the pipeline to train and evaluate the model.
- Save predictions for test data as a `.csv` file.
