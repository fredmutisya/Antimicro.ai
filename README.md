# Antimicro.ai

This code was developed by Dr. Fredrick Mutisya and Dr. Rachael Kanguha using Pfizer data as part of the Vivli data challenge.

# README

This README file provides information about the packages and datasets used in the code provided.

## Packages

The following packages are imported in the code:

- `pandas`: A powerful data manipulation library.
- `xgboost`: An optimized gradient boosting library for machine learning.
- `sklearn.model_selection`: Contains the `train_test_split` function for splitting data into training and testing sets.
- `sklearn.metrics`: Contains the `accuracy_score` function for evaluating the accuracy of a classification model.
- `sklearn.impute`: Contains the `SimpleImputer` class for handling missing values in data.
- `joblib`: A package for saving and loading models.

## Datasets

The code uses the following datasets:

- `amr_without_genes_ml.csv`: This dataset contains the antimicrobial resistance (AMR) data without genotypic information. It is loaded using the `pd.read_csv` function and stored in the `data` variable.

## Streamlit and Additional Packages

The code also uses Streamlit, a Python library for creating interactive web applications. The following additional packages are imported:

- `streamlit`: The main package for creating Streamlit applications.
- `numpy`: A package for scientific computing with Python.
- `os`: A package for interacting with the operating system.
- `joblib`: A package for saving and loading models.
- `pickle`: A package for serializing and deserializing Python objects.
- `requests`: A package for making HTTP requests.
- `streamlit_option_menu`: A custom Streamlit component for creating option menus.
- `streamlit_extras.switch_page_button`: A custom Streamlit component for creating buttons to switch between pages.
- `streamlit_lottie`: A custom Streamlit component for rendering Lottie animations.
- `pandas_profiling`: A package for generating interactive exploratory data analysis reports.
- `streamlit_pandas_profiling`: A custom Streamlit component for displaying pandas profiling reports.
- `pycaret.classification`: A package for automating machine learning workflows.
- `pull`: A function from the `pycaret.classification` module for retrieving the processed data.
- `save_model`: A function from the `pycaret.classification` module for saving the best model.

## Animation

The code includes animations using Lottie files. Lottie is a library for rendering animations in web applications. The Lottie files are loaded using the `load_lottiefile` function and displayed using the `st_lottie` function from the `streamlit_lottie` package.

## Instructions and Option Menu

The code includes instructions and an option menu in the sidebar using Streamlit components. The instructions provide guidance on how to use the different features of the application. The option menu allows users to switch between different pages within the application.

## Antibacterial and Antifungal Predictors

The code includes sections for the AI Antibacterial Predictor and the AI Antifungal Predictor. These sections allow users to input predictor variables and make predictions using the trained models. The predictor variables are selected through dropdown menus, and the predictions are displayed based on the selected variables.

## Build Your Own AI Model

The code includes a section for users to build their own machine learning model. Users can upload their own datasets, perform exploratory data analysis, and train a machine learning model using the `pycaret.classification` package. The section provides instructions on how to perform data analysis, select machine learning settings, compare models, and download the trained model.

## About AntiMicro.ai

The code includes a section providing information about AntiMicro.ai, its purpose, and the developers behind it.

Please note that this is a summary of the code provided, and further details can be found within the code itself.
