# Antimicro.ai

This code was developed by Dr. Fredrick Mutisya and Dr. Rachael Kanguha using Pfizer data as part of the Vivli data challenge. For more information on Vivli and the challenge, visit the link 
[Vivli AMR](https://amr.vivli.org/)

![AntiMicro.ai logo](https://github.com/fredmutisya/Antimicro.ai/blob/main/logos/antimicroai.png)

# Website Home Page

Here is the link to the webapp deployed via Streamlit
[AntiMicro.ai](https://antimicroai.streamlit.app/)

![Website](https://github.com/fredmutisya/Antimicro.ai/blob/main/logos/website.png)

# README- Data Preprocessing

We analysed the datasets using R version 4.3.1. Here we had to wrangle the data into a long format to make it machine learning compatible. The highlights of the analysis can be seen below

![Analysis](https://github.com/fredmutisya/Antimicro.ai/blob/main/logos/analysis.png)
The dataset had the time in years so timeseries methods were not possible. We also removed highly collinear variables. We utilized the CLSI interpretations and not the actual Minimum Inhibitory Concentration values because the values are variable for different antibiotics. In addition, a low MIC doesn't automatically translate to clinical use. Pharmcodynamic and Pharmacokinetic issues may limit its use. For more information visit [CLSI ](https://clsi.org)

### Spatio-Temporal trend of Methicillin Resistant Staph Aureus
![MRSA](https://github.com/fredmutisya/Antimicro.ai/blob/main/logos/MRSA.gif)

### Spatio-Temporal trend of Meropenem resistance
![Meropenem](https://github.com/fredmutisya/Antimicro.ai/blob/main/logos/Meropenem.gif)

### Family and Species of Bacteria

![Bacteria](https://github.com/fredmutisya/Antimicro.ai/blob/main/logos/species.png)

### Family and Species of Fungi

![Fungi](https://github.com/fredmutisya/Antimicro.ai/blob/main/logos/fungi.png)

### Antibiotic Classes

![Antibiotic classes](https://github.com/fredmutisya/Antimicro.ai/blob/main/logos/antibiotics.png)

### Minimum Inhibitory Concentration Interpretation

![MIC](https://github.com/fredmutisya/Antimicro.ai/blob/main/logos/mic.png)

# README- Web Application

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

The code uses the following datasets: These data sets were wrangled using Tidyr and dpylr in R to make them compatible with machine learning models.

- `amr_without_genes_ml.csv`
- `amr_with_genes_ml.csv`
- `antifungals_ml.csv`


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

### Data Cleaning

Before feeding data into a machine learning model, you must clean it to remove inconsistencies, errors, and missing values. Common data cleaning steps include:

Handling Missing Data: Decide how to handle missing values (e.g., remove rows, impute values).
Removing Duplicates: Check for and eliminate duplicate records if they exist.
Data Formatting: Ensure data types are correct (e.g., dates, numerical values).
Outlier Detection and Handling: Identify and address outliers that may skew your model.

### Data Exploration

Explore your data to gain insights and understand its characteristics. Common exploratory data analysis (EDA) tasks include:

Summary Statistics: Compute basic statistics like mean, median, and standard deviation.
Data Visualization: Create plots and charts to visualize relationships and patterns.
Correlation Analysis: Examine correlations between variables.

### Feature Engineering

Feature engineering involves creating new features or transforming existing ones to improve model performance. In the case of antibiotic resistance, you might:

Encode Categorical Variables: Convert categorical data into numerical form using techniques like one-hot encoding.
Scaling: Scale numerical features if they have different scales.
Feature Selection: Choose relevant features that contribute to the prediction task.

### Preprocessing

Different machine learning models may require specific data preprocessing steps. For instance:

Normalization: Normalize data for models like k-Nearest Neighbors (KNN).
Dimensionality Reduction: Apply techniques like Principal Component Analysis (PCA) for high-dimensional data.

If your dataset has class imbalance issues (e.g., more susceptible bacteria than resistant), consider techniques like oversampling, undersampling, or using different evaluation metrics.
### Iterations on model building

After data preparation, train your machine learning model using the training dataset. Select an appropriate algorithm based on your problem (e.g., classification or regression).
Evaluate the model's performance using the validation dataset. Common evaluation metrics for classification tasks include accuracy, precision, recall, F1-score, and ROC curves. For regression tasks, metrics like Mean Absolute Error (MAE) and Mean Squared Error (MSE) are used.

Iteratively adjust hyperparameters and make improvements to the model based on the validation results.

### Substituting the Antimicro.ai for your customised model

Download the model and substitute the pickle file in the cwd

# README- Machine learning code

```
# XGBoost Classifier for AMR Prediction

This repository contains code for training an XGBoost classifier to predict antimicrobial resistance (AMR) based on genetic data. The classifier is trained using a dataset of genetic features and corresponding susceptibility labels.

## Requirements

- Python 3.7 or higher
- pandas
- xgboost
- scikit-learn

## Installation

1. Clone the repository:

```
git clone <repository-url>
```

2. Install the required packages:

```
pip install pandas xgboost scikit-learn
```

## Usage

1. Prepare the dataset:

- Place the dataset file (`amr_with_genes_ml.csv`) in the same directory as the code.

2. Run the code:

- Open a terminal or command prompt.
- Navigate to the directory containing the code.
- Run the following command:

```
python code.py
```

3. Results:

- The code will train an XGBoost classifier on the dataset, perform predictions on a test set, and calculate the accuracy of the model.
- The accuracy score will be displayed in the console.

4. Example Prediction:

- The code includes an example prediction on a new set of genetic features (`example_data`).
- The prediction result will be displayed in the console.

## Model Persistence

- The trained XGBoost classifier is saved as a pickle file (`xgb_model_genes.pkl`) for later use.
- The pickle file can be loaded to make predictions on new data.

