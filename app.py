import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import pickle
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page

#For animation
import json
from streamlit_lottie import st_lottie


#Import profiling
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

#Machine Learning
from pycaret.classification import setup, compare_models, pull , save_model


#Wide format
st.set_page_config(layout="wide")

#Load saved XGBoost model for bacteria with no genes
xgb_model_no_genes = pickle.load(open("\\xgb_model_no_genes.pkl", 'rb'))

#Load saved XGBoost model for bacteria with genes
xgb_model_genes = pickle.load(open("\\xgb_model_genes.pkl", 'rb'))


#Load saved XGBoost model from fungi
xgb_model_fungi = pickle.load(open("\\xgb_model_fungi.pkl", 'rb'))


# Option menu in the top bar


testing = option_menu(
    menu_title=None,
    options=["Home", "AI Antibacterial Predictor" , "AI Antibacterial Predictor (Genotypic data)", "AI Antifungal Predictor", "Build your own AI model", "About AntiMicro.ai"],
    icons=["house","calculator", "calculator","calculator" ,"bricks"],
    orientation='horizontal',
    styles={
        "container": {"padding": "0!important", "background-color": "#F7F9F9"},
        "icon": {"color": "#189AB4", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#B8DCE7",
        },
        "nav-link-selected": {"background-color": "#020659"},
    },
)

#Insert Animations

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_welcome = load_lottiefile("lottiefiles/welcome.json")
lottie_ai_1 = load_lottiefile("lottiefiles/ai_1.json")
lottie_prediction = load_lottiefile("lottiefiles/prediction.json")

#Home message
if testing == "Home":
    st_lottie(lottie_ai_1)
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 class="title">Welcome to Antimicro.ai</h1>', unsafe_allow_html=True)
    with st.sidebar:
        st.image("logos/antimicroai.png")
        st.info("""Welcome to Antimicro.ai, your ultimate companion in predicting antimicrobial resistance.""") 
        st.info("""Our cutting-edge AI-powered platform leverages advanced algorithms and machine learning techniques to provide accurate predictions and insights on antimicrobial resistance. Using Pfizer data obtained as aprt of the Vivli data challenge, an antibacterial and antifungal AI predictor was developed. With Antimicro.ai, you can analyze your own data, identify trends, and anticipate resistance patterns, enabling you to make informed decisions for effective treatment strategies. Whether you're a healthcare professional, researcher, or involved in public health, Antimicro.ai empowers you with the tools to combat antimicrobial resistance and contribute to a healthier future. Explore the power of AI and join us in the fight against antimicrobial resistance today.""")
        st.info("Developed by Dr. Rachael Kanguha and Dr. Fredrick Mutisya using Pfizer data as part of the Vivli data challenge")




if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

# Check if "Build your own AI model" is selected
if testing == "Build your own AI model":
    st.sidebar.title("Instructions")
    with st.sidebar:
        st_lottie(lottie_welcome)
        st.info("""Step 1: Data Analysis

Load the dataset into the AI model and perform exploratory data analysis (EDA) to understand the structure and characteristics of the data. Clean the data by handling missing values, outliers, and any necessary preprocessing steps. Split the dataset into training and testing subsets for model evaluation.

Step 2: Training the AI model

Select the best scoring model from this list
ada: Ada Boost Classifier
dt: Decision Tree Classifier
dummy: Dummy Classifier
et: Extra Trees Classifier
gbc: Gradient Boosting Classifier
lda: Linear Discriminant Analysis
lightgbm: Light Gradient Boosting Machine
lr: Logistic Regression
nb: Naive Bayes
qda: Quadratic Discriminant Analysis
rf: Random Forest Classifier
ridge: Ridge Classifier
svm: SVM - Linear Kernel
xgboost: Extreme Gradient Boosting                

Model metrics used for scoring include:
Accuracy: Represents the proportion of correctly classified instances out of the total instances in the dataset. It provides an overall measure of model performance.
Precision: Indicates the proportion of true positive predictions out of the total predicted positives. It measures the model's ability to avoid false positives.
Recall: Denotes the proportion of true positive predictions out of the total actual positives. It quantifies the model's capability to identify all relevant instances.
F1 Score: Harmonic mean of precision and recall. It balances the trade-off between precision and recall, providing an overall measure of classification performance.
Specificity: Indicates the proportion of true negative predictions out of the total actual negatives. It measures the model's ability to avoid false negatives.
ROC AUC (Receiver Operating Characteristic Area Under the Curve): Measures the model's ability to discriminate between positive and negative instances across various classification thresholds. It provides a single performance value across all thresholds.
Log Loss: Represents the logarithm of the likelihood function, quantifying the accuracy of predicted probabilities. It is commonly used for probabilistic classification tasks.
Runtime: Indicates the time taken for training and evaluating the model.               

Step 3: Model Download
You can download the best model as a Pickle file which is a portable version of the trained AI model               
                """)
    diy = st.sidebar.radio("Do it yourself", [ "Data analysis", "Types of Machine Learning models", "Download"])
    if diy == "Data analysis":
        st.title("If you have your own dataset , upload it here")
        file = st.file_uploader('')
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)
            try:
                st.title("Automated Exploratory Data Analysis")
                profile_report = df.profile_report()
                st_profile_report(profile_report)
            except (NameError, ValueError):
                st.info("No CSV file loaded. Please check the file type")
        
    
    if diy == "Types of Machine Learning models":
        st.title("Machine learning Models")
        file = st.file_uploader('')
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)
        
        target = st.selectbox("Choose your outcome variable", df.columns)
        if st.button("Train AI Model"):
            df_without_missing = df.dropna(subset=[target])
            setup(df_without_missing, target=target)
            setup_df = pull()
            st.info("Insert the Machine Learning settings")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is the AMR Machine Learning Model")
            st.dataframe(compare_df)
            best_model
            save_model(best_model, 'best_model')    


        
    if diy == "Download":
        with open ("best_model.pkl", 'rb') as f:
            st.download_button("Download the file", f, 'trained_model.pkl')











# ANTIBACTERIAL PREDICTOR

#Load the data
Variables = pd.read_csv("Variables.csv")


# Extract predictor names from the CSV file
countries = Variables["Country"].tolist()
source_sample = Variables["Source"].tolist()
bact_species = Variables["Species"].tolist()
antibiotics = Variables["Antibiotics"].tolist()
Year = Variables["Year"].tolist()
Phenotype = Variables["Phenotype"].tolist()
Antibiotic = Variables["Antibiotics"].tolist()
Speciality = Variables["Speciality"].tolist()



if testing == "AI Antibacterial Predictor":
    st.title('AI Antibacterial Predictor')
    html_temp = """
    <div style="background-color:#020659;padding:10px">
    <h2 style="color:white;text-aligh:center;"> Please input your predictor variables below  </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.sidebar.title("Instructions")
    with st.sidebar:
        st_lottie(lottie_prediction)
        st.info("""To input the predictor variables into an AI model for predicting susceptibility using the "AI Antibacterial Predictor," follow these steps:

Select the suspected bacterial species: Begin typing the name of the suspected bacterial species, and the dropdown list will prompt you with options in alphabetical order. For example, you can type "Achromobacter insolitus."

Select Country: Begin typing the name of the country from which the sample was taken, and the dropdown list will prompt you with options in alphabetical order. For instance, type "Argentina."

Select the Speciality: Begin typing the speciality associated with the sample, and the dropdown list will prompt you with options in alphabetical order. For example, type "Clinic / Office."

Select the source of the sample: Begin typing the source of the sample, and the dropdown list will prompt you with options in alphabetical order. For example, type "Abdominal Fluid."

Select the year the sample was taken: Choose the year when the sample was collected from the provided dropdown list. In this case, select "2023."

Select the Phenotype: Begin typing the phenotype associated with the sample, and the dropdown list will prompt you with options in alphabetical order. For example, type "ESBL."

Select the Antibiotic: Begin typing the name of the antibiotic for which you want to predict susceptibility, and the dropdown list will prompt you with options in alphabetical order. For instance, type "Amikacin."

Once you have provided all the necessary information, the AI model will use these predictor variables to make a prediction of susceptibility. The specific details and functionality of the AI model may vary depending on the implementation. """)
 
    
    species = st.selectbox("Select the suspected bacterial species", bact_species)
    country = st.selectbox('Select Country', countries)
    speciality = st.selectbox("Select the Speciality", Speciality)
    source = st.selectbox("Select the source of the sample", source_sample)
    year = st.selectbox("Select the year the sample was taken", [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    phenotype = st.selectbox("Select the Phenotype", Phenotype)
    antibiotic = st.selectbox("Select the Antibiotic", Antibiotic)

    #Prediction code
    susceptibility = ''

    #Creating a button for prediction

    if st.button('AI Model result'):
        example_data = [species, country, speciality, source, year, phenotype, antibiotic]
        encoded_values = pd.DataFrame([example_data], columns=['Species', 'Country', 'Speciality', 'Source', 'Year', 'Phenotype', 'Antibiotics'])
        encoded_values = pd.get_dummies(encoded_values)

        # Align the columns with the training data
        encoded_values = encoded_values.reindex(columns=xgb_model_no_genes.get_booster().feature_names)


        no_genes_prediction = xgb_model_no_genes.predict(encoded_values)


        if no_genes_prediction == 1:
            susceptibility = 'There is a high probability that the organism is susceptible to the antibiotic chosen. Confirm with the appropriate Sensitivity Test. '
        else:
            susceptibility = 'There is a high probability that the organism is resistant to the antibiotic chosen. If genotypic data is available, please go to the genotypic model. Confirm with the appropriate sensitivity test'
     


        st.write('Prediction: ', susceptibility)

        st.write("""Disclaimer: The predictive AI model provided is intended for informational purposes only. It is important to note that any predictions or insights generated by the model are based on the available data and assumptions made during the model's development. Predictive models, including AI models, have limitations and should not be considered as absolute or definitive. 

The accuracy and reliability of the model's predictions depend on various factors, including the quality and representativeness of the training data, the chosen algorithms and methodologies, and the assumptions made during model development. Predictive models are subject to inherent uncertainties, and there is always a possibility of errors or inaccuracies in the predictions.

It is crucial to exercise caution and not solely rely on the predictions generated by the AI model for critical decision-making. The predictions should be used as one of several sources of information, and it is advisable to consult domain experts or professionals to validate and interpret the results in the context of the specific use case.

Furthermore, it is essential to understand that the model's performance may vary when applied to different datasets or scenarios. Local conditions, temporal changes, or other contextual factors that were not explicitly considered during model training may affect the model's predictive capabilities.

The creators, developers, and providers of the AI model shall not be held responsible for any decisions, actions, or consequences resulting from the use of the model's predictions. Users of the model are responsible for independently verifying and validating the results and exercising their judgment and discretion when interpreting and applying the predictions.

Always exercise critical thinking, expert judgment, and professional expertise when using predictive AI models or any other decision-support tools.""")
        


# ANTIBACTERIAL WITH GENOTYPIC DATA PREDICTOR

#Load the data
Variables1 = pd.read_csv("Variables_genes.csv")


# Extract predictor names from the CSV file
countries1 = Variables1["Country"].tolist()
source_sample1 = Variables1["Source"].tolist()
bact_species1 = Variables1["Species"].tolist()
antibiotics1 = Variables1["Antibiotics"].tolist()
Phenotype1 = Variables1["Phenotype"].tolist()
Antibiotic1 = Variables1["Antibiotics"].tolist()
Speciality1 = Variables1["Speciality"].tolist()
ACC_1 = Variables1["ACC"].tolist()
ACTMIR_1 = Variables1["ACTMIR"].tolist()
CMY1MOX_1 = Variables1["CMY1MOX"].tolist()
CMY11_1 = Variables1["CMY11"].tolist()
CTXM1_1 = Variables1["CTXM1"].tolist()
CTXM2_1 = Variables1["CTXM2"].tolist()
CTMX9_1 = Variables1["CTMX9"].tolist()
CTXM825_1 = Variables1["CTXM825"].tolist()
DHA_1 = Variables1["DHA"].tolist()
FOX_1 = Variables1["FOX"].tolist()
GES_1 = Variables1["GES"].tolist()
GIM_1 = Variables1["GIM"].tolist()
IMP_1 = Variables1["IMP"].tolist()
KPC_1 = Variables1["KPC"].tolist()
NDM_1 = Variables1["NDM"].tolist()
OXA_1 = Variables1["OXA"].tolist()
PER_1 = Variables1["PER"].tolist()
SHV_1 = Variables1["SHV"].tolist()
TEM_1 = Variables1["TEM"].tolist()
VEB_1 = Variables1["VEB"].tolist()
VIM_1 = Variables1["VIM"].tolist()




if testing == "AI Antibacterial Predictor (Genotypic data)":
    st.title('AI Antibacterial Predictor (Genotypic data)')
    html_temp = """
    <div style="background-color:#020659;padding:10px">
    <h2 style="color:white;text-aligh:center;"> Please input your predictor variables below </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.sidebar.title("Instructions")
    with st.sidebar:
        st_lottie(lottie_prediction)
        st.info("""To input the predictor variables into an AI model for predicting susceptibility using the "AI Antibacterial Predictor," follow these steps:

Select the suspected bacterial species: Begin typing the name of the suspected bacterial species, and the dropdown list will prompt you with options in alphabetical order. For example, you can type "Achromobacter insolitus."

Select Country: Begin typing the name of the country from which the sample was taken, and the dropdown list will prompt you with options in alphabetical order. For instance, type "Argentina."

Select the Speciality: Begin typing the speciality associated with the sample, and the dropdown list will prompt you with options in alphabetical order. For example, type "Clinic / Office."

Select the source of the sample: Begin typing the source of the sample, and the dropdown list will prompt you with options in alphabetical order. For example, type "Abdominal Fluid."

Select the year the sample was taken: Choose the year when the sample was collected from the provided dropdown list. In this case, select "2023."

Select the Phenotype: Begin typing the phenotype associated with the sample, and the dropdown list will prompt you with options in alphabetical order. For example, type "ESBL."

Select the Antibiotic: Begin typing the name of the antibiotic for which you want to predict susceptibility, and the dropdown list will prompt you with options in alphabetical order. For instance, type "Amikacin."

Select the Gene types: Begin typing the name of the Gene type, and the dropdown list will prompt you with options in alphabetical order. For instance, type "Amikacin."
               

Once you have provided all the necessary information, the AI model will use these predictor variables to make a prediction of susceptibility. The specific details and functionality of the AI model may vary depending on the implementation. """)
 

    species_a = st.selectbox("Select the suspected bacterial species", bact_species1)
    country_a = st.selectbox('Select Country', countries1)
    speciality_a = st.selectbox("Select the Speciality", Speciality1)
    source_a = st.selectbox("Select the source of the sample", source_sample1)
    year_a = st.selectbox("Select the year the sample was taken", [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    ACC = st.selectbox("Which ACC genotype is present?", ACC_1)
    ACTMIR = st.selectbox("Which ACTMIR genotype is present?", ACTMIR_1)
    CMY1MOX = st.selectbox("Which CMY1MOX genotype is present?", CMY1MOX_1)
    CMY11 = st.selectbox("Which CMY11 genotype is present?", CMY11_1)
    CTXM1 = st.selectbox("Which CTXM1 genotype is present?", CTXM1_1)
    CTXM2 = st.selectbox("Which CTXM2 genotype is present?", CTXM2_1)
    CTMX9 = st.selectbox("Which CTMX9 genotype is present?", CTMX9_1)
    CTXM825 = st.selectbox("Which CTXM825 genotype is present?", CTXM825_1)
    DHA = st.selectbox("Which DHA genotype is present?", DHA_1)
    FOX = st.selectbox("Which FOX genotype is present?", FOX_1)
    GES = st.selectbox("Which GES genotype is present?", GES_1)
    GIM = st.selectbox("Which GIM genotype is present?", GIM_1)
    IMP = st.selectbox("Which IMP genotype is present?", IMP_1)
    KPC = st.selectbox("Which KPC genotype is present?", KPC_1)
    NDM = st.selectbox("Which NDM genotype is present?", NDM_1)
    OXA = st.selectbox("Which OXA genotype is present?", OXA_1)
    PER = st.selectbox("Which PER genotype is present?", PER_1)
    SHV = st.selectbox("Which SHV genotype is present?", SHV_1)
    TEM = st.selectbox("Which TEM genotype is present?", TEM_1)
    VEB = st.selectbox("Which VEB genotype is present?", VEB_1)
    VIM = st.selectbox("Which VIM genotype is present?", VIM_1)
    antibiotic_a = st.selectbox("Select the Antibiotic", Antibiotic1)

    #Prediction code
    susceptibility = ''

    #Creating a button for prediction

    if st.button('AI Model result'):
        example_data = [species_a, country_a, speciality_a, source_a, year_a, SHV, TEM, CTXM1, CTXM2, CTXM825, CTMX9, VEB, PER, GES, ACC, CMY1MOX, CMY11, DHA, FOX, ACTMIR, KPC, OXA, NDM, IMP, VIM, GIM, antibiotic_a]
        encoded_values = pd.DataFrame([example_data], columns=['Species', 'Country', 'Speciality', 'Source', 'Year', "SHV", "TEM", "CTXM1", "CTXM2", "CTXM825", "CTXM9", "VEB", "PER", "GES", "ACC", "CMY1MOX", "CMY11", "DHA", "FOX", "ACTMIR", "KPC", "OXA", "NDM", "IMP", "VIM", "GIM" ,'Antibiotics'])
        encoded_values = pd.get_dummies(encoded_values)

        # Align the columns with the training data
        encoded_values = encoded_values.reindex(columns=xgb_model_no_genes.get_booster().feature_names)


        no_genes_prediction = xgb_model_no_genes.predict(encoded_values)


        if no_genes_prediction == 1:
            susceptibility = 'There is a high probability that the organism is susceptible to the antibiotic chosen. Confirm with the appropriate Sensitivity Test. '
        else:
            susceptibility = 'There is a high probability that the organism is resistant to the antibiotic chosen. If genotypic data is available, please go to the genotypic model. Confirm with the appropriate sensitivity test'
     


        st.write('Prediction: ', susceptibility)

        st.write("""Disclaimer: The predictive AI model provided is intended for informational purposes only. It is important to note that any predictions or insights generated by the model are based on the available data and assumptions made during the model's development. Predictive models, including AI models, have limitations and should not be considered as absolute or definitive. 

The accuracy and reliability of the model's predictions depend on various factors, including the quality and representativeness of the training data, the chosen algorithms and methodologies, and the assumptions made during model development. Predictive models are subject to inherent uncertainties, and there is always a possibility of errors or inaccuracies in the predictions.

It is crucial to exercise caution and not solely rely on the predictions generated by the AI model for critical decision-making. The predictions should be used as one of several sources of information, and it is advisable to consult domain experts or professionals to validate and interpret the results in the context of the specific use case.

Furthermore, it is essential to understand that the model's performance may vary when applied to different datasets or scenarios. Local conditions, temporal changes, or other contextual factors that were not explicitly considered during model training may affect the model's predictive capabilities.

The creators, developers, and providers of the AI model shall not be held responsible for any decisions, actions, or consequences resulting from the use of the model's predictions. Users of the model are responsible for independently verifying and validating the results and exercising their judgment and discretion when interpreting and applying the predictions.

Always exercise critical thinking, expert judgment, and professional expertise when using predictive AI models or any other decision-support tools.""")
        

#Load the data
Variables2 = pd.read_csv("Variables2.csv")

# Extract predictor names from the CSV file for fungi
countries2 = Variables2["Country"].tolist()
source_sample2 = Variables2["Source"].tolist()
fungi_species2 = Variables2["Species"].tolist()
antifungal2 = Variables2["Antifungal"].tolist()
Year2 = Variables2["Year"].tolist()
Speciality2 = Variables2["Speciality"].tolist()




if testing == "AI Antifungal Predictor":
    st.title('AI Antifungal Predictor')
    html_temp = """
    <div style="background-color:#020659;padding:10px">
    <h2 style="color:white;text-aligh:center;"> Please input your predictor variables below  </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)


    st.sidebar.title("Instructions")
    with st.sidebar:
        st_lottie(lottie_prediction)
        st.info("""To input the predictor variables into an AI model for predicting susceptibility using the "AI Antifungal Predictor," follow these steps:

Select the suspected fungi species: Begin typing the name of the suspected fungi species, and the dropdown list will prompt you with options in alphabetical order. For example, you can type "Candida Albicans."

Select Country: Begin typing the name of the country from which the sample was taken, and the dropdown list will prompt you with options in alphabetical order. For instance, type "Argentina."

Select the Speciality: Begin typing the speciality associated with the sample, and the dropdown list will prompt you with options in alphabetical order. For example, type "Clinic / Office."

Select the source of the sample: Begin typing the source of the sample, and the dropdown list will prompt you with options in alphabetical order. For example, type "Abdominal Fluid."

Select the year the sample was taken: Choose the year when the sample was collected from the provided dropdown list. In this case, select "2023."

Select the Antifungal: Begin typing the name of the antifungal for which you want to predict susceptibility, and the dropdown list will prompt you with options in alphabetical order. For instance, type "Amikacin."

Once you have provided all the necessary information, the AI model will use these predictor variables to make a prediction of susceptibility. The specific details and functionality of the AI model may vary depending on the implementation. """)
 

    species2 = st.selectbox("Select the suspected fungi/yeast species", fungi_species2)
    country2 = st.selectbox('Select Country', countries2)
    speciality2 = st.selectbox("Select the Speciality", Speciality2)
    source2 = st.selectbox("Select the source of the sample", source_sample2)
    year2 = st.selectbox("Select the year the sample was taken", [ 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    antifungal2 = st.selectbox("Select the Antibiotic", antifungal2)

    #Prediction code
    susceptibility2 = ''

    #Creating a button for prediction

    if st.button('Antifungal AI Prediction'):
        example_data2 = [species2, country2, speciality2, source2, year2, antifungal2]
        encoded_values2 = pd.DataFrame([example_data2], columns=['Species', 'Country', 'Speciality', 'Source', 'Year', 'Antifungals'])
        encoded_values2 = pd.get_dummies(encoded_values2)

        # Align the columns with the training data
        encoded_values2 = encoded_values2.reindex(columns=xgb_model_fungi.get_booster().feature_names)


        fungi_prediction = xgb_model_fungi.predict(encoded_values2)


        if fungi_prediction == 1:
            susceptibility2 = 'There is a high probability that the organism is susceptible to the antifungal chosen. Confirm with the appropriate Sensitivity Test. '
        else:
            susceptibility2 = 'There is a high probability that the organism is resistant to the antifungal chosen. If genotypic data is available, please go to the genotypic model. Confirm with the appropriate sensitivity test'
     
        st.write('Prediction: ', susceptibility2)

        st.write("""Disclaimer: The predictive AI model provided is intended for informational purposes only. It is important to note that any predictions or insights generated by the model are based on the available data and assumptions made during the model's development. Predictive models, including AI models, have limitations and should not be considered as absolute or definitive. 

The accuracy and reliability of the model's predictions depend on various factors, including the quality and representativeness of the training data, the chosen algorithms and methodologies, and the assumptions made during model development. Predictive models are subject to inherent uncertainties, and there is always a possibility of errors or inaccuracies in the predictions.

It is crucial to exercise caution and not solely rely on the predictions generated by the AI model for critical decision-making. The predictions should be used as one of several sources of information, and it is advisable to consult domain experts or professionals to validate and interpret the results in the context of the specific use case.

Furthermore, it is essential to understand that the model's performance may vary when applied to different datasets or scenarios. Local conditions, temporal changes, or other contextual factors that were not explicitly considered during model training may affect the model's predictive capabilities.

The creators, developers, and providers of the AI model shall not be held responsible for any decisions, actions, or consequences resulting from the use of the model's predictions. Users of the model are responsible for independently verifying and validating the results and exercising their judgment and discretion when interpreting and applying the predictions.

Always exercise critical thinking, expert judgment, and professional expertise when using predictive AI models or any other decision-support tools.""")
        







if testing == "About AntiMicro.ai":
    st.title('About us')
    st.info("AntiMicro.ai was developed by Dr. Rachael Kanguha & Dr. Fredrick Mutisya to help health care professionals harness the power of Artificial Intelligence in combating Antimicrobial Resistance. This project was developed using Pfizer data as part of the Vivli Data challenge.")

