# Pediatric Appendicitis Prediction Using Machine Learning

This project aims to predict whether a pediatric patient is likely to have appendicitis using clinical, laboratory, and ultrasound findings. The system uses a machine learning model trained on patient data to support early risk assessment and clinical decision-making.

## Problem Statement

Appendicitis can be difficult to identify in children because its symptoms may resemble other conditions. This project uses the Regensburg Pediatric Appendicitis dataset to investigate whether patient details, symptoms, laboratory results, and selected ultrasound findings can help classify pediatric patients as having appendicitis or not.

This project examines whether machine-learning models can distinguish pediatric appendicitis cases from non-appendicitis cases, and whether adding ultrasound findings improves predictions.

**Context and target**
 The model predicts the patient’s diagnosis, classifying it as either appendicitis or no appendicitis.

**Task**
 This project uses supervised binary classification to predict whether a patient has appendicitis. It compares predictions based on clinical and laboratory data with predictions that also include selected ultrasound findings.

**Objectives and success criteria**
Evaluate and compare the models using accuracy, precision, recall, F1-score, and ROC-AUC. 
- The project aims to achieve a test ROC-AUC of at least 0.80 and an appendicitis recall of at least 0.85. 
- These targets are for project evaluation only, not for clinical use.

**Intended users and impact:** This project is intended for students and researchers studying machine learning in pediatric healthcare. It explores how patient information can estimate appendicitis risk. The model is for educational purposes only and must not be used to diagnose patients or guide treatment.

**Assumptions and Constraints**
- The study is based on a limited dataset obtained from a single source. Missing values may affect the analysis and require appropriate preprocessing. 
- The model's performance may not generalise to other patient populations. Therefore, validation using independent external datasets is necessary before considering any real-world application.

## Objective

The main objective of this project is to:
- Develop a reliable machine learning model for appendicitis prediction
- Compare models using clinical laboratory features with models that also use selected ultrasound features.
- Evaluate model performance using a held-out test set and cross-validation.
- Present predictions through a Streamlit prototype.

## Dataset

The project uses the Regensburg Pediatric Appendicitis dataset from the UCI Machine Learning Repository.

Source:
- UCI Machine Learning Repository
- Dataset ID: 938
- Name: Regensburg Pediatric Appendicitis dataset

Link: https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis

The dataset contains pediatric patient records with demographic, clinical, laboratory, and ultrasound-related variables.

**Size and target**

The dataset contains 782 patient records and 58 columns: 57 potential input variables and the target, **Diagnosis**.
Two records have no recorded diagnosis, leaving 780 labeled records for classification.

The target classes are:

| Diagnosis | Records | Percentage of Labeled Records |
|---|---:|---:|
| Appendicitis | 463 | 59.4% |
| No appendicitis | 317 | 40.6% |

**Feature types and data dictionary**

The dataset includes 18 numerical input variables and 39 categorical input variables. Categorical fields include yes/no findings, labels, and clinical categories.

| Feature Group | Variables and Descriptions |
|---|---|
| Demographics and Measurements | `Age` (patient age), `Sex`, `BMI`, `Height`, `Weight` |
| Clinical Scores | `Alvarado_Score` and `Paedriatic_Appendicitis_Score` (appendicitis assessment scores) |
| Symptoms and Examination | `Migratory_Pain`, `Lower_Right_Abd_Pain`, `Contralateral_Rebound_Tenderness`, `Coughing_Pain`, `Nausea`, `Loss_of_Appetite`, `Dysuria`, `Stool`, `Peritonitis`, `Psoas_Sign`, `Ipsilateral_Rebound_Tenderness` |
| Laboratory Results | `Body_Temperature`, `WBC_Count`, `Neutrophil_Percentage`, `Segmented_Neutrophils`, `Neutrophilia`, `RBC_Count`, `Hemoglobin`, `RDW`, `Thrombocyte_Count`, `Ketones_in_Urine`, `RBC_in_Urine`, `WBC_in_Urine`, `CRP` |
| Ultrasound | `US_Performed`, `US_Number`, `Appendix_on_US`, `Appendix_Diameter`, `Free_Fluids`, `Appendix_Wall_Layers`, `Target_Sign`, `Appendicolith`, `Perfusion`, `Perforation`, `Surrounding_Tissue_Reaction`, `Appendicular_Abscess`, `Abscess_Location`, `Pathological_Lymph_Nodes`, `Lymph_Nodes_Location`, `Bowel_Wall_Thickening`, `Conglomerate_of_Bowel_Loops`, `Ileus`, `Coprostasis`, `Meteorism`, `Enteritis`, `Gynecological_Findings` |
| Target | `Diagnosis`: final recorded diagnosis, with classes `appendicitis` and `no appendicitis` |
| Additional Recorded Fields | `Length_of_Stay`, `Management`, `Severity`, `Diagnosis_Presumptive`. These are present in the source data but excluded from the notebook's model inputs to reduce outcome leakage. |

**Missing values**

Blank values appear in all 58 columns. Missingness is especially high in some fields: **Abscess_Location** is missing for 769 records (98.3%), **Gynecological_Findings** for 756 (96.7%), and **Segmented_Neutrophils** for 728 (93.1%). Missing ultrasound findings are an important limitation. More than 70% of the 14 **ultrasound features** are missing.

**Why this dataset is suitable**

The dataset contains pediatric patient symptoms, examination findings, laboratory results, and ultrasound information relevant to the appendicitis classification problem. 

Its recorded diagnosis provides the target label, and its range of feature types supports comparing models built with clinical and laboratory data against models that also use ultrasound findings.

## Methodology

- The notebook cleans and explores the dataset, removes records without a diagnosis, and excludes features that could leak diagnosis or treatment information. 
- Numerical and categorical features are processed within model pipelines, where missing values are imputed and categorical values are encoded.

- A stratified 80/20 split was used, with 624 records for training and 156 for testing. Five-fold cross-validation was used during model development.
- The project compares logistic regression and a tuned Random Forest, using clinical and laboratory features, then repeats the comparison with selected ultrasound features.

## Model Evaluation

The model is evaluated using relevant classification metrics such as:
- accuracy
- precision
- recall
- F1-score
- ROC-AUC

The project also compares prediction performance across different feature sets, including:
- clinical and lab-only model
- clinical, lab, and ultrasound model

## Application

The project includes a Streamlit web application that allows a user to:
- enter patient details manually
- upload report files
- auto-fill values from uploaded medical reports
- predict appendicitis risk
- view the probability of appendicitis

## Project Structure

- app.py: Streamlit application
- models/: saved trained machine learning models
- data/: dataset files
- results/: generated plots and visual outputs
- README.md: project overview
- requirements.txt: Python dependencies

## Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the app:
    ```bash
   streamlit run app.py

## Results

![Test ROC-AUC comparison for models with clinical and laboratory features, with and without selected ultrasound features.](results/figures/fig10_ultrasound_effect.png)

- On the held-out test set, the tuned Random Forest using clinical and laboratory features achieved 0.712 accuracy, 0.819 ROC-AUC, and 0.860 recall for appendicitis. 
- With the selected ultrasound features added, it achieved 0.872 accuracy, 0.951 ROC-AUC, and 0.946 appendicitis recall.

These are results from one test split of the dataset. The increase with ultrasound features is promising for this split, but requires validation on independent data before drawing conclusions about performance in other clinical settings.

Note: Insert final metrics and evaluation figures here after the model is finalized.

## Limitations
- The dataset comes from a single source and has a limited number of records.
- The model has not yet been tested on data from other hospitals, so its performance may differ across different patient groups and clinical settings.
- Missing values, particularly in ultrasound findings, with some ultrasound features having more than 70% missing values. Although missing values are handled during preprocessing, the actual clinical information cannot always be recovered.
- The project focuses on model performance, but further testing is needed to check prediction reliability and suitable thresholds.