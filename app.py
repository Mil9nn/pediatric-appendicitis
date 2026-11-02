import joblib
import pandas as pd
import streamlit as st
import time

# use the full width of large screens
st.set_page_config(page_title="Appendicitis Risk Predictor", layout="wide")

# load both saved pipelines once and reuse them
@st.cache_resource
def load_models():
    clinical = joblib.load("models/appendicitis_rf_pipeline.joblib")
    with_us = joblib.load("models/appendicitis_rf_us_pipeline.joblib")
    return clinical, with_us

model_clinical, model_us = load_models()

# page title and a short note about what the app is
st.title("Pediatric Appendicitis Risk Predictor")
st.write("Main model uses routine clinical and lab values only. Student project, not a medical tool.")

# switch: off = clinical and lab only (the main idea), on = also use ultrasound findings
use_ultrasound = st.checkbox("Also use ultrasound findings (optional)")

# numeric inputs as (min, max, default, step), defaults are close to the training medians
st.subheader("Measurements")
num_fields = {
    "Age": (0.0, 18.0, 11.0, 1.0),
    "BMI": (5.0, 50.0, 18.0, 0.5),
    "Height": (40.0, 200.0, 150.0, 1.0),
    "Weight": (2.0, 120.0, 42.0, 1.0),
    "Body_Temperature": (34.0, 42.0, 37.2, 0.1),
    "WBC_Count": (0.0, 50.0, 11.6, 0.5),
    "Neutrophil_Percentage": (0.0, 100.0, 75.0, 1.0),
    "RBC_Count": (2.0, 8.0, 4.8, 0.1),
    "Hemoglobin": (5.0, 20.0, 13.3, 0.1),
    "RDW": (10.0, 25.0, 12.8, 0.1),
    "Thrombocyte_Count": (50.0, 800.0, 275.0, 5.0),
    "CRP": (0.0, 400.0, 7.0, 1.0),
}

# show the inputs in two columns, each box uses its own step size
values = {}
cols = st.columns(4)
for i, (name, (low, high, default, step)) in enumerate(num_fields.items()):
    values[name] = cols[i % 4].number_input(
        name, min_value=low, max_value=high, value=default, step=step, format="%.1f"
    )

# yes/no symptoms and findings
st.subheader("Symptoms and findings")
yes_no_fields = [
    "Migratory_Pain", "Lower_Right_Abd_Pain", "Contralateral_Rebound_Tenderness",
    "Coughing_Pain", "Nausea", "Loss_of_Appetite", "Neutrophilia", "Dysuria",
    "Psoas_Sign", "Ipsilateral_Rebound_Tenderness",
]
cols = st.columns(4)
for i, name in enumerate(yes_no_fields):
    values[name] = cols[i % 4].selectbox(name, ["no", "yes"])

# urine tests and peritonitis, ordered from none to most severe, all four in one row
st.subheader("Urine tests and peritonitis")
cols = st.columns(4)
values["Ketones_in_Urine"] = cols[0].selectbox("Ketones_in_Urine", ["no", "+", "++", "+++"])
values["RBC_in_Urine"] = cols[1].selectbox("RBC_in_Urine", ["no", "+", "++", "+++"])
values["WBC_in_Urine"] = cols[2].selectbox("WBC_in_Urine", ["no", "+", "++", "+++"])
values["Peritonitis"] = cols[3].selectbox("Peritonitis", ["no", "local", "generalized"])

# categories with no natural order, three per row like the other sections
st.subheader("Other")
cols = st.columns(4)
values["Sex"] = cols[0].selectbox("Sex", ["female", "male"])
values["Stool"] = cols[1].selectbox("Stool", ["normal", "diarrhea", "constipation", "constipation, diarrhea"])

# ultrasound inputs only appear when the switch is on
if use_ultrasound:
    st.subheader("Ultrasound findings")
    col9, col10 = st.columns(2)
    values["Appendix_Diameter"] = col9.number_input(
        "Appendix_Diameter", min_value=0.0, max_value=30.0, value=7.0, step=0.5, format="%.1f"
    )
    values["Surrounding_Tissue_Reaction"] = col10.selectbox("Surrounding_Tissue_Reaction", ["no", "yes"])
    values["Free_Fluids"] = col9.selectbox("Free_Fluids", ["no", "yes"])
    values["Appendix_on_US"] = col10.selectbox("Appendix_on_US", ["yes", "no"])
    values["US_Performed"] = col9.selectbox("US_Performed", ["yes", "no"])

st.subheader("Prediction")

if st.button("Predict Appendicitis Risk"):
    # pick the model that matches the switch
    model = model_us if use_ultrasound else model_clinical

        # show a spinner while the model runs
    with st.spinner("Analysing patient data..."):
        # short pause so the spinner is visible, since the prediction itself is very fast
        time.sleep(0.8)

        # build one row and put the columns in the order the model was trained with
        input_df = pd.DataFrame([values])[model.feature_names_in_]

        # predict the class and the probability of appendicitis
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    # show the result
    if prediction == 1:
        st.error("Prediction: Appendicitis")
    else:
        st.success("Prediction: No Appendicitis")
    st.write(f"Probability of appendicitis: {probability * 100:.1f}%")
    st.caption("Model used: " + ("clinical + lab + ultrasound" if use_ultrasound else "clinical + lab only"))