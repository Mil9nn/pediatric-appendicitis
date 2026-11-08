import re
import time
import joblib
import docx
import pdfplumber
import pandas as pd
import streamlit as st

# use the full width of large screens
st.set_page_config(page_title="Appendicitis Risk Predictor", layout="wide")

# load both saved pipelines once and reuse them
@st.cache_resource
def load_models():
    clinical = joblib.load("models/appendicitis_rf_pipeline.joblib")
    with_us = joblib.load("models/appendicitis_rf_us_pipeline.joblib")
    return clinical, with_us

model_clinical, model_us = load_models()

# read the text out of an uploaded pdf, word or txt file
def read_report(file):
    file.seek(0)
    if file.name.lower().endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if file.name.lower().endswith(".docx"):
        return "\n".join(p.text for p in docx.Document(file).paragraphs)
    return file.read().decode("utf-8", errors="ignore")

# pull CBC values out of the report text, one stray letter before a number is ignored
def extract_cbc(text):
    patterns = {
        "Hemoglobin": r"^(?:Hemoglobin|Haemoglobin|Hb)\s*[A-Za-z]?\s*([0-9]+(?:\.[0-9]+)?)",
        "RBC_Count": r"^(?:Red Blood Cell Count|RBC(?: Count)?)\s*[A-Za-z]?\s*([0-9]+(?:\.[0-9]+)?)",
        "WBC_Count": r"^(?:Total Leukocyte Count|Total WBC Count|WBC(?: Count)?|TLC)\s*[A-Za-z]?\s*([0-9]+(?:\.[0-9]+)?)",
        "Thrombocyte_Count": r"^(?:Platelet Count|Platelets|PLT)\s*[A-Za-z]?\s*([0-9]+(?:\.[0-9]+)?)",
        "Neutrophil_Percentage": r"^Neutrophils?\s*%?\s*[A-Za-z]?\s*([0-9]+(?:\.[0-9]+)?)",
        "RDW": r"^(?:RDW(?:-CV)?|Red Cell Distribution Width)\s*[A-Za-z]?\s*([0-9]+(?:\.[0-9]+)?)",
        "CRP": r"^(?:C-Reactive Protein|CRP)[^0-9\n]*([0-9]+(?:\.[0-9]+)?)",
    }
    found = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            found[name] = float(match.group(1))
    # reports often give WBC as cells/cu mm (8000), our model uses thousands (8.0)
    if found.get("WBC_Count", 0) > 100:
        found["WBC_Count"] /= 1000
    return found

# turn the wording used on urine reports into the grades our model uses
def to_grade(word):
    w = word.lower().strip()
    if w in ("negative", "nil", "absent", "none", "not detected"):
        return "no"
    if w in ("trace", "+", "1+", "positive"):
        return "+"
    if w in ("++", "2+"):
        return "++"
    if w in ("+++", "3+", "++++", "4+"):
        return "+++"
    return None

# pull the three urine dipstick results out of the report text
def extract_urine(text):
    patterns = {
        "Ketones_in_Urine": r"^Ketones?\s+(\S+)",
        "RBC_in_Urine": r"^Blood\s+(\S+)",
        "WBC_in_Urine": r"^Leukocyte Esterase\s+(\S+)",
    }
    found = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            grade = to_grade(match.group(1))
            if grade:
                found[name] = grade
    return found

# pull age and sex from the patient header, like "30 years / MALE"
def extract_patient(text):
    found = {}
    m = re.search(r"(\d+(?:\.\d+)?)\s*(years?|yrs?|months?|days?)\s*/\s*(male|female)", text, flags=re.IGNORECASE)
    if m:
        age = float(m.group(1))
        unit = m.group(2).lower()
        # convert months or days to years, because the model uses years
        if unit.startswith("month"):
            age /= 12
        elif unit.startswith("day"):
            age /= 365
        found["Age"] = round(age, 1)
        found["Sex"] = m.group(3).lower()
    return found

# pull height, weight and temperature when the report lists them, like "Height: 150 cm"
def extract_vitals(text):
    patterns = {
        "Height": r"Height\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*cm",
        "Weight": r"Weight\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*kg",
        "Body_Temperature": r"(?:Body\s*)?Temp(?:erature)?\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*°?\s*([CF])?",
    }
    found = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if name == "Body_Temperature":
                unit = (match.group(2) or "").upper()
                # convert Fahrenheit to Celsius, a reading above 45 can only be Fahrenheit
                if unit == "F" or val > 45:
                    val = (val - 32) * 5 / 9
            found[name] = round(val, 1)
    return found

# say "yes" if the finding is mentioned, "no" if a negative word comes just before or after it
def finding(text, words):
    for m in re.finditer(words, text, flags=re.IGNORECASE):
        # only look at the same sentence or line
        before = re.split(r"[\n.;]", text[max(0, m.start() - 40):m.start()])[-1].lower()
        after = re.split(r"[\n.;]", text[m.end():m.end() + 25])[0].lower()
        if re.search(r"\b(no|not|without|absent|negative|nil|none)\b", before) or \
           re.search(r"\b(absent|not seen|not visible|nil|negative|none)\b", after):
            return "no"
        return "yes"
    return None

# pull the ultrasound findings our model uses out of the report text
def extract_ultrasound(text):
    found = {"US_Performed": "yes"}

    # appendix diameter, like "appendix measures 8.5 mm" or "diameter: 0.8 cm"
    m = re.search(r"appendi(?:x|ceal)[^.\n]{0,60}?([0-9]+(?:\.[0-9]+)?)\s*(mm|cm)", text, flags=re.IGNORECASE) \
        or re.search(r"diameter[^.\n]{0,30}?([0-9]+(?:\.[0-9]+)?)\s*(mm|cm)", text, flags=re.IGNORECASE)
    if m:
        d = float(m.group(1))
        # the model uses millimetres
        if m.group(2).lower() == "cm":
            d *= 10
        found["Appendix_Diameter"] = float(min(max(round(d, 1), 0.0), 30.0))

    # was the appendix seen on the scan
    if re.search(r"appendix[^.\n]*\b(?:not|non)[ -]?(?:visuali[sz]ed|seen|identified|visible)", text, flags=re.IGNORECASE) or \
       re.search(r"(?:unable to|could not|cannot) (?:visuali[sz]e|identify|see)[^.\n]*appendix", text, flags=re.IGNORECASE):
        found["Appendix_on_US"] = "no"
    elif re.search(r"appendix[^.\n]*(?:visuali[sz]ed|seen|identified|visible)", text, flags=re.IGNORECASE) or "Appendix_Diameter" in found:
        found["Appendix_on_US"] = "yes"

    # free fluid and surrounding tissue reaction
    fluid = finding(text, r"free\s+(?:\w+\s+){0,2}fluids?")
    if fluid:
        found["Free_Fluids"] = fluid
    tissue = finding(text, r"fat stranding|periappendiceal (?:fat|inflammat\w*|reaction)|surrounding (?:tissue|fat) (?:reaction|inflammat\w*)|inflammatory (?:changes|reaction)")
    if tissue:
        found["Surrounding_Tissue_Reaction"] = tissue
    return found

# page title and a short note about what the app is
st.title("Pediatric Appendicitis Risk Predictor")
st.write("Uses clinical, lab and ultrasound findings.")

# switch: on = clinical, lab and ultrasound (main model), off = clinical and lab only
use_ultrasound = st.checkbox("Use ultrasound findings (recommended, more accurate)", value=True)

# drag and drop zones for the reports, shown in three columns
st.subheader("Upload reports (optional)")
up1, up2, up3, up4 = st.columns(4)
cbc_file = up1.file_uploader("CBC report", type=["pdf", "docx", "txt"])
urine_file = up2.file_uploader("Urine routine report", type=["pdf", "docx", "txt"])
us_file = up3.file_uploader("Ultrasound report", type=["pdf", "docx", "txt"])
notes_file = up4.file_uploader("Doctor's findings", type=["pdf", "docx", "txt"])

# empty box under the upload zones where the "filled from report" notes will appear
notes_box = st.container()

# # show the extracted text, so we can see what each report looks like
# for label, file in [("CBC", cbc_file), ("Urine routine", urine_file), ("Doctor's findings", notes_file)]:
#     if file is not None:
#         with st.expander(f"Extracted text: {label}"):
#             st.text(read_report(file))

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

# give every box a starting value, so a report can overwrite it later
for name, (low, high, default, step) in num_fields.items():
    st.session_state.setdefault(name, default)

# when a new CBC report is dropped, extract the values and put them into the boxes
if cbc_file is None:
    # forget the last report, so dropping the same file again fills the boxes again
    st.session_state.pop("cbc_loaded", None)
elif st.session_state.get("cbc_loaded") != (cbc_file.name, cbc_file.size):
    found = extract_cbc(read_report(cbc_file))
    filled = {}
    for name, val in found.items():
        low, high, _, _ = num_fields[name]
        filled[name] = float(min(max(val, low), high))
        st.session_state[name] = filled[name]
    st.session_state["cbc_loaded"] = (cbc_file.name, cbc_file.size)
    st.session_state["cbc_found"] = filled

if cbc_file is not None and "cbc_found" in st.session_state:
    with notes_box:
        if st.session_state["cbc_found"]:
            summary = ", ".join(f"{k} = {v:g}" for k, v in st.session_state["cbc_found"].items())
            st.info("Filled from CBC report: " + summary + ". Please check the values.")
        else:
            st.warning("Could not read any values from the CBC report.")
# starting value for Sex, so a report can overwrite it
st.session_state.setdefault("Sex", "female")

# read age, sex, height, weight and temperature from whichever reports are uploaded
uploaded = [f for f in (cbc_file, urine_file, notes_file) if f is not None]
signature = tuple((f.name, f.size) for f in uploaded)
if not uploaded:
    st.session_state.pop("patient_loaded", None)
elif st.session_state.get("patient_loaded") != signature:
    all_text = "\n".join(read_report(f) for f in uploaded)
    found_patient = {**extract_patient(all_text), **extract_vitals(all_text)}
    filled = {}
    for name, val in found_patient.items():
        if name == "Sex":
            filled[name] = val
        else:
            low, high, _, _ = num_fields[name]
            filled[name] = float(min(max(val, low), high))
        st.session_state[name] = filled[name]
    st.session_state["patient_loaded"] = signature
    st.session_state["patient_found"] = filled

# show what was filled, with the values
if uploaded and st.session_state.get("patient_found"):
    with notes_box:
        summary = ", ".join(f"{k} = {v}" for k, v in st.session_state["patient_found"].items())
        st.info("Filled from patient details: " + summary + ". Please check the values.")

# BMI = weight in kg divided by height in metres squared, recalculated on every run
bmi = st.session_state["Weight"] / (st.session_state["Height"] / 100) ** 2
st.session_state["BMI"] = float(min(max(round(bmi, 1), 5.0), 50.0))

# four inputs per row, each box uses its own step size and its own key
values = {}
cols = st.columns(4)
for i, (name, (low, high, default, step)) in enumerate(num_fields.items()):
    values[name] = cols[i % 4].number_input(
        name, min_value=low, max_value=high, step=step, format="%.1f", key=name,
        disabled=(name == "BMI"),
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

# starting values for the urine boxes, so a report can overwrite them
for name in ["Ketones_in_Urine", "RBC_in_Urine", "WBC_in_Urine"]:
    st.session_state.setdefault(name, "no")

# when a new urine report is dropped, extract the grades and put them into the boxes
if urine_file is None:
    st.session_state.pop("urine_loaded", None)
elif st.session_state.get("urine_loaded") != (urine_file.name, urine_file.size):
    found_urine = extract_urine(read_report(urine_file))
    for name, grade in found_urine.items():
        st.session_state[name] = grade
    st.session_state["urine_loaded"] = (urine_file.name, urine_file.size)
    st.session_state["urine_found"] = found_urine

if urine_file is not None and "urine_found" in st.session_state:
    with notes_box:
        if st.session_state["urine_found"]:
            summary = ", ".join(f"{k} = {v}" for k, v in st.session_state["urine_found"].items())
            st.info("Filled from urine report: " + summary + ". Please check the values.")
        else:
            st.warning("Could not read any values from the urine report.")

                # microscopy counts are not model inputs, so only show them for the user to review
        hpf = re.findall(r"^(WBC|RBC) \(HPF\)\s*([0-9.]+)", read_report(urine_file), flags=re.MULTILINE)
        if hpf:
            st.caption("Not used by the model: " + ", ".join(f"{n} (HPF) = {v}" for n, v in hpf))

# urine tests and peritonitis, ordered from none to most severe, all four in one row
st.subheader("Urine tests and peritonitis")
cols = st.columns(4)
values["Ketones_in_Urine"] = cols[0].selectbox("Ketones_in_Urine", ["no", "+", "++", "+++"], key="Ketones_in_Urine")
values["RBC_in_Urine"] = cols[1].selectbox("RBC_in_Urine", ["no", "+", "++", "+++"], key="RBC_in_Urine")
values["WBC_in_Urine"] = cols[2].selectbox("WBC_in_Urine", ["no", "+", "++", "+++"], key="WBC_in_Urine")
values["Peritonitis"] = cols[3].selectbox("Peritonitis", ["no", "local", "generalized"])

# categories with no natural order
st.subheader("Other")
cols = st.columns(4)
values["Sex"] = cols[0].selectbox("Sex", ["female", "male"], key="Sex")
values["Stool"] = cols[1].selectbox("Stool", ["normal", "diarrhea", "constipation", "constipation, diarrhea"])

# when a new ultrasound report is dropped, extract the findings, they become the starting values of the boxes below
if us_file is None:
    st.session_state.pop("us_loaded", None)
    st.session_state.pop("us_found", None)
elif st.session_state.get("us_loaded") != (us_file.name, us_file.size):
    st.session_state["us_found"] = extract_ultrasound(read_report(us_file))
    st.session_state["us_loaded"] = (us_file.name, us_file.size)

# show what was filled, with the values
if us_file is not None:
    with notes_box:
        found_us = st.session_state.get("us_found", {})
        summary = ", ".join(f"{k} = {v}" for k, v in found_us.items())
        st.info("Filled from ultrasound report: " + summary + ". Please check the values.")

# ultrasound inputs only appear when the switch is on
if use_ultrasound:
    st.subheader("Ultrasound findings")
    # starting values come from the uploaded ultrasound report, if there is one
    d = st.session_state.get("us_found", {})
    cols = st.columns(4)
    values["Appendix_Diameter"] = cols[0].number_input(
        "Appendix_Diameter", min_value=0.0, max_value=30.0,
        value=float(d.get("Appendix_Diameter", 7.0)), step=0.5, format="%.1f"
    )
    values["Surrounding_Tissue_Reaction"] = cols[1].selectbox(
        "Surrounding_Tissue_Reaction", ["no", "yes"],
        index=["no", "yes"].index(d.get("Surrounding_Tissue_Reaction", "no")))
    values["Free_Fluids"] = cols[2].selectbox(
        "Free_Fluids", ["no", "yes"], index=["no", "yes"].index(d.get("Free_Fluids", "no")))
    values["Appendix_on_US"] = cols[3].selectbox(
        "Appendix_on_US", ["yes", "no"], index=["yes", "no"].index(d.get("Appendix_on_US", "yes")))
    values["US_Performed"] = cols[0].selectbox(
        "US_Performed", ["yes", "no"], index=["yes", "no"].index(d.get("US_Performed", "yes")))

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