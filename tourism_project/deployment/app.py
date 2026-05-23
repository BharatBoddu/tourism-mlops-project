import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "BharatBoddu/tourism-purchase-model"
MODEL_FILE = "best_tourism_model.joblib"

@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILE)
    return joblib.load(path)

model = load_model()

st.set_page_config(page_title="Tourism Package Predictor", page_icon="✈️", layout="centered")
st.title("✈️ Wellness Tourism Package Purchase Predictor")
st.write("Predict whether a customer is likely to purchase the new Wellness Tourism Package.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=90, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry", "Self Inquiry"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    designation = st.selectbox("Designation", ["AVP", "VP", "Manager", "Senior Manager", "Executive"])
    passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    monthly_income = st.number_input("Monthly Income (₹)", min_value=1000, max_value=500000, value=30000, step=1000)
    num_persons = st.number_input("Number Of Persons Visiting", min_value=1, max_value=10, value=2)
    preferred_star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
    num_trips = st.number_input("Number Of Trips Per Year", min_value=0, max_value=20, value=2)
    num_children = st.number_input("Number Of Children Visiting", min_value=0, max_value=6, value=0)
    pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    num_followups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=2)
    duration_of_pitch = st.number_input("Duration Of Pitch (minutes)", min_value=5, max_value=300, value=30)

input_data = {
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_of_pitch,
}

if st.button("🔍 Predict Purchase", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"✅ **Likely to Purchase** the Wellness Tourism Package")
    else:
        st.warning(f"❌ **Unlikely to Purchase** the Wellness Tourism Package")
    st.metric("Purchase Probability", f"{probability:.1%}")
