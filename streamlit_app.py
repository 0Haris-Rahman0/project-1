import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Encoding maps
fat_map = {'Low Fat': 0, 'Regular': 1}
size_map = {'Small': 1, 'Medium': 2, 'High': 0}
location_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
type_map = {'Supermarket Type1': 3, 'Supermarket Type2': 2, 'Supermarket Type3': 0, 'Grocery Store': 1}

# Page config
st.set_page_config(page_title="BigMart Sales Predictor", layout="centered")
st.title("🛒 BigMart Sales Predictor")

# Tabs: User input vs Hardcoded
tab1, tab2 = st.tabs(["🔧 Predict via Input", "🧪 Test Hardcoded Example"])

# -------- Tab 1: User Input --------
with tab1:
    st.subheader("Enter Product and Outlet Details")

    item_weight = st.number_input("Item Weight", min_value=0.0, max_value=50.0, value=12.5)
    item_fat_content = st.selectbox("Item Fat Content", ['Low Fat', 'Regular'])
    item_visibility = st.slider("Item Visibility", min_value=0.0, max_value=1.0, value=0.05)
    item_type = st.number_input("Item Type (encoded)", min_value=0, max_value=15, value=4)
    item_mrp = st.number_input("Item MRP", min_value=0.0, max_value=400.0, value=100.0)
    establishment_year = st.number_input("Outlet Establishment Year", min_value=1985, max_value=2022, value=1999)
    outlet_size = st.selectbox("Outlet Size", ['Small', 'Medium', 'High'])
    outlet_location_type = st.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
    outlet_type = st.selectbox("Outlet Type", ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

    # Encode input
    user_features = np.array([
        item_weight,
        fat_map[item_fat_content],
        item_visibility,
        item_type,
        item_mrp,
        establishment_year,
        size_map[outlet_size],
        location_map[outlet_location_type],
        type_map[outlet_type]
    ]).reshape(1, -1)

    if st.button("Predict Sales"):
        user_prediction = model.predict(user_features)
        st.success(f"Predicted Sales: ₹{user_prediction[0]:,.2f}")

# -------- Tab 2: Hardcoded Example --------
with tab2:
    st.subheader("Hardcoded Example Input")

    # Hardcoded data
    example_features = np.array([
        13.5,                    # Item Weight
        fat_map['Low Fat'],      # Fat Content
        0.065,                   # Visibility
        4,                       # Item Type (encoded)
        245.25,                  # MRP
        1999,                    # Establishment Year
        size_map['Medium'],      # Size
        location_map['Tier 2'],  # Location
        type_map['Supermarket Type1']  # Outlet Type
    ]).reshape(1, -1)

    # Display the input
    st.code(f"""
Hardcoded Input:
Item Weight: 13.5
Item Fat Content: Low Fat
Item Visibility: 0.065
Item Type (encoded): 4
Item MRP: 245.25
Establishment Year: 1999
Outlet Size: Medium
Outlet Location Type: Tier 2
Outlet Type: Supermarket Type1
    """)

    if st.button("Predict Hardcoded Example"):
        example_prediction = model.predict(example_features)
        st.success(f"Predicted Sales: ₹{example_prediction[0]:,.2f}")