# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
# Set the title, icon, and layout for the Streamlit page
st.set_page_config(page_title="NASA Asteroid Profiler", page_icon="☄️", layout="wide")

# --- LOAD MODELS & DATA ---
# Use a cache to load assets only once, improving performance
@st.cache_resource
def load_assets():
    # Load the pre-trained K-Means clustering model
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    # Load the scaler used for standardizing data
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    # Load the regression model for predicting diameter
    with open('diameter_regressor.pkl', 'rb') as f:
        regressor = pickle.load(f)
    # Load the classification model for hazard prediction
    with open('hazard_model.pkl', 'rb') as f:
        classifier = pickle.load(f)
        
    # Load the clustered asteroid data for background visualization
    df = pd.read_csv('nasa_asteroid_clustered.csv')
    # Take a random sample of 1000 rows to keep the app responsive
    df_sample = df.sample(n=1000, random_state=42)
    
    return kmeans, scaler, regressor, classifier, df_sample

# Load all assets into memory
kmeans, scaler, regressor, classifier, df_background = load_assets()

# --- SIDEBAR (INPUTS) ---
# Create a header for the sidebar
st.sidebar.header("📡 Input Orbital Parameters")
st.sidebar.markdown("Define the physics for the **Candidate Object**:")

# Create input fields in the sidebar for asteroid parameters
# Default values are set to simulate a potentially dangerous object
a_input = st.sidebar.number_input("Semi-Major Axis (AU)", 0.5, 5.0, 1.53, 0.01)
e_input = st.sidebar.slider("Eccentricity (0=Circle, 1=Oval)", 0.0, 1.0, 0.79, 0.01)
i_input = st.sidebar.slider("Inclination (Degrees)", 0.0, 90.0, 27.5, 0.1)
h_input = st.sidebar.number_input("Absolute Magnitude (H)", 10.0, 30.0, 16.7, 0.1)
moid_input = st.sidebar.number_input("Earth MOID (AU)", 0.0, 1.0, 0.01, 0.001)

# --- MAIN APP UI ---
# Set the main title and introductory text for the app
st.title("☄️ NASA Asteroid Hazard Profiler")
st.markdown("### 🛡️ Autonomous Planetary Defense System")
st.markdown("This system uses a **3-Stage AI Pipeline** to Profile, Measure, and Assess unknown Near-Earth Objects.")
st.markdown("---")

# --- ANALYSIS TRIGGER ---
# This block runs only when the user clicks the button in the sidebar
if st.sidebar.button("🚀 Run Hazard Analysis", type="primary"):
    
    # 1. PREPARE INPUT DATA
    # Create a DataFrame from the user's input
    input_df = pd.DataFrame({
        'semi_major_axis': [a_input], 'eccentricity': [e_input], 'inclination': [i_input],
        'magnitude_h': [h_input], 'moid_au': [moid_input]
    })
    
    # 2. STAGE 1: CLUSTERING (The Sorter)
    # Scale the input data using the pre-fitted scaler
    physics_scaled = scaler.transform(input_df[['semi_major_axis', 'eccentricity', 'inclination']])
    # Predict the cluster ID for the input asteroid
    cluster_id = kmeans.predict(physics_scaled)[0]
    
    # Map the cluster ID to a human-readable family name
    family_map = {0: "Amor (Outer Belt)", 1: "Apollo (Earth Crosser)", 2: "Aten (Inner Orbit)"}
    family_name = family_map.get(cluster_id, "Unknown Family")
    
    # 3. STAGE 2: REGRESSION (The Sizer)
    # Select features for the regression model
    reg_features = input_df[['magnitude_h', 'semi_major_axis', 'eccentricity', 'inclination']]
    # Predict the diameter of the asteroid
    predicted_diameter = regressor.predict(reg_features)[0]
    
    # 4. STAGE 3: CLASSIFICATION (The Guard)
    # Prepare features for the classification model, including the predicted diameter
    class_features = pd.DataFrame({
        'moid_au': [moid_input], 'diameter': [predicted_diameter],
        'eccentricity': [e_input], 'semi_major_axis': [a_input],
        'inclination': [i_input], 'cluster_label': [cluster_id]
    })
    # Predict whether the asteroid is hazardous
    is_hazardous = classifier.predict(class_features)[0]
    # Predict the probability of it being hazardous
    hazard_prob = classifier.predict_proba(class_features)[0][1]

    # --- DISPLAY METRICS ---
    # Create three columns for displaying key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🪐 Orbital Family")
        st.metric(label="Classification", value=family_name)
    
    with col2:
        st.warning("📏 Est. Diameter")
        st.metric(label="Predicted Size", value=f"{predicted_diameter:.3f} km")
        
    with col3:
        # Display threat level with color-coding based on the prediction
        if is_hazardous == 1:
            st.error("🚨 THREAT LEVEL")
            st.metric(label="Status", value="HAZARDOUS")
        else:
            st.success("✅ THREAT LEVEL")
            st.metric(label="Status", value="SAFE")

    # Display the AI's confidence in its prediction
    st.markdown(f"**AI Confidence:** The system is **{hazard_prob:.1%}** certain that this object poses a collision risk.")
    st.markdown("---")

    # --- VISUALIZATION SECTION ---
    st.subheader("📊 Visual Forensics")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["3D Orbital Map", "Hazard Risk Zone"])
    
    # CHART 1: 3D CLUSTER MAP
    with tab1:
        st.markdown(f"**Visualizing the {family_name}**")
        
        # Create a 3D scatter plot of the background asteroid data
        fig_3d = px.scatter_3d(
            df_background, x='semi_major_axis', y='eccentricity', z='inclination',
            color='cluster_label', opacity=0.3, 
            color_continuous_scale='Viridis',
            title="3D Orbital Classification Landscape"
        )
        
        # Add the new candidate asteroid to the 3D plot as a large red diamond
        fig_3d.add_trace(go.Scatter3d(
            x=[a_input], y=[e_input], z=[i_input],
            mode='markers', marker=dict(size=15, color='red', symbol='diamond'),
            name='New Candidate'
        ))
        
        st.plotly_chart(fig_3d, use_container_width=True)
        st.caption("The Red Diamond shows where your new asteroid fits within the known families.")

    # CHART 2: HAZARD SCATTER PLOT
    with tab2:
        st.markdown("**Risk Analysis: MOID vs Diameter**")
        
        # Create a 2D scatter plot of MOID vs. Diameter for the background data
        fig_2d = px.scatter(
            df_background, x='moid_au', y='diameter',
            color='pha_flag', 
            color_continuous_scale=['green', 'red'],
            title="The 'Death Zone' (Red = Hazardous)"
        )
        
        # Add the new candidate object to the 2D plot as a blue 'x'
        fig_2d.add_trace(go.Scatter(
            x=[moid_input], y=[predicted_diameter],
            mode='markers', marker=dict(size=20, color='blue', symbol='x'),
            name='Candidate Object'
        ))
        
        # Add a vertical line at MOID = 0.05 AU to indicate the danger threshold
        fig_2d.add_vline(x=0.05, line_width=2, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig_2d, use_container_width=True)
        st.caption("Objects to the left of the red dashed line are dangerously close to Earth.")

# --- INITIAL STATE ---
# Display a message prompting the user to start the analysis if the button hasn't been clicked yet
else:
    st.info("👈 Enter parameters in the sidebar and click **Run Hazard Analysis** to start.")