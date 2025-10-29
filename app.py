import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Set page configuration
st.set_page_config(
    page_title="Water Quality Classifier",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== FUTURISTIC AQUA THEME (ONLY UI CHANGED) =====================
st.markdown("""
<style>
/* ---------- Global & Background ---------- */
html, body, [class*="css"] {
    background: radial-gradient(circle at 10% 10%, #07142a 0%, #00121a 30%, #000814 100%) !important;
    color: #e0f7fa;
    font-family: 'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* ---------- Main Header ---------- */
.main-header {
    font-size: 2.8rem;
    color: #00f0ff;
    text-align: center;
    font-weight: 800;
    letter-spacing: 1px;
    text-shadow: 0 0 18px rgba(0,240,255,0.12);
    margin-top: 6px;
    margin-bottom: 6px;
}

/* ---------- Sub Header ---------- */
.sub-header {
    font-size: 1.4rem;
    color: #9be9f8;
    font-weight: 600;
}

/* ---------- Info Box ---------- */
.info-box {
    background: linear-gradient(90deg, rgba(0,44,66,0.45), rgba(0,60,90,0.25));
    border-left: 5px solid rgba(0, 224, 255, 0.7);
    padding: 14px;
    border-radius: 10px;
    color: #cfeff6;
}

/* ---------- Input Cards ---------- */
.reportview-container .main .block-container{
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
.input-card {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.45);
    margin-bottom: 12px;
}

/* ---------- Number Input Label ---------- */
.stNumberInput > label {
    font-weight: 600;
    color: #bfefff;
}

/* ---------- Result Card (Glass + Glow) ---------- */
.result-box {
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 20px;
    font-size: 1.15rem;
    font-weight: 700;
    text-align: center;
    color: #041722;
    box-shadow: 0 10px 30px rgba(0,255,255,0.06);
    backdrop-filter: blur(8px);
}

.moderate {
    background: linear-gradient(90deg, #fff176, #ffb300);
    color: #04262b;
}
.poor {
    background: linear-gradient(90deg, #ff8a80, #ff5252);
    color: #fff;
}
.very-poor {
    background: linear-gradient(90deg, #ef5350, #c62828);
    color: #fff;
}

/* ---------- Buttons ---------- */
div.stButton > button {
    background: linear-gradient(90deg, #00eaff, #0077b6);
    color: #012022;
    font-weight: 800;
    border-radius: 12px;
    padding: 10px 18px;
    border: none;
    box-shadow: 0 8px 26px rgba(0,230,255,0.08);
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
div.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 36px rgba(0,230,255,0.16);
}

/* ---------- Tabs Styling ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 0.6rem; }
.stTabs [data-baseweb="tab"] {
    padding: 8px 14px;
    border-radius: 10px;
    background-color: rgba(255,255,255,0.03);
    color: #9be9f8;
    font-weight: 700;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg,#00eaff,#0077b6);
    color: #00121a !important;
}

/* ---------- Sidebar (Futuristic Card) ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(0,20,35,0.95), rgba(0,8,20,0.95));
    border-right: 1px solid rgba(0,230,255,0.04);
    padding: 1rem 1.2rem;
    color: #d0f7fb;
}
.sidebar-title {
    font-size: 1.1rem;
    color: #00f0ff;
    font-weight: 800;
    border-bottom: 2px solid rgba(0,240,255,0.08);
    padding-bottom: 0.25rem;
    margin-bottom: 0.6rem;
}
.sidebar-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(0,230,255,0.06);
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: #c9f3fb;
}

/* ---------- Small text in sidebar ---------- */
.sidebar-card small { color: #92e9f6; }

/* ---------- Plot tweaks ---------- */
.stPlotlyChart, .stAltairChart, .stVegaLiteChart { background: rgba(255,255,255,0.02); border-radius: 8px; }

/* ---------- Remove footer ---------- */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
# ===============================================================================================

# Function to assign WQI label based on WQI value
def assign_wqi_label(wqi):
    if wqi >= 51:
        return 'Moderate'
    elif wqi >= 26:
        return 'Poor'
    else:
        return 'Very Poor'

# Function to load model and components
@st.cache_resource
def load_model_components():
    try:
        # Load model
        with open('best_water_quality_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Create new label mapping for three classes only
        label_mapping = {
            'Very Poor': 0,
            'Poor': 1,
            'Moderate': 2
        }
        
        # Save the new label mapping
        with open('label_mapping.pkl', 'wb') as f:
            pickle.dump(label_mapping, f)
        
        # Load feature names
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load label encoders if they exist
        label_encoders = None
        if os.path.exists('label_encoders.pkl'):
            with open('label_encoders.pkl', 'rb') as f:
                label_encoders = pickle.load(f)
        
        # Reverse the label mapping for display
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        
        return model, scaler, label_mapping, reverse_mapping, feature_names, label_encoders
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None, None, None, None

# Function to load demo cases
@st.cache_data
def load_demo_cases():
    try:
        demo_df = pd.read_csv('demo_test_cases.csv')
        return demo_df
    except Exception as e:
        st.error(f"Error loading demo cases: {e}")
        return None

# Function to get feature ranges from dataset
@st.cache_data
def get_feature_ranges():
    try:
        df = pd.read_csv('water_quality_with_final_wqi.csv')
        ranges = {}
        for col in df.columns:
            if col not in ['WQI', 'Water_Quality_Index', 'WQI_Label']:
                ranges[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    '25%': float(df[col].quantile(0.25)),
                    '75%': float(df[col].quantile(0.75))
                }
        return ranges
    except Exception as e:
        st.error(f"Error getting feature ranges: {e}")
        return {}

# Function to make prediction
def predict_water_quality(input_data, model, scaler, feature_names, reverse_mapping):
    try:
        # Convert input to DataFrame with correct feature order
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Get the predicted label
        predicted_label = reverse_mapping[prediction]
        
        # Get class probabilities
        class_probabilities = {}
        for i, prob in enumerate(prediction_proba):
            class_probabilities[reverse_mapping[i]] = prob
        
        return predicted_label, class_probabilities
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

# Function to get feature importance
def get_feature_importance(model, feature_names):
    try:
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            return None
        
        # Create DataFrame for feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    except Exception as e:
        st.error(f"Error getting feature importance: {e}")
        return None

# Main function
def main():
    # Load model and components
    model, scaler, label_mapping, reverse_mapping, feature_names, label_encoders = load_model_components()
    
    # Load demo cases
    demo_cases = load_demo_cases()
    
    # Get feature ranges
    feature_ranges = get_feature_ranges()
    
    # Title and description
    st.markdown("<h1 class='main-header'>Water Quality Classification System</h1>", unsafe_allow_html=True)
    st.markdown("""
    This application predicts water quality class based on various water parameters using a machine learning model.
    The classification follows the Central Pollution Control Board (CPCB) standards for Water Quality Index (WQI).
    """)
    
    # Display CPCB label mapping
    st.markdown("<h2 class='sub-header'>CPCB Water Quality Classification</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>Water Quality Classes:</h3>
        <ul>
            <li><strong>Moderate (51-100):</strong> Water is suitable for drinking after conventional treatment</li>
            <li><strong>Poor (26-50):</strong> Water can be used for wildlife and fisheries</li>
            <li><strong>Very Poor (0-25):</strong> Water is suitable only for controlled waste disposal</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for input methods
    input_tab, demo_tab = st.tabs(["Manual Input", "Use Demo Cases"])
    
    # Manual Input Tab
    with input_tab:
        st.markdown("<h2 class='sub-header'>Enter Water Parameters</h2>", unsafe_allow_html=True)
        
        # Create columns for input fields
        col1, col2 = st.columns(2)
        
        # Dictionary to store input values
        input_data = {}
        
        # Create input fields for each feature
        for i, feature in enumerate(feature_names):
            # Get feature range
            feature_range = feature_ranges.get(feature, {})
            min_val = feature_range.get('min', 0.0)
            max_val = feature_range.get('max', 100.0)
            default_val = feature_range.get('median', (min_val + max_val) / 2)
            
            # Create input field in alternating columns
            with col1 if i % 2 == 0 else col2:
                input_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=float(min_val - 0.1 * (max_val - min_val)),
                    max_value=float(max_val + 0.1 * (max_val - min_val)),
                    value=float(default_val),
                    step=0.01,
                    format="%.4f"
                )
        
        # Predict button
        if st.button("Predict Water Quality", key="predict_button"):
            if model is not None and scaler is not None:
                # Make prediction
                predicted_label, class_probabilities = predict_water_quality(
                    input_data, model, scaler, feature_names, reverse_mapping
                )
                
                if predicted_label:
                    # Display result
                    st.markdown(f"<h2 class='sub-header'>Prediction Result</h2>", unsafe_allow_html=True)
                    
                    # Determine CSS class based on prediction
                    css_class = predicted_label.lower().replace(" ", "-")
                    
                    # Display prediction in styled box
                    st.markdown(f"""
                    <div class="result-box {css_class}">
                        <h3>Predicted Water Quality Class: {predicted_label}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display class probabilities
                    st.subheader("Class Probabilities")
                    
                    # Create probability DataFrame
                    prob_df = pd.DataFrame({
                        'Class': list(class_probabilities.keys()),
                        'Probability': list(class_probabilities.values())
                    })
                    
                    # Sort by class order
                    class_order = ['Very Poor', 'Poor', 'Moderate']
                    prob_df['Class_Order'] = prob_df['Class'].apply(lambda x: class_order.index(x))
                    prob_df = prob_df.sort_values('Class_Order').drop('Class_Order', axis=1)
                    
                    # Plot probabilities
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = sns.barplot(x='Class', y='Probability', data=prob_df, palette='viridis', ax=ax)
                    
                    # Add probability values on top of bars
                    for i, p in enumerate(bars.patches):
                        bars.annotate(f'{p.get_height():.4f}', 
                                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                                     ha = 'center', va = 'bottom', 
                                     xytext = (0, 5), textcoords = 'offset points')
                    
                    plt.title('Probability of Each Water Quality Class')
                    plt.ylim(0, 1)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display feature importance
                    importance_df = get_feature_importance(model, feature_names)
                    if importance_df is not None:
                        st.subheader("Feature Importance")
                        
                        # Plot feature importance
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis', ax=ax)
                        plt.title('Top 10 Most Important Features')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Highlight top 3 contributing features for this prediction
                        st.markdown("### Top Contributing Features for This Prediction")
                        top_features = importance_df.head(3)['Feature'].tolist()
                        
                        for feature in top_features:
                            feature_value = input_data[feature]
                            st.markdown(f"- **{feature}**: {feature_value:.4f}")
    
    # Demo Cases Tab
    with demo_tab:
        st.markdown("<h2 class='sub-header'>Demo Cases</h2>", unsafe_allow_html=True)
        st.markdown("Select a pre-defined water quality sample to see the prediction.")
        
        if demo_cases is not None:
            # Create a dropdown for demo cases
            demo_options = [f"Sample {i+1}: {row['Actual_Label']}" for i, row in demo_cases.iterrows()]
            selected_demo = st.selectbox("Select a demo case", options=demo_options)
            
            if selected_demo:
                # Get the selected demo case
                selected_index = demo_options.index(selected_demo)
                demo_row = demo_cases.iloc[selected_index]
                
                # Display the selected demo case
                st.markdown("### Selected Sample Parameters")
                
                # Create columns for displaying parameters
                col1, col2 = st.columns(2)
                
                # Display parameters in two columns
                demo_features = {}
                for i, feature in enumerate(feature_names):
                    # Skip non-feature columns
                    if feature in demo_row:
                        demo_features[feature] = demo_row[feature]
                        with col1 if i % 2 == 0 else col2:
                            st.markdown(f"**{feature}**: {demo_row[feature]:.4f}")
                
                # Display actual WQI and label
                st.markdown(f"**Actual WQI**: {demo_row['Actual_WQI']:.2f}")
                st.markdown(f"**Actual Label**: {demo_row['Actual_Label']}")
                
                # Predict button for demo
                if st.button("Predict Water Quality", key="demo_predict_button"):
                    if model is not None and scaler is not None:
                        # Make prediction
                        predicted_label, class_probabilities = predict_water_quality(
                            demo_features, model, scaler, feature_names, reverse_mapping
                        )
                        
                        if predicted_label:
                            # Display result
                            st.markdown(f"<h2 class='sub-header'>Prediction Result</h2>", unsafe_allow_html=True)
                            
                            # Determine CSS class based on prediction
                            css_class = predicted_label.lower().replace(" ", "-")
                            
                            # Display prediction in styled box
                            st.markdown(f"""
                            <div class="result-box {css_class}">
                                <h3>Predicted Water Quality Class: {predicted_label}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display class probabilities
                            st.subheader("Class Probabilities")
                            
                            # Create probability DataFrame
                            prob_df = pd.DataFrame({
                                'Class': list(class_probabilities.keys()),
                                'Probability': list(class_probabilities.values())
                            })
                            
                            # Sort by class order
                            class_order = ['Very Poor', 'Poor', 'Moderate', 'Good', 'Excellent']
                            prob_df['Class_Order'] = prob_df['Class'].apply(lambda x: class_order.index(x))
                            prob_df = prob_df.sort_values('Class_Order').drop('Class_Order', axis=1)
                            
                            # Plot probabilities
                            fig, ax = plt.subplots(figsize=(10, 5))
                            bars = sns.barplot(x='Class', y='Probability', data=prob_df, palette='viridis', ax=ax)
                            
                            # Add probability values on top of bars
                            for i, p in enumerate(bars.patches):
                                bars.annotate(f'{p.get_height():.4f}', 
                                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                                             ha = 'center', va = 'bottom', 
                                             xytext = (0, 5), textcoords = 'offset points')
                            
                            plt.title('Probability of Each Water Quality Class')
                            plt.ylim(0, 1)
                            plt.tight_layout()
                            st.pyplot(fig)
    
    # Sidebar with typical ranges
    st.sidebar.title("Typical Parameter Ranges")
    st.sidebar.markdown("Reference ranges for water quality parameters:")
    
    for feature, range_info in feature_ranges.items():
        st.sidebar.markdown(f"**{feature}**")
        st.sidebar.markdown(f"- Min: {range_info['min']:.2f}")
        st.sidebar.markdown(f"- Max: {range_info['max']:.2f}")
        st.sidebar.markdown(f"- Mean: {range_info['mean']:.2f}")
        st.sidebar.markdown(f"- Median: {range_info['median']:.2f}")
        st.sidebar.markdown("---")
    
    # How labels were defined section
    st.sidebar.title("How Labels Were Defined")
    st.sidebar.markdown("""
    The water quality labels are based on the Central Pollution Control Board (CPCB) standards for Water Quality Index (WQI):
    
    - **Excellent (91-100)**: Water is suitable for drinking with conventional treatment and disinfection
    - **Good (71-90)**: Water is suitable for outdoor bathing
    - **Moderate (51-70)**: Water is suitable for drinking after conventional treatment
    - **Poor (26-50)**: Water can be used for wildlife and fisheries
    - **Very Poor (0-25)**: Water is suitable for irrigation, industrial cooling, controlled waste disposal
    
    These classifications help in determining appropriate water usage based on quality parameters.
    """)

# Run the app
if __name__ == "__main__":
    main()


# Function to assign WQI label based on WQI value
def assign_wqi_label(wqi):
    if wqi >= 51:
        return 'Moderate'
    elif wqi >= 26:
        return 'Poor'
    else:
        return 'Very Poor'
