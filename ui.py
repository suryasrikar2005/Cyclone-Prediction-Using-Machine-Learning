import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pyttsx3
from io import StringIO

def setup_page():
    """Sets up the Streamlit page configuration and custom CSS."""
    st.set_page_config(
        page_title="Cyclone Detection ML",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
        .main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; font-weight: bold; }
        .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; border-left: 5px solid #1f77b4; }
        .success-box { background-color: #d4edda; color: #155724; padding: 1rem; border-radius: 5px; border: 1px solid #c3e6cb; }
        .warning-box { background-color: #fff3cd; color: #856404; padding: 1rem; border-radius: 5px; border: 1px solid #ffeaa7; }
        .error-box { background-color: #f8d7da; color: #721c24; padding: 1rem; border-radius: 5px; border: 1px solid #f5c6cb; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🌪️ Cyclone Detection using Machine Learning</h1>', unsafe_allow_html=True)

def display_sidebar(dataset, processed_data, models):
    """Displays the sidebar with app information."""
    with st.sidebar:
        st.header("🌪️ App Information")
        st.info("""
        This application uses machine learning to detect cyclone formation based on meteorological parameters.
        
        **Features:**
        - Data preprocessing and analysis
        - KNN and Decision Tree models
        - Real-time prediction testing
        - Model performance comparison
        """)
        if dataset is not None:
            st.success(f"✅ Dataset loaded: {len(dataset)} records")
        if processed_data is not None:
            st.success("✅ Data preprocessed")
        if models:
            st.success(f"✅ {len(models)} model(s) trained")

def display_upload_tab():
    """Displays the data upload tab."""
    st.header("📁 Dataset Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            dataset = pd.read_csv(uploaded_file)
            st.markdown('<div class="success-box">✅ Dataset loaded successfully!</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(dataset))
            with col2:
                st.metric("Features", len(dataset.columns))
            st.subheader("Dataset Preview")
            st.dataframe(dataset.head())
            st.subheader("Dataset Info")
            buffer = StringIO()
            dataset.info(buf=buffer)
            st.text(buffer.getvalue())
            return dataset
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Error loading dataset: {str(e)}</div>', unsafe_allow_html=True)
    return None

def display_preprocessing_tab(dataset, processed_data):
    """Displays the data preprocessing tab."""
    st.header("⚙️ Data Preprocessing")
    if dataset is not None:
        if st.button("🔄 Preprocess Dataset", type="primary"):
            return True
        if processed_data:
             st.markdown('<div class="success-box">✅ Preprocessing completed successfully!</div>', unsafe_allow_html=True)
             col1, col2, col3, col4 = st.columns(4)
             with col1:
                 st.metric("Original Records", processed_data['original_size'])
             with col2:
                 st.metric("After Cleaning", processed_data['cleaned_size'])
             with col3:
                 st.metric("Training Set", len(processed_data['X_train']))
             with col4:
                 st.metric("Test Set", len(processed_data['X_test']))
    else:
        st.markdown('<div class="warning-box">⚠️ Please upload a dataset first!</div>', unsafe_allow_html=True)
    return False

def display_analysis_tab(dataset):
    """Displays the data analysis tab."""
    st.header("📊 Data Analysis & Visualization")
    if dataset is not None:
        data = dataset.copy()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cyclone Distribution")
            fig_pie = px.pie(data, names='Cyclone', title="Cyclone Occurrence", color_discrete_sequence=['#1f77b4', '#ff7f0e'])
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.subheader("Sea Surface Temperature vs Atmospheric Pressure")
            fig_scatter = px.scatter(data, x='Sea_Surface_Temperature', y='Atmospheric_Pressure', color='Cyclone', title="Sea Surface Temperature vs Atmospheric Pressure", color_discrete_sequence=['#3498db', '#e74c3c'])
            st.plotly_chart(fig_scatter, use_container_width=True)
        st.subheader("Correlation Heatmap")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Feature Correlation Matrix")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.markdown('<div class="warning-box">⚠️ Please upload a dataset first!</div>', unsafe_allow_html=True)

def display_knn_tab(processed_data, results):
    """Displays the KNN model tab."""
    st.header("🤖 K-Nearest Neighbors Model")
    if processed_data is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            k_value = st.slider("Select K value", min_value=1, max_value=20, value=5)
            if st.button("🚀 Train KNN Model", type="primary"):
                return k_value
        if 'knn' in results:
            with col2:
                st.subheader("KNN Model Performance")
                res = results['knn']
                st.metric("Accuracy", f"{res['acc']:.2%}")
                st.metric("Precision", f"{res['prec']:.2%}")
                st.metric("Recall", f"{res['recall']:.2%}")
                st.metric("F1-Score", f"{res['f1']:.2%}")
                st.subheader("Confusion Matrix")
                fig_cm = px.imshow(res['cm'], text_auto=True, labels=dict(x="Predicted", y="Actual"), x=['No Cyclone', 'Cyclone'], y=['No Cyclone', 'Cyclone'])
                st.plotly_chart(fig_cm)
    else:
        st.markdown('<div class="warning-box">⚠️ Please preprocess the data first!</div>', unsafe_allow_html=True)
    return None

def display_dt_tab(processed_data, results):
    """Displays the Decision Tree model tab."""
    st.header("🌳 Decision Tree Model")
    if processed_data is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            max_depth = st.slider("Select Max Depth", 2, 20, 5)
            min_samples_leaf = st.slider("Select Min Samples Leaf", 1, 20, 1)
            if st.button("🌳 Train Decision Tree Model", type="primary"):
                return max_depth, min_samples_leaf
        if 'dt' in results:
            with col2:
                st.subheader("Decision Tree Performance")
                res = results['dt']
                st.metric("Accuracy", f"{res['acc']:.2%}")
                st.metric("Precision", f"{res['prec']:.2%}")
                st.metric("Recall", f"{res['recall']:.2%}")
                st.metric("F1-Score", f"{res['f1']:.2%}")
                st.subheader("Confusion Matrix")
                fig_cm = px.imshow(res['cm'], text_auto=True, labels=dict(x="Predicted", y="Actual"), x=['No Cyclone', 'Cyclone'], y=['No Cyclone', 'Cyclone'])
                st.plotly_chart(fig_cm)
    else:
        st.markdown('<div class="warning-box">⚠️ Please preprocess the data first!</div>', unsafe_allow_html=True)
    return None

def display_rf_tab(processed_data, results):
    """Displays the Random Forest model tab."""
    st.header("🌳 Random Forest Model")
    if processed_data is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            n_estimators = st.slider("Select N Estimators", 10, 200, 100)
            max_depth = st.slider("Select Max Depth (RF)", 2, 20, 10)
            if st.button("🌳 Train Random Forest Model", type="primary"):
                return n_estimators, max_depth
        if 'rf' in results:
            with col2:
                st.subheader("Random Forest Performance")
                res = results['rf']
                st.metric("Accuracy", f"{res['acc']:.2%}")
                st.metric("Precision", f"{res['prec']:.2%}")
                st.metric("Recall", f"{res['recall']:.2%}")
                st.metric("F1-Score", f"{res['f1']:.2%}")
                st.subheader("Confusion Matrix")
                fig_cm = px.imshow(res['cm'], text_auto=True, labels=dict(x="Predicted", y="Actual"), x=['No Cyclone', 'Cyclone'], y=['No Cyclone', 'Cyclone'])
                st.plotly_chart(fig_cm)
    else:
        st.markdown('<div class="warning-box">⚠️ Please preprocess the data first!</div>', unsafe_allow_html=True)
    return None

def display_svm_tab(processed_data, results):
    """Displays the SVM model tab."""
    st.header("🤖 Support Vector Machine Model")
    if processed_data is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            C = st.slider("Select C value", 0.1, 10.0, 1.0)
            kernel = st.selectbox("Select Kernel", ['linear', 'rbf', 'poly'])
            if st.button("🤖 Train SVM Model", type="primary"):
                return C, kernel
        if 'svm' in results:
            with col2:
                st.subheader("SVM Performance")
                res = results['svm']
                st.metric("Accuracy", f"{res['acc']:.2%}")
                st.metric("Precision", f"{res['prec']:.2%}")
                st.metric("Recall", f"{res['recall']:.2%}")
                st.metric("F1-Score", f"{res['f1']:.2%}")
                st.subheader("Confusion Matrix")
                fig_cm = px.imshow(res['cm'], text_auto=True, labels=dict(x="Predicted", y="Actual"), x=['No Cyclone', 'Cyclone'], y=['No Cyclone', 'Cyclone'])
                st.plotly_chart(fig_cm)
    else:
        st.markdown('<div class="warning-box">⚠️ Please preprocess the data first!</div>', unsafe_allow_html=True)
    return None

def display_comparison_tab(results):
    """Displays the model comparison tab."""
    st.header("📈 Model Comparison")
    if results:
        res_df = pd.DataFrame(results).T.reset_index()
        res_df.rename(columns={'index': 'Model', 'acc': 'Accuracy', 'prec': 'Precision', 'recall': 'Recall', 'f1': 'F1-Score'}, inplace=True)
        res_df = res_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
        st.dataframe(res_df, hide_index=True)
    else:
        st.markdown('<div class="warning-box">⚠️ No models trained yet!</div>', unsafe_allow_html=True)

def display_prediction_tab(models, processed_data):
    """Displays the prediction test tab."""
    st.header("🧪 Prediction Test")
    if 'knn' in models and processed_data is not None:
        st.subheader("Enter Parameters for Cyclone Prediction")
        col1, col2 = st.columns(2)
        with col1:
            sea_temp = st.number_input("Sea Surface Temperature (°C)", value=28.5, step=0.1)
            pressure = st.number_input("Atmospheric Pressure (hPa)", value=1005.0, step=1.0)
            humidity = st.number_input("Relative Humidity (%)", value=80.0, step=1.0)
            wind_shear = st.number_input("Wind Shear (m/s)", value=10.0, step=1.0)
        with col2:
            vorticity = st.number_input("Vorticity (s⁻¹)", value=0.0001, step=0.00001, format="%.5f")
            latitude = st.number_input("Latitude", value=15.0, step=0.1)
            ocean_depth = st.number_input("Ocean Depth (m)", value=4000.0, step=1.0)
            coastline_distance = st.number_input("Proximity to Coastline (km)", value=100.0, step=1.0)
        
        if st.button("🔮 Predict Cyclone Risk", type="primary", use_container_width=True):
            try:
                input_data = np.array([[sea_temp, pressure, humidity, wind_shear, vorticity, latitude, ocean_depth, coastline_distance]])
                input_scaled = processed_data['scaler'].transform(input_data)
                prediction = models['knn'].predict(input_scaled)[0]
                prediction_proba = models['knn'].predict_proba(input_scaled)[0]
                
                if prediction == 1:
                    st.markdown("""
                    <div style="background-color: #f8d7da; color: #721c24; padding: 2rem; border-radius: 10px; text-align: center; border: 2px solid #f5c6cb;">
                        <h2>🌪️ CYCLONE DETECTED!</h2>
                        <h3>⚠️ HIGH RISK ALERT</h3>
                        <p>Confidence: {:.1f}%</p>
                    </div>
                    """.format(prediction_proba[1] * 100), unsafe_allow_html=True)
                    try:
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 150)
                        engine.say("Cyclone detected, alert the citizens as soon as possible")
                        engine.runAndWait()
                    except:
                        pass
                    st.warning("🚨 EMERGENCY ALERT: Cyclone conditions detected! Alert citizens immediately!")
                else:
                    st.markdown("""
                    <div style="background-color: #d4edda; color: #155724; padding: 2rem; border-radius: 10px; text-align: center; border: 2px solid #c3e6cb;">
                        <h2>✅ NO CYCLONE DETECTED</h2>
                        <h3>🌤️ SAFE CONDITIONS</h3>
                        <p>Confidence: {:.1f}%</p>
                    </div>
                    """.format(prediction_proba[0] * 100), unsafe_allow_html=True)
                
                st.subheader("Prediction Confidence")
                prob_df = pd.DataFrame({
                    'Outcome': ['No Cyclone', 'Cyclone'],
                    'Probability': [f"{prediction_proba[0]*100:.1f}%", f"{prediction_proba[1]*100:.1f}%"]
                })
                st.dataframe(prob_df, use_container_width=True)
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Prediction failed: {str(e)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ Please train the KNN model first!</div>', unsafe_allow_html=True)
