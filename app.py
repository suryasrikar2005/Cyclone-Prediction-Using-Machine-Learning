import streamlit as st
from model import model
from ui import ui

def main():
    """Main function to run the Streamlit application."""
    # Initialize session state
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'results' not in st.session_state:
        st.session_state.results = {}

    ui.setup_page()

    # Create tabs
    tabs = st.tabs([
        "📁 Upload Data", 
        "⚙️ Preprocessing", 
        "📊 Data Analysis", 
        "🤖 KNN Model", 
        "🌳 Decision Tree", 
        "🌳 Random Forest",
        "🤖 SVM",
        "📈 Model Comparison", 
        "🧪 Prediction Test"
    ])

    with tabs[0]:
        dataset = ui.display_upload_tab()
        if dataset is not None:
            st.session_state.dataset = dataset

    with tabs[1]:
        if ui.display_preprocessing_tab(st.session_state.dataset, st.session_state.processed_data):
            try:
                st.session_state.processed_data = model.preprocess_data(st.session_state.dataset)
                st.experimental_rerun()
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Preprocessing failed: {str(e)}</div>', unsafe_allow_html=True)

    with tabs[2]:
        ui.display_analysis_tab(st.session_state.dataset)

    with tabs[3]:
        k_value = ui.display_knn_tab(st.session_state.processed_data, st.session_state.results)
        if k_value:
            try:
                knn_model, knn_results = model.train_knn(st.session_state.processed_data, k_value)
                st.session_state.models['knn'] = knn_model
                st.session_state.results['knn'] = knn_results
                st.experimental_rerun()
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ KNN training failed: {str(e)}</div>', unsafe_allow_html=True)

    with tabs[4]:
        dt_params = ui.display_dt_tab(st.session_state.processed_data, st.session_state.results)
        if dt_params:
            try:
                max_depth, min_samples_leaf = dt_params
                dt_model, dt_results = model.train_decision_tree(st.session_state.processed_data, max_depth, min_samples_leaf)
                st.session_state.models['dt'] = dt_model
                st.session_state.results['dt'] = dt_results
                st.experimental_rerun()
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Decision Tree training failed: {str(e)}</div>', unsafe_allow_html=True)

    with tabs[5]:
        rf_params = ui.display_rf_tab(st.session_state.processed_data, st.session_state.results)
        if rf_params:
            try:
                n_estimators, max_depth = rf_params
                rf_model, rf_results = model.train_random_forest(st.session_state.processed_data, n_estimators, max_depth)
                st.session_state.models['rf'] = rf_model
                st.session_state.results['rf'] = rf_results
                st.experimental_rerun()
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Random Forest training failed: {str(e)}</div>', unsafe_allow_html=True)

    with tabs[6]:
        svm_params = ui.display_svm_tab(st.session_state.processed_data, st.session_state.results)
        if svm_params:
            try:
                C, kernel = svm_params
                svm_model, svm_results = model.train_svm(st.session_state.processed_data, C, kernel)
                st.session_state.models['svm'] = svm_model
                st.session_state.results['svm'] = svm_results
                st.experimental_rerun()
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ SVM training failed: {str(e)}</div>', unsafe_allow_html=True)

    with tabs[7]:
        ui.display_comparison_tab(st.session_state.results)

    with tabs[8]:
        ui.display_prediction_tab(st.session_state.models, st.session_state.processed_data)

    ui.display_sidebar(st.session_state.dataset, st.session_state.processed_data, st.session_state.models)

if __name__ == "__main__":
    main()
