import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Auto-ML Predictive Dashboard",
    page_icon="🤖",
    layout="wide"
)

# --- HELPER FUNCTIONS ---
def get_model(task_type):
    """Returns a model based on the task type."""
    if task_type == 'Classification':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif task_type == 'Regression':
        return RandomForestRegressor(n_estimators=100, random_state=42)
    return None

# --- Initialize Session State ---
# This ensures our variables exist even on the first run
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'original_features' not in st.session_state:
    st.session_state.original_features = {}
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = []
if 'model_data' not in st.session_state:
    st.session_state.model_data = {}


# --- MAIN APP ---
st.title("🤖 Auto-ML Predictive Dashboard")
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Sidebar Configuration ---
    st.header("1. Model Configuration")
    target_column = st.selectbox("Select the Target Variable (Y)", df.columns, key='target_select')
    
    available_features = [col for col in df.columns if col != target_column]
    selected_features = st.multiselect("Select Feature Variables (X)", available_features, default=available_features, key='feature_select')

    # --- Train Button Logic ---
    if st.button("Train Model"):
        if target_column and selected_features:
            with st.spinner("Training model, please wait..."):
                # --- 1. Preprocessing ---
                X = df[selected_features]
                y = df[target_column]
                X_processed = pd.get_dummies(X, drop_first=True)
                
                # Store info about original features for the simulator
                original_features_info = {}
                for feature in selected_features:
                    if pd.api.types.is_numeric_dtype(df[feature]):
                        original_features_info[feature] = {
                            'type': 'numeric', 
                            'min': df[feature].min(), 
                            'max': df[feature].max(), 
                            'mean': df[feature].mean()
                        }
                    else:
                        original_features_info[feature] = {
                            'type': 'categorical', 
                            'unique': df[feature].unique()
                        }
                
                # --- 2. Task Type Detection ---
                if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                    task_type = "Regression"
                else:
                    task_type = "Classification"
                
                # --- 3. Model Training ---
                model = get_model(task_type)
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # --- 4. Store results in Session State ---
                st.session_state.model_trained = True
                st.session_state.trained_model = model
                st.session_state.original_features = original_features_info
                st.session_state.X_columns = X_processed.columns.tolist() # List of one-hot encoded columns
                
                # Store data needed for display
                st.session_state.model_data = {
                    'task_type': task_type,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'df_head': df.head(),
                    'df_shape': df.shape,
                    'target_describe': df[target_column].describe(),
                    'feature_importance': model.feature_importances_
                }
                
                st.success(f"Model trained successfully! Detected Task Type: **{task_type}**")
        else:
            st.error("Please select both target and feature variables.")

    # --- Display Results and Simulator IF Model is Trained ---
    if st.session_state.model_trained:
        st.markdown("---")
        st.header("2. Model Results")

        # --- Data Preview and Exploration ---
        col1, col2 = st.columns((2, 1))
        with col1:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.model_data['df_head'])
        with col2:
            st.subheader("Data Shape")
            st.write(f"Rows: {st.session_state.model_data['df_shape'][0]}, Columns: {st.session_state.model_data['df_shape'][1]}")
            st.write("Target Variable Info:")
            st.write(st.session_state.model_data['target_describe'])

        st.markdown("---")

        # --- Model Performance ---
        st.subheader("Model Performance")
        task_type = st.session_state.model_data['task_type']
        y_test = st.session_state.model_data['y_test']
        y_pred = st.session_state.model_data['y_pred']

        if task_type == "Classification":
            accuracy = accuracy_score(y_test, y_pred)
            st.metric(label="Accuracy Score", value=f"{accuracy:.2%}")
            
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

        else: # Regression
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            st.metric(label="R-squared (R²)", value=f"{r2:.3f}")
            st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.3f}")

        # Feature Importance
        st.subheader("Feature Importance")
        importances = st.session_state.model_data['feature_importance']
        feature_importance_df = pd.DataFrame({'feature': st.session_state.X_columns, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15)) # Show top 15
        ax.set_title('Top 15 Feature Importances')
        st.pyplot(fig)
        
        # --- "WHAT-IF" SIMULATOR ---
        st.markdown("---")
        st.header("3. What-If Simulator")
        
        user_input_raw = {}
        for feature, info in st.session_state.original_features.items():
            if info['type'] == 'numeric':
                min_val = float(info['min'])
                max_val = float(info['max'])
                mean_val = float(info['mean'])
                user_input_raw[feature] = st.slider(f"**{feature}**", min_val, max_val, value=mean_val)
            else: # Categorical
                unique_vals = info['unique']
                user_input_raw[feature] = st.selectbox(f"**{feature}**", unique_vals)

        if st.button("Predict"):
            # Create a dataframe from the raw user input
            input_df_raw = pd.DataFrame([user_input_raw])
            
            # Apply the *same* pd.get_dummies transformation
            input_df_processed = pd.get_dummies(input_df_raw, drop_first=True)
            
            # Ensure columns are in the same order as during training
            input_df_final = input_df_processed.reindex(columns=st.session_state.X_columns, fill_value=0) 
            
            prediction = st.session_state.trained_model.predict(input_df_final)
            
            st.markdown("---")
            st.subheader("Prediction Result")
            st.success(f"**{prediction[0]}**")
    
    # This message shows if the model *hasn't* been trained yet
    elif not st.session_state.model_trained:
         st.info("Configure the model parameters and click 'Train Model' to proceed.")

else:
    st.title("Welcome to the Auto-ML Dashboard!")
    st.info("Upload a CSV file to get started.")
    st.image("https.streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=400)
    st.markdown("""
    This app allows you to:
    - **Upload** your own dataset.
    - **Automatically train** a classification or regression model.
    - **Analyze** model performance and feature importance.
    - **Simulate** outcomes with the 'What-If' predictor.
    """)
    # Clear the state if the file is removed or changed
    for key in st.session_state.keys():
        del st.session_state[key]