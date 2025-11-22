import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="CardioCare AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #e74c3c;
        text-align: center;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .healthy {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .disease {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

# Define the same model architecture
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TabularTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, d_ff=128, num_classes=2, dropout=0.1):
        super(TabularTransformer, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.input_embedding(x)
        x = x.unsqueeze(1)
        x = x + self.position_embedding
        x = self.dropout(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x.squeeze(1)
        output = self.classifier(x)
        return output

# Load models and preprocessors
@st.cache_resource
def load_models_and_preprocessors():
    """Load all models and preprocessing objects"""
    try:
        # Load preprocessing objects
        with open('saved_models/preprocessing_objects.pkl', 'rb') as f:
            preprocessing_data = pickle.load(f)
        
        # Load model configurations
        with open('saved_models/model_configs.json', 'r') as f:
            model_configs = json.load(f)
        
        # Load performance metrics
        with open('saved_models/performance_metrics.pkl', 'rb') as f:
            performance_metrics = pickle.load(f)
        
        # Load models
        models = {'binary': {}, 'multiclass': {}}
        sampling_techniques = ['No_Sampling', 'SMOTE', 'SMOTETomek']
        
        for task_type in ['binary', 'multiclass']:
            for technique in sampling_techniques:
                model_path = f'saved_models/{task_type}_{technique}_model.pth'
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                config = checkpoint['model_config']
                
                model = TabularTransformer(
                    input_dim=config['input_dim'],
                    d_model=config['d_model'],
                    num_heads=config['num_heads'],
                    num_layers=config['num_layers'],
                    d_ff=config['d_ff'],
                    num_classes=config['num_classes'],
                    dropout=config['dropout']
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                models[task_type][technique] = model
        
        return models, preprocessing_data, performance_metrics
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Helper function for model predictions
def model_predict_proba(X, model):
    """Predict probabilities for SHAP and LIME"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        return probs.numpy()

# Generate SHAP explanations
def generate_shap_explanations(model, X_background, X_test, feature_names, class_idx=1):
    """Generate SHAP waterfall and force plots"""
    
    def model_predict(X):
        return model_predict_proba(X, model)
    
    # Create SHAP explainer with smaller background
    background_size = min(50, len(X_background))
    explainer = shap.KernelExplainer(model_predict, X_background[:background_size])
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Handle both binary and multiclass
    if isinstance(shap_values, list):
        # For list output (multi-class), get the specific class
        values = shap_values[class_idx][0]  # [0] to get first sample
        base_value = explainer.expected_value[class_idx]
    else:
        # For single array output, select the class column
        if len(shap_values.shape) == 3:
            # Shape is (n_samples, n_features, n_classes)
            values = shap_values[0, :, class_idx]
        elif len(shap_values.shape) == 2:
            # Shape is (n_features, n_classes) - already single sample
            values = shap_values[:, class_idx]
        else:
            # Shape is (n_features,) - single output
            values = shap_values[0] if shap_values.ndim > 1 else shap_values
        
        base_value = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    
    return values, base_value, explainer.expected_value

# Generate LIME explanations
def generate_lime_explanation(model, X_train, X_test, feature_names, class_names):
    """Generate LIME explanation"""
    
    def model_predict(X):
        return model_predict_proba(X, model)
    
    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    
    # Generate explanation
    explanation = explainer.explain_instance(
        X_test[0], 
        model_predict, 
        num_features=len(feature_names)
    )
    
    return explanation

# Initialize
models, preprocessing_data, performance_metrics = load_models_and_preprocessors()

if models is None:
    st.error("Models not found! Please ensure the saved_models directory exists.")
    st.stop()

st.title("🫀 CardioCare AI")

st.markdown(
    """
    <h3 style='text-align: center; margin-top: -20px;'>
        AI-Powered Cardiac Risk Assessment with Explainable AI
    </h3>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    task_type = st.radio(
        "Prediction Type:",
        ["Binary (Disease/No Disease)", "Multiclass (Severity Levels)"],
        help="Choose between binary classification or severity level prediction"
    )
    task = "binary" if "Binary" in task_type else "multiclass"
    
    sampling_technique = st.selectbox(
        "Sampling Technique:",
        ["No_Sampling", "SMOTE", "SMOTETomek"],
        help="Select the data balancing technique"
    )
    
    st.divider()

# Main content with 5 tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Make Prediction", " Model Analysis", " About", " Usage Guide"])

with tab1:
    st.header("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        
        st.subheader("Medical History")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
                          format_func=lambda x: "Yes" if x == 1 else "No")
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                              format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
    
    with col2:
        st.subheader("Symptoms")
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", 
                                               "Non-anginal Pain", "Asymptomatic"][x])
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No")
        
        st.subheader("Test Results")
        slope = st.selectbox("ST Slope", options=[0, 1, 2],
                            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    
    with col3:
        st.subheader("Clinical Measurements")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                   min_value=90, max_value=200, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 
                              min_value=100, max_value=600, value=200)
        thalach = st.number_input("Max Heart Rate Achieved", 
                                 min_value=60, max_value=220, value=150)
        oldpeak = st.number_input("ST Depression (0-6)", 
                                 min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        ca = st.number_input("Number of Major Vessels (0-4)", 
                            min_value=0, max_value=4, value=0)
        thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                           format_func=lambda x: ["Normal", "Fixed Defect", 
                                                 "Reversible Defect", "Unknown"][x])
    
    st.divider()
    
    if st.button("🔬 Predict", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
            'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
            'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
        })
        
        # Preprocess
        scaler = preprocessing_data['scaler']
        encoders = preprocessing_data['encoders']
        numerical_features = preprocessing_data['numerical_features']
        categorical_features = preprocessing_data['categorical_features']
        feature_cols = preprocessing_data['feature_cols']
        
        input_processed = input_data.copy()
        input_processed[numerical_features] = scaler.transform(input_processed[numerical_features])
        
        for col in categorical_features:
            if col in input_processed.columns:
                input_processed[col] = encoders[col].transform(input_processed[col])
        
        X_input = torch.FloatTensor(input_processed[feature_cols].values)
        X_input_numpy = input_processed[feature_cols].values
        
        # Make prediction
        model = models[task][sampling_technique]
        with torch.no_grad():
            output = model(X_input)
            probs = torch.softmax(output, dim=1).numpy()[0]
            prediction = np.argmax(probs)
        
        # Display results
        st.divider()
        
        if task == "binary":
            if prediction == 0:
                st.markdown("""
                    <div class="prediction-box healthy">
                        <h2 style="color: #28a745;"> No Heart Disease Detected</h2>
                        <p style="font-size: 18px;">The model predicts a low risk of heart disease.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="prediction-box disease">
                        <h2 style="color: #dc3545;"> Heart Disease Detected</h2>
                        <p style="font-size: 18px;">The model predicts presence of heart disease.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Probability visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=['No Disease', 'Disease'],
                    y=[probs[0]*100, probs[1]*100],
                    marker_color=['#28a745', '#dc3545'],
                    text=[f'{probs[0]*100:.1f}%', f'{probs[1]*100:.1f}%'],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Prediction Confidence",
                yaxis_title="Probability (%)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # multiclass
            severity_labels = ["No Disease", "Mild", "Moderate", "Severe", "Very Severe"]
            colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1']
            
            st.markdown(f"""
                <div class="prediction-box" style="background-color: {colors[prediction]}22; border-color: {colors[prediction]};">
                    <h2 style="color: {colors[prediction]};">Predicted: {severity_labels[prediction]}</h2>
                    <p style="font-size: 18px;">Confidence: {probs[prediction]*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Probability distribution
            fig = go.Figure(data=[
                go.Bar(
                    x=severity_labels[:len(probs)],
                    y=probs*100,
                    marker_color=colors[:len(probs)],
                    text=[f'{p*100:.1f}%' for p in probs],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Severity Level Probabilities",
                yaxis_title="Probability (%)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors
        st.subheader(" Key Risk Factors")
        risk_factors = []
        
        if age > 60:
            risk_factors.append(("Age", f"{age} years (elevated risk)", "warning"))
        if trestbps > 140:
            risk_factors.append(("Blood Pressure", f"{trestbps} mm Hg (high)", "error"))
        if chol > 240:
            risk_factors.append(("Cholesterol", f"{chol} mg/dl (high)", "error"))
        if thalach < 100:
            risk_factors.append(("Max Heart Rate", f"{thalach} bpm (low)", "warning"))
        if oldpeak > 2:
            risk_factors.append(("ST Depression", f"{oldpeak} (significant)", "error"))
        if ca >= 2:
            risk_factors.append(("Vessel Blockage", f"{ca} major vessels", "error"))
        
        if risk_factors:
            for factor, value, status in risk_factors:
                if status == "error":
                    st.error(f"**{factor}**: {value}")
                else:
                    st.warning(f"**{factor}**: {value}")
        else:
            st.success(" No significant risk factors detected in clinical measurements.")
        
        # Explainable AI Section
        st.divider()
        st.header(" Explainable AI - Understanding the Prediction")
        
        st.markdown("""
        This section provides insights into **why** the model made this prediction. 
        We use two advanced explainability techniques:
        - **SHAP (SHapley Additive exPlanations)**: Shows feature importance and contribution
        - **LIME (Local Interpretable Model-agnostic Explanations)**: Provides local explanations
        """)
        
        # Create background dataset for SHAP
        np.random.seed(42)
        n_background = 100
        
        X_background = np.random.randn(n_background, len(feature_cols))
        
        for i, col in enumerate(feature_cols):
            if col in numerical_features:
                idx = numerical_features.index(col)
                X_background[:, i] = X_background[:, i] * scaler.scale_[idx] + scaler.mean_[idx]
            else:
                X_background[:, i] = np.random.randint(0, 3, n_background)
        
        X_background_df = pd.DataFrame(X_background, columns=feature_cols)
        X_background_processed = X_background_df.copy()
        X_background_processed[numerical_features] = scaler.transform(X_background_processed[numerical_features])
        for col in categorical_features:
            if col in X_background_processed.columns:
                valid_classes = encoders[col].classes_
                X_background_processed[col] = np.clip(
                    X_background_processed[col].astype(int), 
                    0, 
                    len(valid_classes) - 1
                )
        
        X_background_array = X_background_processed.values
        
        try:
            with st.spinner(" Generating SHAP explanations..."):
                class_idx = prediction if task == "multiclass" else 1
                shap_values, base_value, expected_values = generate_shap_explanations(
                    model, 
                    X_background_array, 
                    X_input_numpy,
                    feature_cols,
                    class_idx
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(" SHAP Waterfall Plot")
                    st.markdown("Shows how each feature contributes to pushing the prediction from the base value.")
                    
                    fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 8))
                    
                    explanation = shap.Explanation(
                        values=shap_values,
                        base_values=base_value if np.isscalar(base_value) else base_value,
                        data=X_input_numpy[0],
                        feature_names=feature_cols
                    )
                    
                    shap.plots.waterfall(explanation, show=False)
                    st.pyplot(fig_waterfall, use_container_width=True)
                    plt.close()
                
                with col2:
                    st.subheader(" Feature Contributions")
                    st.markdown("Shows positive (red) and negative (blue) contributions to the prediction.")
                    
                    feature_impact_force = pd.DataFrame({
                        'Feature': feature_cols,
                        'SHAP Value': shap_values
                    })
                    feature_impact_force = feature_impact_force.sort_values('SHAP Value')
                    
                    colors = ['#ff0051' if x > 0 else '#008bfb' for x in feature_impact_force['SHAP Value']]
                    
                    fig_force = go.Figure(data=[
                        go.Bar(
                            y=feature_impact_force['Feature'],
                            x=feature_impact_force['SHAP Value'],
                            orientation='h',
                            marker=dict(color=colors),
                            text=[f'{val:.4f}' for val in feature_impact_force['SHAP Value']],
                            textposition='auto',
                        )
                    ])
                    
                    fig_force.update_layout(
                        title=f"Feature Contributions (Base value: {base_value:.3f})",
                        xaxis_title="SHAP Value (Impact on Prediction)",
                        yaxis_title="Feature",
                        height=600,
                        showlegend=False
                    )
                    
                    fig_force.add_vline(x=0, line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig_force, use_container_width=True)
                    
                    
                
               
                
        except Exception as e:
            st.error(f"Error generating SHAP explanations: {e}")
        
        st.divider()
        
        try:
            with st.spinner(" Generating LIME explanation..."):
                st.subheader("🔬 LIME Explanation")
                st.markdown("Shows which features had the most influence on this specific prediction.")
                
                if task == "binary":
                    class_names = ['No Disease', 'Disease']
                else:
                    class_names = ['No Disease', 'Mild', 'Moderate', 'Severe', 'Very Severe']
                
                lime_explanation = generate_lime_explanation(
                    model,
                    X_background_array,
                    X_input_numpy,
                    feature_cols,
                    class_names
                )
                
                # Display LIME plot in columns to match SHAP layout
                col1_lime, col2_lime = st.columns([1, 1])
                
                with col1_lime:
                    # Display LIME plot with constrained size
                    fig_lime = lime_explanation.as_pyplot_figure()
                    fig_lime.set_size_inches(10, 8)
                    st.pyplot(fig_lime, use_container_width=True)
                    plt.close()
                
                with col2_lime:
                    st.subheader(" LIME Feature Weights")
                    
                    lime_list = lime_explanation.as_list()
                    lime_df = pd.DataFrame(lime_list, columns=['Feature', 'Weight'])
                    lime_df['Abs_Weight'] = np.abs(lime_df['Weight'])
                    lime_df = lime_df.sort_values('Abs_Weight', ascending=False)
                    
                    st.dataframe(lime_df.style.format({'Weight': '{:.4f}', 'Abs_Weight': '{:.4f}'}), 
                               use_container_width=True, height=400)
                
                st.info("""
                 How to interpret these results:
                
                - **Positive values** (red/orange): Push prediction towards disease/higher severity
                - **Negative values** (blue/green): Push prediction towards no disease/lower severity
                - **Larger absolute values**: More influential features
                
                Both SHAP and LIME help understand which patient characteristics most influenced this diagnosis.
                """)
                
        except Exception as e:
            st.error(f"Error generating LIME explanation: {e}")

with tab2:
    st.header(" Model Performance Analysis")
    
    if performance_metrics:
        st.subheader(" Performance Comparison Across Techniques")
        
        metrics_df = []
        for tech in ["No Sampling", "SMOTE", "SMOTETomek"]:
            if tech in performance_metrics[task]:
                metrics = performance_metrics[task][tech]
                metrics_df.append({
                    'Technique': tech,
                    'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1-Score': metrics['F1-Score'],
                    'AUC': metrics['AUC']
                })
        
        df = pd.DataFrame(metrics_df)
        
        fig = go.Figure()
        for idx, row in df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['AUC']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
                fill='toself',
                name=row['Technique']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Performance Metrics Comparison",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df.style.format({
            'Accuracy': '{:.3f}',
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1-Score': '{:.3f}',
            'AUC': '{:.3f}'
        }).highlight_max(axis=0, color='lightgreen'), use_container_width=True)

with tab3:
    st.markdown("""
    ###  CardioCare AI
    
    This application uses state-of-the-art **Transformer neural networks** to predict heart disease 
    from clinical measurements. The system provides two types of predictions:
    
    #### Binary Classification
    - Predicts presence or absence of heart disease
    - Provides confidence scores for each class
    
    #### Multiclass Classification
    - Predicts severity levels (0-4)
    - Helps in risk stratification
    
    ###  Model Architecture
    
    The system uses a **Tabular Transformer** architecture with:
    - Multi-head attention mechanisms
    - Layer normalization
    - Feed-forward neural networks
    - Dropout regularization
    
    ###  Explainable AI Features
    
    The system includes advanced explainability tools:
    
    1. **SHAP (SHapley Additive exPlanations)**
       - Waterfall plots showing feature contributions
       - Force plots for interactive visualization
       - Feature importance rankings
    
    2. **LIME (Local Interpretable Model-agnostic Explanations)**
       - Local explanations for individual predictions
       - Feature weight analysis
    
    ###  Data Balancing Techniques
    
    Three sampling techniques are available:
    
    1. **No Sampling**: Uses original data distribution
    2. **SMOTE**: Synthetic Minority Over-sampling Technique
    3. **SMOTETomek**: Combined over-sampling and under-sampling
    
    ###  Disclaimer
    
    This tool is for educational and research purposes only. It should not be used as a substitute 
    for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
    professionals for medical decisions.
    
    ###  Technical Details
    
    - **Framework**: PyTorch
    - **Model**: Transformer-based Neural Network
    - **Training**: 5-Fold Cross-Validation
    - **Dataset**: Heart Disease Dataset (UCI)
    - **Explainability**: SHAP & LIME
    """)

with tab4:
    st.header(" Usage Guide")
    
   
    st.markdown("""
        <div class="feature-card">
            <h3>🎥 Video Tutorial</h3>
            <p>Watch this short video to understand the complete workflow:</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Placeholder for video
    st.video("https://youtu.be/13eq--U-TBA")  # Replace with actual video URL
    
    
    st.markdown("""
    ###  Step-by-Step Guide
    
    #### 1️ **Configuration (Sidebar)**
    
    - **Select Prediction Type**: Choose between Binary (Disease/No Disease) or Multiclass (Severity Levels)
    - **Select Sampling Technique**: Pick from No_Sampling, SMOTE, or SMOTETomek
    
    #### 2️ **Enter Patient Information**
    
    Fill in all required fields across three sections:
    
    **Demographics & Medical History:**
    - Age (20-100 years)
    - Sex (Male/Female)
    - Fasting Blood Sugar status
    - Resting ECG results
    
    **Symptoms:**
    - Chest Pain Type (4 categories)
    - Exercise Induced Angina
    - ST Slope measurement
    
    **Clinical Measurements:**
    - Resting Blood Pressure
    - Serum Cholesterol
    - Maximum Heart Rate Achieved
    - ST Depression (oldpeak)
    - Number of Major Vessels colored by fluoroscopy
    - Thalassemia type
    
    #### 3️ **Make Prediction**
    
    - Click the **"🔬 Predict"** button
    - Wait for the model to process (usually < 1 second)
    
    #### 4️ **Interpret Results**
    
    The system provides comprehensive results:
    
    **Prediction Display:**
    - Clear indication of disease presence/absence or severity level
    - Confidence percentage
    - Visual probability charts
    
    **Risk Factors:**
    - Automated identification of elevated risk factors
    - Color-coded warnings (red = high risk, yellow = moderate)
    
    **Explainable AI Section:**
    
    - **SHAP Waterfall Plot**: Shows how each feature pushes the prediction up or down
    - **Feature Contributions**: Horizontal bar chart of positive/negative impacts
    - **LIME Explanation**: Local interpretation showing which features mattered most for this specific patient
    
    #### 5️ **Review Model Performance**
    
    Navigate to the **" Model Analysis"** tab to see:
    - Performance metrics across different sampling techniques
    - Radar charts comparing Accuracy, Precision, Recall, F1-Score, and AUC
    - Detailed metrics table
    
    ###  Tips for Best Results
    
    -  **Accurate Data**: Ensure all measurements are accurate and recent
    -  **Complete Information**: Fill in all fields - missing data can affect accuracy
    -  **Multiple Techniques**: Try different sampling techniques to compare results
    -  **Understand Context**: Use the Explainable AI section to understand the reasoning
    -  **Consult Professionals**: Always discuss results with healthcare providers
    
    ###  Quick Tips
    
    - **Hover over inputs** for additional information and guidance
    - **Use the sidebar** to quickly switch between prediction types
    - **Export results** by taking screenshots of the prediction and explanation charts
    - **Compare techniques** by running predictions with different sampling methods
    
    ###  Common Questions
    
    **Q: How accurate is this model?**  
    A: Check the Model Analysis tab for detailed performance metrics. Accuracy varies by sampling technique.
    
    **Q: What do the SHAP values mean?**  
    A: SHAP values show how much each feature pushed the prediction higher (red) or lower (blue) compared to the average prediction.
    
    **Q: Should I trust this diagnosis?**  
    A: This is a research/educational tool. Always consult qualified healthcare professionals for medical decisions.
    
    **Q: Can I save my results?**  
    A: Currently, you can screenshot the results. A download feature may be added in future versions.
    """)


# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit and PyTorch | © 2025</p>
        <p>Contact: nafisa21@iut-dhaka.edu </p>
     
    </div>
""", unsafe_allow_html=True)