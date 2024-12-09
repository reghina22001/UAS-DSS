import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Set page configuration
st.set_page_config(
    page_title="Mental Health DSS",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #fee6ed;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF69B4;
        color: white;
        border: none;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF1493;
    }
    .metric-card {
        background-color: #FFE4E1;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 3px 4px rgba(255,105,180,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #FFF0F5;
    }
    .stSelectbox, .stSlider {
        color: #FF69B4;
    }
    .stProgress .st-bo {
        background-color: #FFB6C1;
    }
    /* Headers styling */
    h1, h2, h3 {
        color: #FF1493;
    }
    .stAlert {
        background-color: #FFE4E1;
        color: #FF1493;
        border: 1px solid #FF69B4;
    }
    .stMetric {
        background-color: #FFE4E1;
    }
    .stMetric label {
        color: #FF1493;
    }
    /* Custom styling for plots */
    .js-plotly-plot .plotly .modebar {
        background-color: #FFF0F5;
    }
    </style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("MentalHealthSurvey.csv")
    # Ensure all required columns exist and handle missing values
    required_columns = ['age', 'gender', 'academic_year', 'cgpa', 'depression', 
                       'anxiety', 'stress', 'isolation']
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.random.randint(1, 6, size=len(df)) 
    
    df = df.fillna(df.mean(numeric_only=True))
    return df

# Load data
df = load_data()

# Prepare model and data
@st.cache_resource
def prepare_ml_components():
    # Prepare features
    features = ['age', 'depression', 'anxiety', 'stress', 'isolation']
    X = df[features].copy()
    
    # Create target variable
    df['mental_health_status'] = df.apply(
        lambda x: 'Need Support' if (x['depression'] > 3 or x['anxiety'] > 3 or x['stress'] > 3) 
        else 'Moderate' if (x['depression'] > 2 or x['anxiety'] > 2 or x['stress'] > 2)
        else 'Good', axis=1
    )
    y = df['mental_health_status']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Naive Bayes model
    nb = GaussianNB()
    nb.fit(X_scaled, y)
    
    return nb, scaler, features

# Initialize components
nb_model, scaler, ml_features = prepare_ml_components()

# Main title
st.title("🌿 Bright Minds, Balanced Lives: A Mental Health Decision System")
st.markdown('<hr style="border: 3px solid #FF69B4;">', unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.markdown('<h2 style="color: #FF1493;">Input Your Information</h2>', unsafe_allow_html=True)

# User inputs
age = st.sidebar.number_input("Age", min_value=17, max_value=30, value=20)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
academic_year = st.sidebar.selectbox("Academic Year", ["1st year", "2nd year", "3rd year", "4th year"])
cgpa = st.sidebar.selectbox("CGPA", ["0.0-0.0", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0"])

# Mental health indicators
st.sidebar.subheader("Mental Health Indicators")
depression = st.sidebar.slider("Depression Level", 1, 5, 3)
anxiety = st.sidebar.slider("Anxiety Level", 1, 5, 3)
stress = st.sidebar.slider("Academic Pressure", 1, 5, 3)
isolation = st.sidebar.slider("Isolation Level", 1, 5, 3)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h3 style="color: #FF1493;">📊 Statistical Analysis</h3>', unsafe_allow_html=True)
    
    # Calculate averages for comparison
    avg_depression = df['depression'].mean()
    avg_anxiety = df['anxiety'].mean()
    avg_isolation = df['isolation'].mean()
    
    # Create comparison metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Depression", depression, f"{depression - avg_depression:.1f} vs avg")
    with metrics_col2:
        st.metric("Anxiety", anxiety, f"{anxiety - avg_anxiety:.1f} vs avg")
    with metrics_col3:
        st.metric("Isolation", isolation, f"{isolation - avg_isolation:.1f} vs avg")

with col2:
    st.markdown('<h3 style="color: #FF1493;">📈 Visualization</h3>', unsafe_allow_html=True)
    
    # Create depression distribution plot
    fig = px.histogram(df, x='depression', 
                      title='Depression Distribution',
                      labels={'depression': 'Depression Level', 'count': 'Number of Students'})
    fig.add_vline(x=depression, line_dash="dash", line_color="#FF1493",
                  annotation_text="Your Level")
    fig.update_layout(
        plot_bgcolor='#FFF0F5',
        paper_bgcolor='#FFF0F5'
    )
    st.plotly_chart(fig, use_container_width=True)

# Analysis Section
st.markdown('<hr style="border: 3px solid #FF69B4;">', unsafe_allow_html=True)
st.markdown('<h3 style="color: #FF1493;">✍️ Analysis</h3>', unsafe_allow_html=True)

# Prepare user input
user_input = np.array([[age, depression, anxiety, stress, isolation]])
user_input_scaled = scaler.transform(user_input)

# Get Naive Bayes prediction and probabilities
nb_prediction = nb_model.predict(user_input_scaled)[0]
nb_proba = nb_model.predict_proba(user_input_scaled)[0]

# Display the results
st.markdown('<h4 style="color: #FF1493;">Naive Bayes Assessment</h4>', unsafe_allow_html=True)
st.write(f"Status: *{nb_prediction}*")
st.write("Confidence Levels:")
for label, prob in zip(nb_model.classes_, nb_proba):
    st.write(f"- {label}: {prob:.2%}")

# Recommendations based on the predictions
st.markdown('<h3 style="color: #FF1493;">💡 Recommendations</h3>', unsafe_allow_html=True)
if nb_prediction == "Need Support":
    st.warning("""
    Based on analysis:
    - Consider scheduling a consultation with a mental health professional
    - Join support groups or counseling sessions
    - Develop a stress management routine
    - Regular exercise and mindfulness practices
    """)
elif nb_prediction == "Moderate":
    st.info("""
    Recommendations:
    - Maintain regular check-ins with academic advisors
    - Practice stress-relief activities
    - Consider joining peer support groups
    - Establish a balanced study routine
    """)
else:
    st.success("""
    Keep up your current well-being practices:
    - Continue your existing support systems
    - Maintain work-life balance
    - Regular exercise and healthy habits
    - Stay connected with friends and family
    """)

# Life Balance Analysis
st.markdown('<hr style="border: 3px solid #FF69B4;">', unsafe_allow_html=True)
st.markdown('<h3 style="color: #FF1493;">❤️‍🩹 Life Balance Analysis</h3>', unsafe_allow_html=True)

# Radar chart
categories = ['Depression', 'Anxiety', 'Isolation', 'Academic Pressure']
user_values = [depression, anxiety, isolation, stress]
avg_values = [
    df['depression'].mean(),
    df['anxiety'].mean(),
    df['isolation'].mean(),
    df['stress'].mean()
]

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=user_values,
    theta=categories,
    fill='toself',
    name='Your Scores',
    line_color='#FF69B4',
    fillcolor='rgba(255,105,180,0.3)'
))
fig.add_trace(go.Scatterpolar(
    r=avg_values,
    theta=categories,
    fill='toself',
    name='Average Scores',
    line_color='#FF1493',
    fillcolor='rgba(255,20,147,0.1)'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True, 
            range=[0, 5],
            gridcolor='#FFB6C1'
        ),
        bgcolor='#FFF0F5'
    ),
    showlegend=True,
    paper_bgcolor='#FFF0F5',
    plot_bgcolor='#FFF0F5'
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown('<hr style="border: 3px solid #FF69B4;">', unsafe_allow_html=True)
st.markdown("""
    <div style="background-color: #FFE4E1; padding: 1rem; border-radius: 0.5rem; border: 1px solid #FF69B4;">
        📌 <strong>Note</strong>: This analysis is based on survey data and should not be considered as professional medical advice. 
        If you're experiencing severe mental health issues, please contact professional mental health services.
    </div>
""", unsafe_allow_html=True)