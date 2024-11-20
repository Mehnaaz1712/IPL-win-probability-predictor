import pandas as pd
import streamlit as st
import pickle

# Load the model
try:
    pipe = pickle.load(open('pipe2.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found! Please ensure 'pipe2.pkl' is in the correct directory.")
    st.stop()

# Teams and cities
teams = ['Sunrisers Hyderabad', 'Royal Challengers Bengaluru', 'Mumbai Indians', 
         'Punjab Kings', 'Rajasthan Royals', 'Lucknow Super Giants', 
         'Kolkata Knight Riders', 'Gujarat Titans', 'Chennai Super Kings', 'Delhi Capitals']

cities = ['Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur',
          'Hyderabad', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban', 
          'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein', 
          'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 
          'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Rajkot', 'Kanpur', 
          'Bengaluru', 'Indore', 'Dubai', 'Sharjah', 'Navi Mumbai', 'Lucknow', 
          'Guwahati', 'Mohali']

# App title
st.title('🏏 IPL Win Predictor')
st.markdown("Use this app to predict the probability of an IPL team's victory based on the current match situation!")

# Add background color and style
st.markdown(
    """
    <style>
        body {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Input fields
st.sidebar.header("Match Details")
batting_team = st.sidebar.selectbox('🏏 Select the batting team', sorted(teams))
bowling_team = st.sidebar.selectbox('🎯 Select the bowling team', sorted(teams))

if batting_team == bowling_team:
    st.sidebar.error("Batting and bowling teams cannot be the same.")
    st.stop()

selected_city = st.sidebar.selectbox('📍 Select host city', sorted(cities))
target = st.sidebar.number_input('🎯 Target', min_value=1, max_value=500, step=1)

col1, col2 = st.columns(2)
with col1:
    score = st.number_input('📊 Current Score', min_value=0, max_value=int(target), step=1)
with col2:
    overs = st.number_input('⏱️ Overs completed', min_value=0.0, max_value=20.0, step=0.1)

wickets = st.slider('⚡ Wickets out', min_value=0, max_value=10, step=1)

# Prediction button
if st.button('🏁 Predict Probability'):
    # Validation checks
    if overs == 0:
        st.error("Overs completed cannot be zero when making predictions.")
    elif score >= target:
        st.error("Current score cannot exceed or equal the target. Match already won!")
    elif wickets >= 10:
        st.error("All wickets are out; prediction not possible.")
    else:
        # Calculate derived metrics
        runs_left = target - score
        balls_left = int(120 - (overs* 6))
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Prepare input DataFrame
        input_df = pd.DataFrame({'batting_team': [batting_team],
                                 'bowling_team': [bowling_team],
                                 'city': [selected_city],
                                 'runs_left': [runs_left],
                                 'balls_left': [balls_left],
                                 'wickets_left': [wickets_left],
                                 'target_runs': [target],
                                 'crr': [crr],
                                 'rrr': [rrr]})

        # Prediction
        try:
            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]

            # Display results
            st.markdown("### 🔮 Prediction Results")
            st.success(f"**{batting_team} Win Probability:** {round(win * 100)}%")
            st.warning(f"**{bowling_team} Win Probability:** {round(loss * 100)}%")

            # Display probabilities with progress bars
            st.progress(int(win * 100))
            st.progress(int(loss * 100))

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
