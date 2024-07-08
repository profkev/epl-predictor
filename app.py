import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

model = pickle.load(open('C:/Users/DELL/Desktop/dataset/trained_model.pkl', 'rb'))

cleaned_data_path = 'C:/Users/DELL/Desktop/dataset/cleaned_team_data.csv'
df = pd.read_csv(cleaned_data_path)

st.title("EPL Team Performance Predictor for season 2024/25")
st.write("Select a team and an opponent to predict the win probability, average xG, and expected possession in the upcoming matches.")

teams = df.filter(like='team_').columns.str.replace('team_', '').tolist()
opponents = df.filter(like='opponent_').columns.str.replace('opponent_', '').tolist()

selected_team = st.selectbox("Select Team", teams)
selected_opponent = st.selectbox("Select Opponent", opponents)

# Calculate historical average xG and possession against the selected opponent for the selected team
team_col = 'team_' + selected_team
opponent_col = 'opponent_' + selected_opponent

avg_xg = df[(df[team_col] == 1) & (df[opponent_col] == 1)]['xg_x'].mean()
avg_possession = df[(df[team_col] == 1) & (df[opponent_col] == 1)]['poss'].mean()

# Calculate win probability based on xG
team_xg = df[df[team_col] == 1]['xg_x'].mean()
opponent_xg = df[df[opponent_col] == 1]['xg_x'].mean()
win_probability = (team_xg / (team_xg + opponent_xg)) * 100

# Display the predicted average xG, win probability, and expected possession
if st.button('Predict'):
    st.write(f"Predicted Average xG for {selected_team} against {selected_opponent}: {avg_xg:.2f}")
    st.write(f"Win Probability for {selected_team}: {win_probability:.2f}%")
    st.write(f"Expected Possession for {selected_team}: {avg_possession:.2f}%")

    # Visualization
    st.subheader("xG and Possession Comparison")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].bar(['Team xG', 'Opponent xG'], [team_xg, opponent_xg], color=['blue', 'red'])
    ax[0].set_ylabel('xG')
    ax[0].set_title(f'xG Comparison: {selected_team} vs {selected_opponent}')

    ax[1].bar(['Team Possession'], [avg_possession], color=['green'])
    ax[1].set_ylabel('Possession (%)')
    ax[1].set_title(f'Expected Possession for {selected_team}')

    st.pyplot(fig)
