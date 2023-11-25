import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

st.title("Penguin Classifier")
st.write("""This app uses 6 inputs to predict the species of penguin using 
         a model built on the Palmer Penguins dataset.
         Use the form below to get started!""")

penguin_df = pd.read_csv("penguins.csv")
rf_pickle = open("penguin-rf-model.pkl", 'rb')
map_pickle = open("penguin-output.pkl", 'rb')
rfc = pickle.load(rf_pickle) # random forest classifier model
unique_penguin_map = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()


with st.form("user_inputs"):
    island = st.selectbox("Penguin Island", options=['Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox("Sex", options=['MALE', 'FEMALE'])
    bill_length = st.number_input("Bill length (mm)", min_value=0)
    bill_depth = st.number_input("Bill depth (mm)", min_value=0)
    flipper_length = st.number_input("Flipper length (mm)", min_value=0)
    body_mass = st.number_input("Body mass (g)", min_value=0)
    st.form_submit_button()
    
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1
    
sex_male, sex_female = 0, 0
if sex == 'MALE':
    sex_male = 1
elif sex == 'FEMALE':
    sex_female = 1

new_prediction = rfc.predict(
    [
        [
            bill_length,
            bill_depth,
            flipper_length,
            body_mass,
            island_biscoe,
            island_dream,
            island_torgerson,
            sex_female,
            sex_male
        ]
    ]
)

prediction_species = unique_penguin_map[new_prediction][0]

st.subheader("Predicting Your Penguin's Species")
st.write(f'We predict your penguin is of the {prediction_species} species!')
st.write(
    """
    We used a machine learning (Random Forest) model to predict the species, the features used in this
    prediction are ranked by relative importance below.
    """
)
st.image("feature_importance.png")

st.write(
    """Below are the histograms for each
    continous variable separated by penguin species.
    The vertical line represents the inputted value.
    """
)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df.bill_length_mm,
                 hue=penguin_df.species,)
plt.axvline(x=bill_length, color='red')
plt.title("Bill Length by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df.flipper_length_mm,
                 hue=penguin_df.species,)
plt.axvline(x=flipper_length, color='red')
plt.title("Flipper Length by Species")
st.pyplot(ax)