import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2_EN'
DATA = 'welcome_survey_simple_v2_EN.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2_EN.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(get_model(), data=all_df)
    return df_with_clusters

with st.sidebar:
    st.header("Tell us about yourself")
    st.markdown("We'll help you find people with similar interests.")
    
    age_options = ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown']
    edu_level_options = ['Primary', 'Secondary', 'Higher']
    fav_animals_options = ['None', 'Dogs', 'Cats', 'Other', 'Both Cats and Dogs']
    fav_place_options = ['By the Water', 'In the Forest', 'In the Mountains', 'Other']
    gender_options = ['Male', 'Female']

    age = st.selectbox("Age", age_options)
    edu_level = st.selectbox("Education Level", edu_level_options)
    fav_animals = st.selectbox("Favorite Animals", fav_animals_options)
    fav_place = st.selectbox("Favorite Place", fav_place_options)
    gender = st.radio("Gender", gender_options)

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[str(predicted_cluster_id)]

st.header(f"You are closest to the {predicted_cluster_data['name']} group")
st.markdown(predicted_cluster_data['description'])

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]

# Summary table
summary_stats = {
    "Total Participants": len(all_df),
    "Participants in Your Group": len(same_cluster_df)
}
st.table(pd.DataFrame(list(summary_stats.items()), columns=["Statistic", "Value"]))

st.header("People in Your Group")

# Pie charts for categorical distributions
fig_age_pie = px.pie(
    same_cluster_df, 
    names="age",
    title="Age Distribution in Your Group"
)
st.plotly_chart(fig_age_pie)

fig_edu_level_pie = px.pie(
    same_cluster_df, 
    names="edu_level",
    title="Education Level Distribution in Your Group"
)
st.plotly_chart(fig_edu_level_pie)

fig_fav_animals_pie = px.pie(
    same_cluster_df, 
    names="fav_animals",
    title="Favorite Animals Distribution in Your Group"
)
st.plotly_chart(fig_fav_animals_pie)

fig_fav_place_pie = px.pie(
    same_cluster_df, 
    names="fav_place",
    title="Favorite Place Distribution in Your Group"
)
st.plotly_chart(fig_fav_place_pie)

fig_gender_pie = px.pie(
    same_cluster_df, 
    names="gender",
    title="Gender Distribution in Your Group"
)
st.plotly_chart(fig_gender_pie)

# Interactive filtering (commented out for now)
# st.header("Filter Data")
# filter_by_age = st.multiselect("Age", age_options, default=age_options)
# filtered_same_cluster_df = same_cluster_df[same_cluster_df["age"].isin(filter_by_age)]

# fig_filtered_age_pie = px.pie(
#     filtered_same_cluster_df, 
#     names="age",
#     title="Age Distribution After Filter"
# )
# st.plotly_chart(fig_filtered_age_pie)

# filtered_summary_stats = {
#     "Total Participants After Filter": len(filtered_same_cluster_df),
# }
# st.table(pd.DataFrame(list(filtered_summary_stats.items()), columns=["Statistic", "Value"]))
