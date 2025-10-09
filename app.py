import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px

# Load translations from JSON files
def load_translations(language):
    with open(f"json/{language}.json", 'r') as file:
        return json.load(file)

# Initialize session state if not already set
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = 'en'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions(language):
    with open(f"data/welcome_survey_cluster_names_and_descriptions_v2_{language}.json", "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(get_model(), data=all_df)
    return df_with_clusters

# Load translations based on the selected language
translations = load_translations(st.session_state.selected_language)

# Language switch button
if st.button(translations["switch_language"]):
    if st.session_state.selected_language == 'en':
        st.session_state.selected_language = 'pl'
    else:
        st.session_state.selected_language = 'en'

    # Clear cache when language is switched to refresh the translations
    st.cache_data.clear()

# Update translations after switching languages
translations = load_translations(st.session_state.selected_language)


MODEL_NAME = f'data/welcome_survey_clustering_pipeline_v2_{st.session_state.selected_language}'
DATA = f'data/welcome_survey_simple_v2_{st.session_state.selected_language}.csv'

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions(st.session_state.selected_language)
# cluster_names_and_descriptions = get_cluster_names_and_descriptions(st.session_state.selected_language)
with st.sidebar:
    st.header(translations["tell_us_about_yourself"])
    st.markdown(translations["we_ll_help_you_find_people_with_similar_interests"])
    
    age_options = ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', translations["unknown"]]
    edu_level_options = ['Primary', 'Secondary', 'Higher']
    fav_animals_options = ['None', 'Dogs', 'Cats', 'Other', translations["both_cats_and_dogs"]]
    fav_place_options = [translations["by_the_water"], translations["in_the_forest"], translations["in_the_mountains"], translations["other"]]
    gender_options = [translations["male"], translations["female"]]

    age = st.selectbox(translations["age"], age_options)
    edu_level = st.selectbox(translations["education_level"], edu_level_options)
    fav_animals = st.selectbox(translations["favorite_animals"], fav_animals_options)
    fav_place = st.selectbox(translations["favorite_place"], fav_place_options)
    gender = st.radio(translations["gender"], gender_options)

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[str(predicted_cluster_id)]

st.header(f"{translations['you_are_closest_to_the']} {predicted_cluster_data['name']} {translations['group']}")
st.markdown(predicted_cluster_data['description'])

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]

# Summary table
summary_stats = {
    translations["total_participants"]: len(all_df),
    translations["participants_in_your_group"]: len(same_cluster_df)
}
st.table(pd.DataFrame(list(summary_stats.items()), columns=["Statistic", "Value"]))

st.header(translations["people_in_your_group"])

# Pie charts for categorical distributions
fig_age_pie = px.pie(
    same_cluster_df, 
    names="age",
    title=f"{translations['age_distribution']} {translations['in_your_group']}"
)
st.plotly_chart(fig_age_pie)

fig_edu_level_pie = px.pie(
    same_cluster_df, 
    names="edu_level",
    title=f"{translations['education_level_distribution']} {translations['in_your_group']}"
)
st.plotly_chart(fig_edu_level_pie)

fig_fav_animals_pie = px.pie(
    same_cluster_df, 
    names="fav_animals",
    title=f"{translations['favorite_animals_distribution']} {translations['in_your_group']}"
)
st.plotly_chart(fig_fav_animals_pie)

fig_fav_place_pie = px.pie(
    same_cluster_df, 
    names="fav_place",
    title=f"{translations['favorite_place_distribution']} {translations['in_your_group']}"
)
st.plotly_chart(fig_fav_place_pie)

fig_gender_pie = px.pie(
    same_cluster_df, 
    names="gender",
    title=f"{translations['gender_distribution']} {translations['in_your_group']}"
)
st.plotly_chart(fig_gender_pie)
