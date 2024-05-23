import streamlit as st
import pandas as pd
from joblib import load
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import base64
import seaborn as sns
import matplotlib.pyplot as plt

# Fonction pour obtenir la représentation base64 d'un fichier binaire
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Fonction pour définir une image comme fond d'écran
def set_background(png_file):
    # Obtenir la représentation base64 de l'image
    bin_str = get_base64(png_file)
    # Utiliser la représentation base64 pour définir l'image comme fond d'écran
    page_bg_img = f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: cover;
            }}
        </style>
    """
    # Appliquer le style avec l'image de fond à la page Streamlit
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Appeler la fonction pour définir l'image comme fond d'écran
set_background("background.png")

# Charger le DataFrame
chemin = Path(__file__).parent
fichier_data = chemin / "df_final.csv"
df = pd.read_csv(fichier_data, sep=',', lineterminator='\n')

# Charger les composants sauvegardés
chemin = Path(__file__).parent
tfidf_matrix = load(chemin / 'tfidf_matrix.joblib')
indices = load(chemin / "indices.joblib")
titles = load(chemin / "titles.joblib")

# Interface utilisateur Streamlit
st.markdown("<h1 style='color: red ;'>RECO-SIFFREDI</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='color: red;'>L' Outil de recommandation qui en a une grosse!!!!</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='color: red;'>(base de donnée bien évidemment petit cochonou <(°@°)>)</h3>", unsafe_allow_html=True)

st.write('\n')
st.write('\n')

# le slider pour le choix des nearestneighbors repis plus tard dans le .fit()
x = st.select_slider("Nombre de films à recommander :", options=range(1, 21))
nb = x + 1

# tri colonne genre
all_genres = ' '.join(df['genres'].str.replace(',', " ")).split()
choix_genre = list(set(all_genres))

# Widget multiselect pour filtrer les genres
genres_selectionnes = st.multiselect('Choix du genre majoritaire:', options= choix_genre)

# Vérifier si des genres ont été sélectionnés
if genres_selectionnes:
    # Obtenir les indices des films correspondant aux genres sélectionnés
    indices_genres = df[df['genres'].str.contains('|'.join(genres_selectionnes))].index
    # Filtrer la matrice TF-IDF et les titres correspondants
    tfidf_matrix_filtre = tfidf_matrix[indices_genres]
    df_filtre = df.loc[indices_genres].reset_index(drop=True)
    titles_filtre = titles[indices_genres].reset_index(drop=True)
else:
    # Utiliser la matrice TF-IDF complète si aucun genre n'est sélectionné
    tfidf_matrix_filtre = tfidf_matrix
    df_filtre = df.reset_index(drop=True)
    titles_filtre = titles.reset_index(drop=True)

#st.write(indices_genres)
#st.write(df[df['genres'].str.contains('|'.join(genres_selectionnes))])
#st.write(df_filtre['originalTitle'].str.contains('Scarface'))
#st.write(titles_filtre)

dico_bands = {name: index for name, index in zip(titles, indices)}

# Onglets
tab1, tab2, tab3 = st.tabs(["FILM DETECTOR", "INFOS GENERALES DE LA BDD", "DATAVIZ"])

with tab1:
    col1, col2 = st.columns(2)
    eureka = False

    with col1:
        st.write("Choisi un film camarade:")
        user_input = st.selectbox('', titles_filtre.sort_values(ascending=True), index=None)

        if st.button('Rechercher:'):
            user_index = dico_bands[user_input]

            # Vérifier si l'utilisateur a sélectionné des genres
            if genres_selectionnes:
                # recupérer user_index du DF filtré
                user_index_filtre = df_filtre.index[df_filtre['originalTitle'] == user_input][0]
            else:
                user_index_filtre = user_index

            st.write('Tu as choisi:')
            choix = user_input
            st.markdown(f"<p style='color: red; font-weight: bold;'>{choix}</p>", unsafe_allow_html=True)
            st.write(f"Nom : **{choix}**")
            st.write(f"genres : {df.loc[user_index, 'genres']}")
            st.write(f"Année : {df.loc[user_index, 'startYear']}")
            st.write(f"Director : {df.loc[user_index, 'directors']}")
            st.write(f"Note : {df.loc[user_index, 'averageRating']}")
            st.write(f"Synopsys : {df.loc[user_index, 'overview']}")
            st.image(f"https://image.tmdb.org/t/p/original/{df.loc[user_index, 'poster_path']}")
            st.write('\n')

            modelNN = NearestNeighbors(n_neighbors=nb)
            modelNN.fit(tfidf_matrix_filtre)
            _, indices = modelNN.kneighbors(tfidf_matrix_filtre[user_index_filtre])
            eureka = True

    with col2:
        if eureka:
            st.header("Regarde ces films:")
            for index in indices[0]:
                if index != user_index_filtre:
                    recommendation = df_filtre.loc[index]
                    st.markdown(f"<p style='color: red; font-weight: bold;'>{recommendation['originalTitle']}</p>", unsafe_allow_html=True)
                    st.write(f"Nom : **{recommendation['originalTitle']}**")
                    st.write(f"genre : {recommendation['genres']}")
                    st.write(f"Année : {recommendation['startYear']}")
                    st.write(f"Director : {recommendation['directors']}")
                    st.write(f"Note : {recommendation['averageRating']}")
                    st.write(f"Synopsys : {recommendation['overview']}")
                    st.image(f"https://image.tmdb.org/t/p/original/{recommendation['poster_path']}")
                    st.write('\n')

with tab2:
    st.header("Un peu d'informations sur la base")

