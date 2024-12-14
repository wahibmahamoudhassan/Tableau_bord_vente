import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import cufflinks as cf
from PIL import Image
from streamlit_option_menu import option_menu
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from wordcloud import WordCloud
from unidecode import unidecode
import re
from nltk.stem import SnowballStemmer
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configuration de la barre latérale
with st.sidebar:
    selection = option_menu(
        "Menu",
        ["Contexte du projet", "Étude du jeu de données", "Dashboard de vente", "Text Mining", "Machine Learning"],
        icons=["house", "graph-up", "bar-chart", "chart-text", "robot"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )

# Fonction pour la page de contexte du projet
def project_context():
    st.title("Contexte du projet")
    # Centrer l'image avec st.image()
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
         st.image("logo_sorbonne.jpeg", width=250)
    with col2:
        st.image("logo_data.jpg", width=250)
    with col3:
        st.image("Tableau_bord.png", width=310)

    st.markdown("""
    Cette application vise à analyser un jeu de données de vente, à prétraiter les données, à créer un dashboard de vente,
    à effectuer du text mining et à appliquer des techniques de machine learning pour la régression.

    **Objectifs :**
    - Analyser les tendances des ventes.
    - Identifier les segments de clientèle les plus rentables.
    - Effectuer une analyse textuelle pour extraire des informations clés.
    - Appliquer des modèles de machine learning pour améliorer les prévisions des ventes.
    """)
    st.markdown("[Source des données](https://www.kaggle.com/datasets/abhishekrp1517/sales-data-for-economic-data-analysis/data)")
    voir_contact = st.checkbox("Réalisé par :")
    if voir_contact:
        st.markdown("""
                    - Wahib MAHMOUD HASSAN
                    - Abdourahman KARIEH DINI
                    - Mamoudou Kourdiou  DIALLO
                    """)

# Fonction pour la page d'étude du jeu de données
def data_study():
    st.title("Étude du jeu de données")
    voir_description = st.checkbox("**Description des colonnes de l'ensemble de données**")
    if voir_description:
        st.markdown("""
                    L'ensemble de données contient des informations sur les transactions de vente, incluant des variables démographiques,
    des données produits et des chiffres financiers. Voici une liste des colonnes disponibles :

    - **Year** : Année de la transaction.
    - **Month** : Mois de la transaction.
    - **Customer Age** : Âge du client au moment de la transaction.
    - **Customer Gender** : Sexe du client.
    - **Country** : Pays où la transaction a eu lieu.
    - **State** : État spécifique.
    - **Product Category** : Grande catégorie du produit.
    - **Sub Category** : Sous-catégorie précise.
    - **Quantity** : Quantité de produits vendus.
    - **Unit Cost** : Coût de production ou d'acquisition par unité.
    - **Unit Price** : Prix de vente par unité.
    - **Cost** : Coût total des produits vendus.
    - **Revenue** : Chiffre d'affaires total.
    """)

    charge_donnee = st.file_uploader("Choisissez un fichier CSV", type="csv")

    if charge_donnee is not None:
        try:
            if charge_donnee.name.endswith("csv"):
                data = pd.read_csv(charge_donnee, delimiter=";")
            elif charge_donnee.name.endswith("txt"):
                data = pd.read_csv(charge_donnee, delimiter="\t")
            elif charge_donnee.name.endswith(("xlsx", "xls")):
                data = pd.read_excel(charge_donnee)
            elif charge_donnee.name.endswith("json"):
                data = pd.read_json(charge_donnee)
            else:
                st.error("Type de fichier non pris en charge.")
                return
            st.success("Fichier chargé avec succès !")

            tabs = st.tabs(["Aperçu de données", "Prétraitement des données", "Statistiques descriptives"])

            with tabs[0]:
                st.dataframe(data.head())
                st.write(f"Nombre de lignes : {data.shape[0]}")
                st.write(f"Nombre de colonnes : {data.shape[1]}")
                st.write(f"Types de colonnes :")
                st.write(data.dtypes.value_counts())

                quant_vars = ['Year', 'Customer Age', 'Quantity', 'Unit Cost', 'Unit Price', 'Cost', 'Revenue']
                qual_vars = ['Date', 'Month', 'Customer Gender', 'Country', 'State', 'Product Category', 'Sub Category']

                st.write(f"**Variables quantitatives :** {len(quant_vars)}")
                st.write(", ".join(quant_vars))
                st.write(f"**Variables qualitatives :** {len(qual_vars)}")
                st.write(", ".join(qual_vars))

            with tabs[1]:
                # Effectuer le bootstrap pour atteindre 300,000 lignes
                n_lignes = 300000  # Nombre de lignes souhaitées
                base_bootstrap = data.sample(n=n_lignes, replace=True, random_state=42)

                # Enregistrer la base résultante dans un nouveau fichier CSV
                fichier_sortie = "base_bootstrap.csv"
                base_bootstrap.to_csv(fichier_sortie, index=False)
                st.markdown("""Nous avons utilisé la technique du bootstrap pour agrandir notre base de données
                             initiale tout en conservant ses caractéristiques d'origine.
                             Cette méthode consiste à reproduire nos enregistrements plusieurs fois.""")
                st.write(f"La dimension des nouvelles données est : **{base_bootstrap.shape}**")
                st.success(f"Fichier bootstrap enregistré avec succès : {fichier_sortie}")
                st.markdown("[Boostrap](https://gsalvatovallverdu.gitlab.io/post/2011-09-16-schema-expliquant-le-princpe-du-bootstrap/)")
                # Continuer avec les autres étapes de prétraitement
                if 'Date' in base_bootstrap.columns:
                    base_bootstrap['Date'] = pd.to_datetime(base_bootstrap["Date"], errors="coerce")

                base_bootstrap['Year'] = base_bootstrap['Year'].fillna(0).astype(int)

                mois_mapping = {
                    "january": 1, "february": 2, "march": 3, "april": 4,
                    "may": 5, "june": 6, "july": 7, "august": 8,
                    "september": 9, "october": 10, "november": 11, "december": 12
                }
                base_bootstrap['Month'] = base_bootstrap['Month'].astype(str).str.lower().map(mois_mapping).fillna(0).astype(int)

                # Nettoyage des valeurs incorrectes dans la colonne "Month"
                base_bootstrap['Month'] = base_bootstrap['Month'].apply(lambda x: x if 1 <= x <= 12 else np.nan)
                base_bootstrap.dropna(subset=['Month'], inplace=True)

                base_bootstrap['Semestre'] = base_bootstrap['Date'].apply(lambda x: 1 if x.month <= 7 else 2)

                base_bootstrap.drop(columns=["Column1", "index"], inplace=True, errors="ignore")

                base_bootstrap.rename(columns={'Cost': 'Cout_tot', 'Revenue': 'Chiffre_affaires'}, inplace=True)

                base_bootstrap['Benefice'] = base_bootstrap['Chiffre_affaires'] - base_bootstrap['Cout_tot']

                st.subheader("Résumé des étapes de prétraitement")
                resume_pretraitement = {
                    "Description": [
                        "Colonne 'Date' convertie au format datetime.",
                        "Noms des mois convertis en valeurs numériques.",
                        "Ajout d'une colonne indiquant le semestre de la vente.",
                        "Colonnes inutiles supprimées pour simplifier l'analyse.",
                        "Colonnes renommées pour plus de clarté.",
                        "Création d'une nouvelle colonne calculant le bénéfice."
                    ],
                    "Code utilisé": [
                        "`data['Date'] = pd.to_datetime(data['Date'], errors='coerce')`",
                        "`data['Month'] = data['Month'].astype(str).str.lower().map(mois_mapping)`",
                        "`data['Semestre'] = data['Date'].apply(lambda x: 1 if x.month <= 7 else 2)`",
                        "`data.drop(columns=['Column1'], inplace=True, errors='ignore')`",
                        "`data.rename(columns={'Cost': 'Cout_tot', 'Revenue': 'Chiffre_affaires'})`",
                        "`data['Benefice'] = data['Chiffre_affaires'] - data['Cout_tot']`"
                    ]
                }
                resume_df = pd.DataFrame(resume_pretraitement)
                st.table(resume_df)
                st.success("Données prétraitées avec succès !")

                quant_vars = base_bootstrap.select_dtypes(include=['int64', 'float64']).columns.tolist()
                qual_vars = base_bootstrap.select_dtypes(include=['object']).columns.tolist()

                st.write(f"**Variables quantitatives :** {len(quant_vars)}")
                st.write(", ".join(quant_vars))
                st.write(f"**Variables qualitatives :** {len(qual_vars)}")
                st.write(", ".join(qual_vars))

                # Sauvegarder les données prétraitées pour le tableau de bord
                st.session_state.preprocessed_data = base_bootstrap

            with tabs[2]:
                st.header("Statistiques descriptives")

                var_type = st.selectbox("Choisissez le type de variable", ["Quantitatives", "Qualitatives"])

                if var_type == "Quantitatives":
                    selected_quant_vars = st.multiselect(
                        "Choisissez les variables quantitatives",
                        quant_vars
                    )

                    quant_vis_options = st.selectbox(
                        "Choisissez le type de visualisation pour les variables quantitatives",
                        ["Histogramme", "Box Plot", "Scatter Plot", "Line Plot"]
                    )

                    if selected_quant_vars and quant_vis_options:
                        st.header("Visualisations des variables quantitatives")
                        for var in selected_quant_vars:
                            if quant_vis_options == "Histogramme":
                                st.subheader(f"Histogramme de {var}")
                                fig = px.histogram(base_bootstrap, x=var, title=f"Histogramme de {var}")
                                st.plotly_chart(fig)

                            elif quant_vis_options == "Box Plot":
                                st.subheader(f"Box Plot de {var}")
                                fig = px.box(base_bootstrap, y=var, title=f"Box Plot de {var}")
                                st.plotly_chart(fig)

                            elif quant_vis_options == "Scatter Plot":
                                st.subheader(f"Scatter Plot de {var}")
                                fig = px.scatter(base_bootstrap, x=base_bootstrap.index, y=var, title=f"Scatter Plot de {var}")
                                st.plotly_chart(fig)

                            elif quant_vis_options == "Line Plot":
                                st.subheader(f"Line Plot de {var}")
                                fig = px.line(base_bootstrap, x=base_bootstrap.index, y=var, title=f"Line Plot de {var}")
                                st.plotly_chart(fig)

                elif var_type == "Qualitatives":
                    selected_qual_vars = st.multiselect(
                        "Choisissez les variables qualitatives",
                        qual_vars
                    )

                    qual_vis_options = st.selectbox(
                        "Choisissez le type de visualisation pour les variables qualitatives",
                        ["Bar Chart", "Pie Chart", "Word Cloud"]
                    )

                    if selected_qual_vars and qual_vis_options:
                        st.header("Visualisations des variables qualitatives")
                        for var in selected_qual_vars:
                            if qual_vis_options == "Bar Chart":
                                st.subheader(f"Bar Chart de {var}")
                                fig = px.bar(base_bootstrap[var].value_counts().sort_values(ascending=False), title=f"Bar Chart de {var}")
                                st.plotly_chart(fig)

                            elif qual_vis_options == "Pie Chart":
                                st.subheader(f"Pie Chart de {var}")
                                fig = px.pie(base_bootstrap, names=var, title=f"Pie Chart de {var}")
                                st.plotly_chart(fig)

                            elif qual_vis_options == "Word Cloud":
                                st.subheader(f"Word Cloud de {var}")
                                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(base_bootstrap[var].astype(str)))
                                plt.figure(figsize=(10, 5))
                                plt.imshow(wordcloud, interpolation='bilinear')
                                plt.axis('off')
                                st.pyplot(plt)
                voir_stats_descriptives = st.checkbox("**Résumé des statistiques descriptives des variables quantitatives**")
                if voir_stats_descriptives:
                      stats = base_bootstrap[['Customer Age', 'Cout_tot', 'Chiffre_affaires', 'Benefice']].describe()
                      stats_transposed = stats.T
                      st.write(stats_transposed)
                voir_corr_df = st.checkbox("**Heatmap des variables numériques**")
                if voir_corr_df:
                    corr_df = base_bootstrap[['Customer Age', 'Cout_tot', 'Chiffre_affaires', 'Benefice']]
                    corr_matrix = corr_df.corr()
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=0.5)
                    st.pyplot(plt)

        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")

def tableau_bord_vente():
    st.title("Dashboard de vente 📊")

    if 'preprocessed_data' in st.session_state:
        data = st.session_state.preprocessed_data

        st.sidebar.header("Filtres")
        pays = st.sidebar.multiselect("Filtrer par pays", options=data["Country"].unique(), default=data["Country"].unique())
        categories = st.sidebar.multiselect("Filtrer par catégorie de produit", options=data["Product Category"].unique(), default=data["Product Category"].unique())

        data_filtre = data[(data["Country"].isin(pays)) & (data["Product Category"].isin(categories))]

        # KPI principaux
        st.subheader("Indicateurs clés de performance (KPI)")
        st.markdown("""
            <style>
            .kpi-box {
                border: 2px solid #007BFF;
                border-radius: 10px;
                padding: 10px;
                margin: 10px;
                text-align: center;
                background-color: #f0f8ff;
            }
            .kpi-box h3 {
                color: #007BFF;
            }
            .kpi-box p {
                font-size: 15px;
                font-weight: bold;
                color: #007BFF;
            }
            </style>
        """, unsafe_allow_html=True)

        total_ventes = data_filtre["Chiffre_affaires"].sum()
        total_benefice = data_filtre["Benefice"].sum()
        nombre_transactions = len(data_filtre)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f'<div class="kpi-box"><h3>Chiffre d\'affaires  (€)</h3><p>{total_ventes:,.2f}</p></div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="kpi-box"><h3>Bénéfice  (€)</h3><p>{total_benefice:,.2f}</p></div>', unsafe_allow_html=True)

        with col3:
            st.markdown(f'<div class="kpi-box"><h3>Transactions </h3><p>{nombre_transactions}</p></div>', unsafe_allow_html=True)

        # Tendances mensuelles des ventes
        st.subheader("Tendances mensuelles des ventes")
        data_filtre["Month"] = pd.to_datetime(data_filtre["Month"], format="%m").dt.strftime("%B")
        data_filtre["Month"] = pd.Categorical(data_filtre["Month"], categories=[
            'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'
        ], ordered=True)
        repartir_par = st.radio(" ", key="visibility", options=["Chiffre_affaires", "Benefice"])
        ventes_par_mois = data_filtre.groupby("Month")[repartir_par].sum().reset_index()
        fig1 = px.bar(
         ventes_par_mois,
         x="Month",
         y=repartir_par,
         title=f"{repartir_par} par mois",
         labels={"Chiffre_affaires": "Chiffre d'affaires (€)",
            "Benefice": "Bénéfice (€)",
            "Month": "Mois"},
         color=repartir_par,
        color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig1)

        # Analyse de la Segmentation Client
        st.subheader("Analyse de la segmentation Client")
        segmentation_age = data_filtre.groupby('Customer Age')['Chiffre_affaires'].sum().reset_index()
        segmentation_age = segmentation_age.sort_values(by="Chiffre_affaires", ascending=False)
        fig_segmentation_age = px.bar(
            segmentation_age,
            x='Customer Age',
            y='Chiffre_affaires',
            title='Chiffre d\'affaires par âge des clients',
            labels={'Chiffre_affaires': 'Chiffre d\'affaires (€)', 'Customer Age': 'Âge des clients'},
            color='Chiffre_affaires',
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_segmentation_age)

        segmentation_gender = data_filtre.groupby('Customer Gender')['Chiffre_affaires'].sum().reset_index()
        segmentation_gender = segmentation_gender.sort_values(by="Chiffre_affaires", ascending=False)
        fig_segmentation_gender = px.pie(
            segmentation_gender,
            names='Customer Gender',
            values='Chiffre_affaires',
            title='Chiffre d\'affaires par genre des clients',
            labels={'Chiffre_affaires': 'Chiffre d\'affaires (€)', 'Customer Gender': 'Genre des clients'}
        )
        st.plotly_chart(fig_segmentation_gender)

        # Analyse de la Fidélité Client
        st.subheader("Analyse de la fidélité Client")
        fidelite_client = data_filtre.groupby('Customer Age')['Quantity'].sum().reset_index()
        fidelite_client = fidelite_client.sort_values(by="Quantity", ascending=False)
        fig_fidelite_client = px.scatter(
            fidelite_client,
            x='Customer Age',
            y='Quantity',
            title='Fidélité des clients par âge',
            labels={'Quantity': 'Quantité totale vendue', 'Customer Age': 'Âge des clients'},
            color='Quantity',
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_fidelite_client)

        # Analyse des Produits
        st.subheader("Analyse des Produits")
        produits_vendus = data_filtre.groupby('Sub Category')['Quantity'].sum().reset_index()
        produits_vendus = produits_vendus.sort_values(by="Quantity", ascending=False)
        fig_produits_vendus = px.bar(
            produits_vendus,
            x='Sub Category',
            y='Quantity',
            title='Quantité vendue par sous-catégorie de produit',
            labels={'Quantity': 'Quantité vendue', 'Sub Category': 'Sous catégorie de produit'},
            color='Quantity',
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_produits_vendus)

        # Répartition des ventes par catégorie de produit
        repartir_par2 = st.radio("choix ", key="invisibility", options=["Chiffre_affaires", "Benefice"])
        produits_benefice = data_filtre.groupby('Sub Category')[repartir_par2].sum().reset_index()
        produits_benefice = produits_benefice.sort_values(by=repartir_par2, ascending=False)

        fig_produits_benefice = px.bar(
            produits_benefice,
            x='Sub Category',
            y=repartir_par2,
            title= f"{repartir_par2} par sous catégorie de produit",
            labels={"Chiffre_affaires": "Chiffre d'affaires (€)",
            "Benefice": "Bénéfice (€)",  'Sub Category': 'Sous catégorie de produit'},
            color=repartir_par2,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_produits_benefice)

        # Analyse Géographique
        st.subheader("Analyse Géographique")
        ventes_par_pays = data_filtre.groupby("State")["Chiffre_affaires"].sum().reset_index()
        ventes_par_pays = ventes_par_pays.sort_values(by="Chiffre_affaires",ascending = False)
        fig5 = px.bar(
            ventes_par_pays,
            x="State",
            y="Chiffre_affaires",
            title="Chiffre d'affaires par région",
            labels={"Chiffre_affaires": "Chiffre d'affaires (€)", "State": "Régions"},
            color="Chiffre_affaires",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig5)

        # Bénéfices par Mois et par Pays
        st.subheader("Bénéfices par mois et par Pays")
        benefits_by_month_country = data_filtre.groupby(['Month', 'Country'])['Benefice'].sum().reset_index()
        benefits_by_month_country['Month'] = pd.Categorical(benefits_by_month_country['Month'], categories=[
            'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'
        ], ordered=True)
        benefits_by_month_country = benefits_by_month_country.sort_values(by=['Month', 'Country'])

        fig = px.line(
            benefits_by_month_country,
            x='Month',
            y='Benefice',
            color='Country',
            title='Bénéfices par Mois et par Pays',
            labels={'Benefice': 'Bénéfices', 'Month': 'Mois', 'Country': 'Pays'},
            line_shape='linear',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            xaxis=dict(title='Mois', tickmode='array', tickvals=benefits_by_month_country['Month'].cat.categories),
            yaxis=dict(title='Bénéfices'),
            legend_title='Pays',
            template='plotly_white',
            height=600,
            width=900
        )
        st.plotly_chart(fig)

        # Évolution des chiffres d'affaires et des coûts par mois
        st.subheader("Évolution des chiffres d'affaires et des coûts par mois")
        df_grouped = data_filtre.groupby(['Month'])[['Chiffre_affaires', 'Cout_tot']].sum().reset_index()

        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=["Évolution des chiffres d'affaires et des coûts par mois"]
        )

        fig.add_trace(
            go.Scatter(
                x=df_grouped['Month'],
                y=df_grouped['Chiffre_affaires'],
                mode='lines',
                name='Chiffre d\'affaires',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df_grouped['Month'],
                y=df_grouped['Cout_tot'],
                mode='lines',
                name='Coût total',
                line=dict(color='red')
            ),
            row=1, col=1
        )

        fig.update_layout(
            title='Évolution des chiffres d\'affaires et des coûts par mois',
            xaxis_title='Mois',
            yaxis_title='Valeur',
            height=500,
            width=1000,
            showlegend=True
        )

        st.plotly_chart(fig)

        st.info("Veuillez d'abord prétraiter les données dans la section 'Étude du jeu de données'.")
def text_mining():
    st.title("Text Mining")

    # La source du texte :
    st.markdown("[Decathlon : Comment le vendeur d'articles de sport parvient à se distinguer de ses concurrents](https://www.lectra.com/fr/librairie/decathlon-comment-le-vendeur-darticles-de-sport-parvient-a-se-distinguer-de-ses)")

    st.markdown("#### Les points qui seront abordés :")
    st.markdown("- **Assortiments**")
    st.markdown("- **Le vêtement de sport est-il plus une affaire d’hommes qu’une affaire de femmes ?**")
    st.markdown("- **Rendre les meilleurs produits de sport accessibles à tous.**")
    st.markdown("- **L’expansion se poursuit**")

    texte = [
        "Assortiments ",
        """Contrairement à ses concurrents, Decathlon propose uniquement des vêtements de
        sport et aucun produit « lifestyle ». Il s’agit d’ailleurs sans doute d’une stratégie de
        la part de Decathlon de se concentrer uniquement sur le segment du vêtement de sport.
        Les concurrents de Decathlon, Adidas et Nike, investissent quant à eux de plus en plus
        dans le segment lifestyle ou activewear.

        Les assortiments de Decathlon reflètent le nombre élevé de sports différents présents dans
        le portefeuille de la marque. Chaque sport a ses particularités et correspond donc à des
        vêtements spécifiques. Nike et Adidas restent en revanche fidèles à leur modèle économique
        axé sur la chaussure de sport. La chaussure constitue la principale source de revenus de ces
        deux marques.
        """,
        "Le vêtement de sport est-il plus une affaire d’hommes qu’une affaire de femmes ?",
        """Selon les données recueillies par Retviews, on constate que chez Decathlon et Nike,
        la catégorie Homme occupe presque 60 % de l’assortiment (vêtements techniques et
        activewear combinés). C’est tout le contraire de l’assortiment de Zara.

        On observe néanmoins qu’Adidas tend à atteindre la parité dans le nombre de modèles
        réservés aux deux sexes. Serait-ce le premier pas vers l’égalité des sexes dans l’univers
        du sport ? """,
        """En matière de vêtements de sport, les t-shirts sont les articles les plus vendus, après
        les baskets. De manière générale, Decathlon propose des prix bien plus abordables que ses
        concurrents. « Rendre les meilleurs produits de sport accessibles à tous » fait partie intégrante
        de l’identité de la marque.
        """,
        "Rendre les meilleurs produits de sport accessibles à tous.",
        """En matière de stratégie tarifaire, on observe une différence entre la marque française et ses concurrents.
        On constate en effet que leurs prix d’entrée sont très différents. Chez Decathlon, le prix des t-shirts
        pour hommes et pour femmes commence à 2,99 €. À titre de comparaison, le prix d’entrée d’Adidas est de 8,95 €
        et celui de Nike de 16,99 €.

        De même, le prix de vente le plus courant chez Decathlon est inférieur de 20 € à celui de ses concurrents.
        La différence en matière de prix maximum est également particulièrement frappante. Adidas et Nike sont
        célèbres pour leurs produits lifestyle, le prix de ces produits pouvant atteindre 399,99 € pour la nouvelle
        collection créée par Nike et appelée Nike ESC, et 299,95 € pour Adidas. Par comparaison, les maillots d’équipes
        de football nationales à 89,99 € de Decathlon semblent presque abordables.

        Retviews utilise un outil basé sur l’intelligence artificielle qui lui permet d’identifier les articles similaires.
        Dans le cadre de la comparaison tarifaire que nous avons effectuée, nous avons découvert que le prix d’un pantalon
        de jogging basique en molleton, brossé ou non, était de 39,99 € chez Nike et de 39,95 chez Adidas. Chez Decathlon,
        en revanche, des produits semblables sont proposés à 9,99 €.

        Le fait de proposer des vêtements de sport associant qualité et prix abordable est un élément clé de la réussite de
        Decathlon.
        """,
        "L’expansion se poursuit",
        """La marque française a fait du chemin depuis sa création en 1976 sur un petit parking par des amis passionnés de sport. 
        La stratégie de Decathlon associe stylisme, innovation et service client afin d’obtenir les meilleurs résultats possibles.
        Comme l’indique la marque sur son site Web, son approche globale permet à la marque de vêtements de sport de proposer des
        produits d’excellente qualité, à un prix honnête et raisonnable.

        L’expansion de Decathlon ne montre d’ailleurs aucun signe de ralentissement. La marque poursuit sa croissance en ouvrant
        de nouveaux magasins en Europe. Pour le moment, le chiffre d’affaires de Decathlon n’atteint que la moitié de celui d’Adidas,
        mais qui sait ce que nous réserve l’avenir. Cette entreprise familiale française risque fort de faire de l’ombre aux grands noms
        du vêtement de sport.
        """
    ]

    st.write(f"Le texte est composé de {len(texte)} documents")

    col1, col2 = st.columns(2)

    with col1:
        ### Nombre de caractères dans chaque document
        st.subheader("Nombre de caractères dans chaque document")
        st.write("Cette section affiche le nombre de caractères dans chaque document. Cela permet de comprendre la longueur de chaque texte.")
        if st.checkbox("Afficher le nombre de caractères dans chaque document", key="char_count"):
            for i, carac in enumerate(texte, start=1):
                st.write(f"Le document {i} est de longueur : {len(carac)}")

        ### Nombre de mots de chaque document
        st.subheader("Nombre de mots de chaque document")
        st.write("Cette section affiche le nombre de mots dans chaque document. Cela permet de comprendre la complexité de chaque texte.")
        if st.checkbox("Afficher le nombre de mots de chaque document", key="word_count"):
            for i, doc in enumerate(texte, start=1):
                liste_mots = doc.split()
                st.write(f"Le document {i} contient {len(liste_mots)} mots")
                if st.checkbox(f"Afficher la liste de mots du document {i}", key=f"list_mots_{i}"):
                    st.write(f"La liste de mots du document {i} : {liste_mots} \n")

        # Mise du texte en minuscule
        st.subheader("Mise du texte en minuscule")
        st.write("Cette section convertit tout le texte en minuscule pour uniformiser les données.")
        texte = [doc.lower() for doc in texte]
        if st.checkbox("Afficher le texte en minuscule", key="lower_case"):
            for i, doc in enumerate(texte, start=1):
                st.write(f"Document {i} en minuscule :\n{doc}\n")

    with col2:
        # Suppression de la ponctuation
        st.subheader("Suppression de la ponctuation")
        st.write("Cette section supprime la ponctuation du texte pour faciliter l'analyse.")
        def supprimer_ponctuation(texte):
            return texte.translate(str.maketrans('', '', string.punctuation))

        texte_sans_ponctuation = [supprimer_ponctuation(doc) for doc in texte]
        if st.checkbox("Afficher le texte sans ponctuation", key="no_punctuation"):
            for i, doc in enumerate(texte_sans_ponctuation, start=1):
                st.write(f"Document {i} sans ponctuation :\n{doc}\n")

        # Tokenisation
        st.subheader("Tokenisation")
        st.write("Cette section divise le texte en mots individuels (tokens) pour faciliter l'analyse.")
        chaine = " ".join(texte)
        mots = chaine.split()
        if st.checkbox("Afficher les mots tokenisés", key="tokenization"):
            st.write(f"Mots tokenisés : {mots}")

        # Téléchargement des stopwords
        nltk.download('stopwords')
        stop_words = set(stopwords.words('french'))

        # Suppression des stopwords
        st.subheader("Suppression des stopwords")
        st.write("Cette section supprime les mots courants (stopwords) qui n'apportent pas de valeur significative à l'analyse.")
        texte_sans_stopwords = [mot for mot in mots if mot.lower() not in stop_words]
        if st.checkbox("Afficher le texte sans stopwords", key="no_stopwords"):
            st.write(f"Texte sans stopwords : {texte_sans_stopwords}")

        # Suppression des accents
        st.subheader("Suppression des accents")
        st.write("Cette section supprime les accents des mots pour uniformiser les données.")
        texte_sans_accents = [unidecode(word) for word in texte_sans_stopwords]
        if st.checkbox("Afficher le texte sans accents", key="no_accents"):
            st.write(f"Texte sans accents : {texte_sans_accents}")

    # Analyse de fréquence
    st.subheader("Analyse de fréquence")
    st.write("Cette section analyse la fréquence des mots dans le texte pour identifier les mots les plus courants.")
    frequence = Counter(texte_sans_accents)
    mots_frequents = frequence.most_common(15)

    mots = [mot[0] for mot in mots_frequents]
    frequences = [mot[1] for mot in mots_frequents]

    # Création du graphique des barres
    st.subheader("Graphique des barres des mots les plus fréquents")
    st.write("Ce graphique montre les 15 mots les plus fréquents dans le texte.")
    if st.checkbox("Afficher le graphique des barres", key="bar_chart"):
        plt.figure(figsize=(10, 6))
        plt.barh(mots, frequences, color='skyblue')
        plt.xlabel('Fréquence')
        plt.ylabel('Mots')
        plt.title('Les 20 mots les plus fréquents')
        plt.gca().invert_yaxis()
        st.pyplot(plt)

    # Création du nuage de mots
    st.subheader("Nuage de mots des mots les plus fréquents")
    st.write("Ce nuage de mots visualise les mots les plus fréquents dans le texte.")
    if st.checkbox("Afficher le nuage de mots", key="word_cloud",label_visibility="visible"):
        dictionnaire_mots = dict(mots_frequents)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dictionnaire_mots)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Nuage de mots des 15 mots les plus fréquents')
        st.pyplot(plt)

def machine_learning():
    st.title("Machine Learning")
    st.header("Régression linéaire pour prédire le chiffre d'affaires ou le bénéfice")

    if 'preprocessed_data' in st.session_state:
        data = st.session_state.preprocessed_data

        # Description de l'étude
        st.subheader("Description de l'étude")
        st.markdown("""
        Dans cette étude, nous allons utiliser un modèle de régression linéaire pour prédire le chiffre d'affaires ou le bénéfice en fonction de plusieurs variables indépendantes.
        Les variables indépendantes incluent le pays, la catégorie de produit, l'âge du client et le sexe du client.
        """)

        # Choix des variables
        st.subheader("Choix des variables")
        st.markdown("""
        - **Variable dépendante (Y)** : Chiffre d'affaires ou Bénéfice
        - **Variables indépendantes (X)** : Pays, Catégorie de produit, Âge du client, Sexe du client
        """)

        # Sélection de la variable dépendante
        target = st.selectbox("Sélectionnez la variable dépendante (Y)", ["Chiffre_affaires", "Benefice"])

        # Sélection des variables indépendantes
        features = st.multiselect(
            "Sélectionnez les variables indépendantes (X)",
            ["Country", "Product Category", "Customer Age", "Customer Gender"],
            default=["Country", "Product Category", "Customer Age", "Customer Gender"]
        )

        # Vérification que les données ne sont pas vides
        if not features:
            st.error("Veuillez sélectionner au moins une variable indépendante.")
            return

        # Préparation des données pour le modèle
        X = pd.get_dummies(data[features], drop_first=True)
        y = data[target]

        # Division des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entraînement du modèle de régression linéaire
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)

        # Évaluation du modèle
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Résultats du modèle de régression linéaire")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R² Score: {r2:.2f}")

        # Affichage des coefficients du modèle
        st.subheader("Coefficients du modèle")
        coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
        st.write(coefficients)

        # Prédiction en temps réel
        st.subheader("Prédiction en temps réel")
        input_data = {}
        for feature in features:
            if feature in ["Country", "Product Category", "Customer Gender"]:
                input_data[feature] = st.selectbox(f"{feature}", data[feature].unique())
            elif feature == "Customer Age":
                input_data[feature] = st.number_input("Âge du client", min_value=0, max_value=100, value=30)

        # Encodage des entrées utilisateur
        input_data_encoded = pd.get_dummies(pd.DataFrame([input_data]), drop_first=True)
        input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)

        # Prédiction
        if st.button("Prédire"):
            prediction = model.predict(input_data_encoded)
            st.write(f"{target} prédit : {prediction[0]:.2f} €")

        # Courbe de prédiction en fonction des mois
        st.subheader("Prédiction en fonction des mois")
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        predictions = []
        for month in months:
            month_data = input_data.copy()
            month_data['Month'] = month
            month_data_encoded = pd.get_dummies(pd.DataFrame([month_data]), drop_first=True)
            month_data_encoded = month_data_encoded.reindex(columns=X.columns, fill_value=0)
            prediction = model.predict(month_data_encoded)
            predictions.append(prediction[0])

    else:
        st.warning("Veuillez d'abord prétraiter les données dans la section 'Étude du jeu de données'.")

if selection == "Contexte du projet":
    project_context()
elif selection == "Étude du jeu de données":
    data_study()
elif selection == "Dashboard de vente":
    tableau_bord_vente()
elif selection == "Text Mining":
    text_mining()
elif selection == "Machine Learning":
    machine_learning()
