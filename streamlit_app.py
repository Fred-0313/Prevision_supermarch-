"""
Interface Streamlit — Prévision des ventes supermarché (Bénin)

Place ce fichier dans project/ui/streamlit_app.py et lance-le avec :
    streamlit run ui/streamlit_app.py
(depuis le dossier `project/`, dans un terminal séparé de celui de l'API)

L'API (api/app.py) doit tourner en parallèle sur http://localhost:8000.
Ce fichier ne charge PAS le modèle directement — il appelle l'API, exactement
le découplage dont on avait parlé (API/interface séparées).
"""

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

URL_API = "http://localhost:8000"
CHEMIN_DONNEES_BRUTES = "../data/supermarket_sales_forecasting.csv"  # ajuste si besoin

st.set_page_config(page_title="Prévision des ventes — Bénin", page_icon="🛒", layout="wide")

# ─── Valeurs par défaut si l'API ne répond pas encore (évite un formulaire vide) ──
VALEURS_PAR_DEFAUT = {
    "Store_ID": [f"S{i:03d}" for i in range(1, 11)],
    "Store_Type": ["A", "B", "C"],
    "City": ["Cotonou", "Porto-Novo", "Parakou", "Ouidah", "Abomey-Calavi",
             "Natitingou", "Kandi", "Bohicon", "Lokossa", "Djougou"],
}


@st.cache_data(ttl=60)
def recuperer_valeurs_valides():
    """Va chercher les valeurs valides directement à l'API (/stores) plutôt que de les recopier à la main —
    si tu changes un magasin côté API, le formulaire reste synchronisé."""
    try:
        reponse = requests.get(f"{URL_API}/stores", timeout=5)
        reponse.raise_for_status()
        return reponse.json()
    except requests.exceptions.RequestException:
        return VALEURS_PAR_DEFAUT


@st.cache_data
def charger_donnees_brutes():
    """Charge le dataset brut pour l'onglet EDA. Retourne None si le fichier n'est pas trouvé
    (l'onglet affiche alors le rapport en texte seul, sans planter)."""
    try:
        df = pd.read_csv(CHEMIN_DONNEES_BRUTES, parse_dates=["Date"])
        return df
    except FileNotFoundError:
        return None


SAISONS = {0: "Saison sèche principale", 1: "Petite saison des pluies",
           2: "Petite saison sèche", 3: "Grande saison des pluies"}
JOURS = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}

st.title("🛒 Prévision des ventes — Supermarchés du Bénin")

onglet_eda, onglet_donnees, onglet_prediction, onglet_insights = st.tabs(
    ["🔍 EDA (Personne A)", "📁 Données", "📈 Prédiction", "📊 Insights du modèle"]
)

# ─── Onglet EDA — reprend le rapport d'analyse exploratoire (Personne A) ──────
with onglet_eda:
    df_brut = charger_donnees_brutes()

    st.markdown("<h2 style='color:#2563EB'>Analyse exploratoire des données</h2>", unsafe_allow_html=True)
    st.caption("Jour 1 — Personne A : comprendre les données et préparer les décisions pour la modélisation.")

    if df_brut is not None:
        st.sidebar.title("— Statistiques clés (EDA) —")
        st.sidebar.markdown('<span style="font-size:20px; color:#F59E0B">◊ Variable cible : Sales</span>',
                             unsafe_allow_html=True)
        st.sidebar.text(f"Moyenne : {df_brut['Sales'].mean():,.2f}")
        st.sidebar.text(f"Médiane : {df_brut['Sales'].median():,.2f}")
        st.sidebar.text(f"Écart-type : {df_brut['Sales'].std():,.2f}")
        st.sidebar.text(f"Min : {df_brut['Sales'].min():,.2f}")
        st.sidebar.text(f"Max : {df_brut['Sales'].max():,.2f}")

    st.markdown("### Table des variables")
    table_variables = pd.DataFrame([
        {"Variable": "Date", "Type": "Date", "Rôle": "Créer les variables temporelles (mois, saison...)"},
        {"Variable": "Store_ID", "Type": "Catégorielle", "Rôle": "Niveau de ventes propre à chaque magasin"},
        {"Variable": "Store_Type", "Type": "Catégorielle", "Rôle": "Différences structurelles entre magasins"},
        {"Variable": "Store_Size", "Type": "Numérique", "Rôle": "Les grands magasins vendent souvent plus"},
        {"Variable": "City", "Type": "Catégorielle", "Rôle": "Différences démographiques/économiques locales"},
        {"Variable": "Promo", "Type": "Binaire (0/1)", "Rôle": "Facteur commercial direct"},
        {"Variable": "Holiday", "Type": "Binaire (0/1)", "Rôle": "Influence positive ou négative selon l'activité"},
        {"Variable": "School_Holiday", "Type": "Binaire (0/1)", "Rôle": "Modifie le comportement des consommateurs"},
        {"Variable": "Weekend", "Type": "Binaire (0/1)", "Rôle": "Facteur souvent important dans le commerce"},
        {"Variable": "Temperature", "Type": "Numérique", "Rôle": "Variable météo, influence possible sur les achats"},
        {"Variable": "Rainfall", "Type": "Numérique", "Rôle": "Peut modifier la fréquentation des magasins"},
        {"Variable": "Fuel_Price", "Type": "Numérique", "Rôle": "Affecte indirectement mobilité et consommation"},
        {"Variable": "Inflation_Rate", "Type": "Numérique", "Rôle": "Indicateur macroéconomique, pouvoir d'achat"},
        {"Variable": "Economic_Index", "Type": "Numérique", "Rôle": "Santé économique générale"},
        {"Variable": "Competitor_Distance", "Type": "Numérique", "Rôle": "Concurrence locale"},
        {"Variable": "Sales", "Type": "Numérique (Target)", "Rôle": "Variable cible à prédire"},
    ])
    st.dataframe(table_variables, width='stretch', hide_index=True)

    st.markdown(
        "5 500 observations, du 1er janvier 2023 au 31 décembre 2025, **aucune valeur manquante** — "
        "bonne qualité pour les analyses statistiques et la modélisation."
    )

    st.markdown("<h3 style='color:#F59E0B'>◊ Variable cible : Sales</h3>", unsafe_allow_html=True)
    col_hist, col_texte = st.columns([2, 1])
    with col_hist:
        if df_brut is not None:
            fig, ax = plt.subplots()
            sns.histplot(df_brut["Sales"], bins=30, kde=True, ax=ax, color="#2563EB")
            ax.set_title("Distribution de Sales")
            st.pyplot(fig)
        else:
            st.info(f"Place le fichier de données brutes à `{CHEMIN_DONNEES_BRUTES}` (relatif à `ui/`) "
                    "pour afficher ce graphique en direct.")
    with col_texte:
        st.markdown(
            "Moyenne de **9 889,98** (écart-type 2 588,55) — variabilité importante entre observations.\n\n"
            "- Vente minimale : 1 507,07\n"
            "- Vente maximale : 18 551,23\n"
            "- Médiane : 9 807,08\n\n"
            "La proximité moyenne/médiane suggère une distribution relativement équilibrée, "
            "à confirmer graphiquement (hypothèse de normalité)."
        )

    st.markdown("<h3 style='color:#F59E0B'>◊ Effet des promotions, jours fériés, week-end</h3>",
                unsafe_allow_html=True)
    col_promo, col_texte2 = st.columns([2, 1])
    with col_promo:
        if df_brut is not None:
            effets = pd.DataFrame({
                "Facteur": ["Sans promo", "Avec promo", "Jour normal", "Jour férié", "Semaine", "Week-end"],
                "Sales moyen": [
                    df_brut.loc[df_brut["Promo"] == 0, "Sales"].mean(),
                    df_brut.loc[df_brut["Promo"] == 1, "Sales"].mean(),
                    df_brut.loc[df_brut["Holiday"] == 0, "Sales"].mean(),
                    df_brut.loc[df_brut["Holiday"] == 1, "Sales"].mean(),
                    df_brut.loc[df_brut["Weekend"] == 0, "Sales"].mean(),
                    df_brut.loc[df_brut["Weekend"] == 1, "Sales"].mean(),
                ],
            })
            fig, ax = plt.subplots()
            sns.barplot(data=effets, x="Facteur", y="Sales moyen", hue="Facteur", ax=ax,
                        palette=["#94A3B8", "#2563EB"] * 3, legend=False)
            ax.set_title("Sales moyen selon promo / jour férié / week-end")
            plt.xticks(rotation=20, ha="right")
            st.pyplot(fig)
        else:
            st.info("Graphique disponible une fois les données brutes trouvées.")
    with col_texte2:
        st.markdown(
            "- **Promo** : +27,9% de ventes en moyenne avec promotion active.\n"
            "- **Jour férié** : gain moyen d'environ 18%.\n"
            "- **Week-end** : ventes plus élevées, confirmant un comportement d'achat renforcé le week-end.\n\n"
            "Les campagnes promotionnelles jouent un rôle majeur dans la stimulation des ventes."
        )

    st.markdown("<h3 style='color:#F59E0B'>◊ Type de magasin</h3>", unsafe_allow_html=True)
    col_type, col_texte3 = st.columns([2, 1])
    with col_type:
        if df_brut is not None:
            fig, ax = plt.subplots()
            sns.barplot(data=df_brut, x="Store_Type", y="Sales", hue="Store_Type", ax=ax, order=["A", "B", "C"],
                        palette=["#10B981", "#2563EB", "#F59E0B"], legend=False, errorbar=None)
            ax.set_title("Sales moyen par Store_Type")
            st.pyplot(fig)
        else:
            st.info("Graphique disponible une fois les données brutes trouvées.")
    with col_texte3:
        st.markdown(
            "Les magasins de **type A** enregistrent les meilleures performances, suivis des types B puis C. "
            "Cette hiérarchie suggère que Store_Type est un facteur déterminant, à intégrer dans le modèle."
        )

    st.markdown("<h3 style='color:#F59E0B'>◊ Corrélations avec Sales</h3>", unsafe_allow_html=True)
    col_corr, col_texte4 = st.columns([2, 1])
    with col_corr:
        correlations = pd.DataFrame({
            "Variable": ["Store_Size", "Competitor_Distance", "Promo", "Weekend", "day_of_week",
                         "Economic_Index", "Holiday", "Rainfall"],
            "Corrélation": [0.684, -0.475, 0.451, 0.209, 0.171, 0.156, 0.135, -0.195],
        }).sort_values("Corrélation")
        fig, ax = plt.subplots()
        couleurs = ["#EF4444" if v < 0 else "#10B981" for v in correlations["Corrélation"]]
        ax.barh(correlations["Variable"], correlations["Corrélation"], color=couleurs)
        ax.set_title("Corrélations avec Sales")
        ax.axvline(0, color="black", linewidth=0.8)
        st.pyplot(fig)
    with col_texte4:
        st.markdown(
            "- **Store_Size** (0,684) : les grands magasins vendent davantage.\n"
            "- **Competitor_Distance** (-0,475) : plus le concurrent est proche, plus les ventes baissent.\n"
            "- **Promo** (0,451) : impact favorable confirmé.\n"
            "- **Rainfall** (-0,195) : les fortes précipitations réduisent légèrement la fréquentation."
        )

    st.markdown("<h3 style='color:#F59E0B'>◊ Conclusions et recommandations (Livrable Jour 1)</h3>",
                unsafe_allow_html=True)
    col_imp, col_moy, col_faible = st.columns(3)
    with col_imp:
        st.markdown("**Variables très importantes**")
        st.markdown("- Store_Size\n- Promo\n- Competitor_Distance\n- Store_Type\n- Weekend")
    with col_moy:
        st.markdown("**Variables moyennement importantes**")
        st.markdown("- Holiday\n- School_Holiday\n- Economic_Index\n- day_of_week\n- month")
    with col_faible:
        st.markdown("**Variables probablement faibles**")
        st.markdown("- Temperature\n- Fuel_Price")

    st.markdown(
        "**Recommandations transmises à Personne B :** encoder Store_ID/Store_Type/City · "
        "conserver les variables temporelles · tester les interactions Promo×Weekend et Promo×Store_Type · "
        "utiliser un modèle non linéaire (Random Forest ou XGBoost), plusieurs effets semblant non additifs."
    )

# ─── Onglet Données — visualiser/modifier le CSV sans jamais toucher l'original ──
with onglet_donnees:
    st.markdown("<h2 style='color:#2563EB'>Données brutes</h2>", unsafe_allow_html=True)
    st.caption(f"Source : `{CHEMIN_DONNEES_BRUTES}` — l'original n'est jamais écrasé automatiquement.")

    # Copie en mémoire de session : on ne modifie jamais l'objet mis en cache par
    # charger_donnees_brutes(), sinon toute la session partagerait les mêmes
    # modifications et une réédition écraserait silencieusement le cache Streamlit.
    if "df_edite" not in st.session_state:
        donnees_originales = charger_donnees_brutes()
        st.session_state.df_edite = donnees_originales.copy() if donnees_originales is not None else None

    if st.session_state.df_edite is None:
        st.info(f"Place le fichier de données à `{CHEMIN_DONNEES_BRUTES}` (relatif à `ui/`) pour l'éditer ici.")
    else:
        col_reset, col_info = st.columns([1, 3])
        with col_reset:
            if st.button("↩️ Annuler mes modifications"):
                st.session_state.df_edite = charger_donnees_brutes().copy()
                st.rerun()
        with col_info:
            st.caption("Ajoute, modifie ou supprime des lignes directement dans le tableau. "
                       "Rien n'est écrit sur disque tant que tu ne cliques pas sur Enregistrer/Télécharger.")

        df_modifie = st.data_editor(
            st.session_state.df_edite,
            num_rows="dynamic",
            width='stretch',
            height=420,
            key="editeur_donnees",
        )
        st.session_state.df_edite = df_modifie  # reste en mémoire de session, jamais sur le fichier d'origine

        st.divider()
        col_nom, col_save, col_download = st.columns([2, 1, 1])
        nom_original = CHEMIN_DONNEES_BRUTES.rsplit("/", 1)[-1]
        with col_nom:
            nom_fichier_sortie = st.text_input(
                "Nom du fichier de sauvegarde", value="data_modifiee.csv",
                help="Choisis un nom différent du fichier d'origine — la sauvegarde est bloquée si tu gardes le même nom.",
            )
        with col_save:
            st.write("")  # alignement vertical avec le champ texte
            if st.button("💾 Enregistrer sous", type="primary"):
                if nom_fichier_sortie.strip() == nom_original:
                    st.error(f"Choisis un nom différent de `{nom_original}` pour ne pas écraser les données d'origine.")
                elif not nom_fichier_sortie.strip():
                    st.error("Donne un nom de fichier.")
                else:
                    dossier_original = CHEMIN_DONNEES_BRUTES.rsplit("/", 1)[0]
                    chemin_sortie = f"{dossier_original}/{nom_fichier_sortie.strip()}"
                    st.session_state.df_edite.to_csv(chemin_sortie, index=False)
                    st.success(f"Enregistré → {chemin_sortie}")
        with col_download:
            st.write("")
            st.download_button(
                "⬇️ Télécharger le CSV",
                data=st.session_state.df_edite.to_csv(index=False).encode("utf-8"),
                file_name=nom_fichier_sortie.strip() or "data_modifiee.csv",
                mime="text/csv",
            )

# ─── Onglet Prédiction ────────────────────────────────────────────────────────
with onglet_prediction:
    valeurs = recuperer_valeurs_valides()

    col_magasin, col_evenements, col_contexte = st.columns(3)

    with col_magasin:
        st.subheader("Magasin")
        store_id = st.selectbox("Identifiant du magasin", valeurs.get("Store_ID", VALEURS_PAR_DEFAUT["Store_ID"]))
        store_type = st.selectbox("Type de magasin", valeurs.get("Store_Type", VALEURS_PAR_DEFAUT["Store_Type"]))
        city = st.selectbox("Ville", valeurs.get("City", VALEURS_PAR_DEFAUT["City"]))
        store_size = st.number_input("Surface du magasin (m²)", min_value=100, value=4700, step=100)
        competitor_distance = st.number_input("Distance concurrent (km)", min_value=0.0, value=2.5, step=0.1)

    with col_evenements:
        st.subheader("Promotions & événements")
        promo = st.checkbox("Promotion active", value=True)
        holiday = st.checkbox("Jour férié")
        school_holiday = st.checkbox("Congés scolaires")
        weekend = st.checkbox("Week-end", value=True)

    with col_contexte:
        st.subheader("Contexte du jour")
        date = st.date_input("Date")
        temperature = st.number_input("Température (°C)", value=28.5)
        rainfall = st.number_input("Pluviométrie (mm)", min_value=0.0, value=15.0)
        fuel_price = st.number_input("Prix carburant (FCFA/litre)", value=680.0)
        inflation_rate = st.number_input("Taux d'inflation (%)", value=5.5)
        economic_index = st.number_input("Indice économique", value=98.0)

    if st.button("🔮 Prédire les ventes", type="primary"):
        payload = {
            "Store_ID": store_id, "Store_Type": store_type, "City": city,
            "Store_Size": int(store_size), "Competitor_Distance": float(competitor_distance),
            "Promo": int(promo), "Holiday": int(holiday),
            "School_Holiday": int(school_holiday), "Weekend": int(weekend),
            "Temperature": float(temperature), "Rainfall": float(rainfall), "Fuel_Price": float(fuel_price),
            "Inflation_Rate": float(inflation_rate), "Economic_Index": float(economic_index),
            "month": date.month, "day_of_week": date.weekday(),
            "week_of_year": date.isocalendar()[1], "quarter": (date.month - 1) // 3 + 1,
            "season": 0 if date.month in (11, 12, 1, 2) else 1 if date.month in (3, 4, 5) else 2 if date.month in (6, 7) else 3,
            "year": date.year,
        }
        try:
            reponse = requests.post(f"{URL_API}/predict", json=payload, timeout=10)
            reponse.raise_for_status()
            resultat = reponse.json()

            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric("Ventes prédites", f"{resultat['Sales_prediction']:,.0f} FCFA")
                st.caption(resultat["confidence_note"])
            with col2:
                metriques = resultat.get("model_metrics", {})
                st.write("**Métriques du modèle**")
                st.write(f"MAE : {metriques.get('MAE', '—')} | RMSE : {metriques.get('RMSE', '—')} | "
                         f"R² : {metriques.get('R2', '—')}")
        except requests.exceptions.RequestException as erreur:
            st.error(f"Impossible de contacter l'API sur {URL_API}. Vérifie qu'elle tourne "
                     f"(`uvicorn app:app --reload` dans le dossier api/). Détail : {erreur}")

# ─── Onglet Insights (à partir de tes propres résultats) ─────────────────────
with onglet_insights:
    st.subheader("Performance des modèles testés")
    resultats_modeles = pd.DataFrame([
        {"Modèle": "Régression Linéaire", "MAE": 696, "RMSE": 864, "R²": 0.8923},
        {"Modèle": "Decision Tree", "MAE": 880, "RMSE": 1120, "R²": 0.8190},
        {"Modèle": "Random Forest", "MAE": 790, "RMSE": 1004, "R²": 0.8545},
        {"Modèle": "XGBoost (optimisé)", "MAE": 696, "RMSE": 874, "R²": 0.8898},
    ])
    st.dataframe(resultats_modeles, width='stretch', hide_index=True)

    st.subheader("Importance des variables (XGBoost)")
    try:
        feature_imp = pd.read_csv("../reports/feature_importance.csv")
        st.bar_chart(feature_imp.set_index("Feature")["Importance"].head(15))
    except FileNotFoundError:
        st.info("Place `feature_importance.csv` dans `reports/` (chemin relatif depuis `ui/`) pour afficher ce graphique.")

    st.subheader("À retenir")
    st.markdown("""
    - **Store_Type** est le facteur n°1 (33.9%) : un magasin de type A vend en moyenne 58% de plus qu'un type C.
    - **Promo** est le levier le plus actionnable (+27.9% de ventes), **amplifié le week-end** (+42% contre +28% en semaine).
    - Le modèle explique **89% des variations de ventes** (R²=0.89), avec une erreur moyenne de ±7.55% (MAPE).
    """)
