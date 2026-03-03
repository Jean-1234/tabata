import streamlit as st
import pandas as pd
import numpy as np
import os
from tabata import Opset, Selector, Tube
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pickle

st.set_page_config(page_title="AppTube - Analyse de vols", layout="wide")
st.title("AppTube - Analyse des vols")

if 'current_vol' not in st.session_state:
    st.session_state.current_vol = 0
if 'ds_montee' not in st.session_state:
    st.session_state.ds_montee = None
if 'df_phys' not in st.session_state:
    st.session_state.df_phys = None
if 'tube' not in st.session_state:
    st.session_state.tube = None

st.sidebar.header("Donnees")

default_path = r"C:\Users\12412748\MACS2\Travaux Python\tabata\notebooks\data\out"
data_path = st.sidebar.text_input("Chemin du dossier data/out", value=default_path)

if not os.path.exists(data_path):
    st.sidebar.error(f"Dossier introuvable : {data_path}")
    st.sidebar.info("Modifiez le chemin ci-dessus pour pointer vers votre dossier data/out")
    st.stop()

fichiers_h5  = [f for f in os.listdir(data_path) if f.endswith('.h5')]
fichiers_csv = [f for f in os.listdir(data_path) if f.endswith('.csv')]

if not fichiers_h5:
    st.sidebar.warning("Aucun fichier .h5 trouve dans ce dossier")
    st.stop()

selected_file = st.sidebar.selectbox("Fichier H5 (montees)", fichiers_h5)
filepath = os.path.join(data_path, selected_file)

if st.sidebar.button("Charger"):
    with st.spinner("Chargement..."):
        st.session_state.ds_montee = Opset(filepath)
        st.session_state.current_vol = 0
    st.sidebar.success(f"OK {len(st.session_state.ds_montee)} vols charges")

if fichiers_csv:
    selected_csv = st.sidebar.selectbox("CSV physique (df_phys)", ["— aucun —"] + fichiers_csv)
    if selected_csv != "— aucun —":
        csv_path = os.path.join(data_path, selected_csv)
        st.session_state.df_phys = pd.read_csv(csv_path)
        st.sidebar.success(f"OK df_phys : {len(st.session_state.df_phys)} vols")

if st.session_state.ds_montee is None:
    st.info("Chargez un fichier .h5 dans le menu lateral")
    st.stop()

ds_montee = st.session_state.ds_montee
df_phys   = st.session_state.df_phys
filepath  = ds_montee.storename

st.sidebar.markdown("---")
st.sidebar.header("Navigation")

vol_idx = st.sidebar.slider("Numero de vol", 0, len(ds_montee)-1,
                             st.session_state.current_vol)
st.session_state.current_vol = vol_idx
df_vol = ds_montee[vol_idx]

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Precedent"):
        st.session_state.current_vol = max(0, vol_idx - 1)
        st.rerun()
with c2:
    if st.button("Suivant"):
        st.session_state.current_vol = min(len(ds_montee)-1, vol_idx + 1)
        st.rerun()

st.sidebar.write(f"Vol {vol_idx} : {df_vol.index.name}")

if df_phys is not None and vol_idx < len(df_phys):
    row = df_phys.iloc[vol_idx]
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Grandeurs physiques**")
    for col in ['TAS_moy_ms', 'Vz_moy_ms', 'eta', 'TSFC', 'conso_kg']:
        if col in df_phys.columns:
            st.sidebar.write(f"{col} : {row[col]:.4f}")

mode = st.sidebar.radio("Mode", ["Visualisation", "Tubes", "Scores", "Degradation"])

# MODE VISUALISATION
if mode == "Visualisation":
    st.header(f"Vol {vol_idx} - {df_vol.index.name}")
    variables = df_vol.columns.tolist()
    colname = st.selectbox("Variable", variables, index=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_vol.index, y=df_vol[colname], mode='lines', name=colname))
    fig.update_layout(height=400, title=colname)
    st.plotly_chart(fig, use_container_width=True)
    if 'ALT [ft]' in df_vol.columns and colname != 'ALT [ft]':
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        fig2.add_trace(go.Scatter(x=df_vol.index, y=df_vol['ALT [ft]'],
                                  name='ALT [ft]', line=dict(color='steelblue')), row=1, col=1)
        fig2.add_trace(go.Scatter(x=df_vol.index, y=df_vol[colname],
                                  name=colname, line=dict(color='orange')), row=2, col=1)
        fig2.update_layout(height=500, title=f"ALT vs {colname}")
        st.plotly_chart(fig2, use_container_width=True)

# MODE TUBES
elif mode == "Tubes":
    st.header("Tubes de confiance")
    variables = df_vol.columns.tolist()
    col_left, col_right = st.columns(2)
    with col_left:
        target = st.selectbox("Variable cible", variables,
                              index=variables.index('Q_1 [lb/h]') if 'Q_1 [lb/h]' in variables else 0)
    with col_right:
        default_factors = [v for v in [
    'N1_1 [% rpm]', 'ALT [ft]', 'TAT [deg C]',
    'M [Mach]', 'TLA_1 [deg]', 'TLA_2 [deg]'
] if v in variables]
        factors = st.multiselect("Facteurs", variables, default=default_factors)
    with st.expander("Parametres avances"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            max_features = st.number_input("Max features", 1, 10, 3)
        with c2:
            retry = st.number_input("Retry number", 5, 50, 10)
        with c3:
            sample = st.number_input("Sample %", 0.001, 0.1, 0.01, format="%.3f")
        with c4:
            tube_factor = st.number_input("Tube factor", 1.0, 50.0, 10.0)

    if st.button("Lancer l'apprentissage") and factors:
        with st.spinner("Apprentissage en cours..."):
            T = Tube(filepath)
            T.variables = {target}
            T.factors   = set(factors)
            T.learn_params['max_features']    = max_features
            T.learn_params['retry_number']    = retry
            T.learn_params['samples_percent'] = sample
            T.tube_params['tube_factor']      = tube_factor
            T.feature_params['use_time']      = 'Yes'
            T.fit()
            st.session_state.tube = T
        st.success("Tube cree")
        save_path = os.path.join(data_path, f"tube_{target.replace(' ','_').replace('/','_')}.pkl")
        reg_serializable = {k: list(v) for k, v in T._reg.items()}
        save_data = T.__dict__.copy()
        save_data['_reg'] = reg_serializable
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        st.info(f"Sauvegarde : {save_path}")

    if st.session_state.tube is not None and isinstance(st.session_state.tube, Tube):
        T = st.session_state.tube
        T.rewind(vol_idx)
        if target in T._reg:
            z, zmin, zmax = T.estimate(target)
            y = df_vol[target].values if target in df_vol.columns else None
            if y is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y, name=target, line=dict(color='steelblue')))
                fig.add_trace(go.Scatter(y=z, name='Prediction',
                                         line=dict(color='darkgreen', dash='dot')))
                fig.add_trace(go.Scatter(y=zmin, name='Tube min',
                                         line=dict(color='green', width=0),
                                         fill='tonexty', fillcolor='rgba(0,180,0,0.2)'))
                fig.add_trace(go.Scatter(y=zmax, name='Tube max',
                                         line=dict(color='green', width=0),
                                         fill='tonexty', fillcolor='rgba(0,180,0,0.2)'))
                fig.update_layout(height=400, title=f"Tube sur {target} - Vol {vol_idx}")
                st.plotly_chart(fig, use_container_width=True)

# MODE SCORES
elif mode == "Scores":
    st.header("Scores des tubes")
    tube_files = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
    if not tube_files:
        st.info("Aucun tube sauvegarde. Creez-en un dans le mode Tubes.")
    else:
        selected = st.selectbox("Choisir un tube", tube_files)
        if st.button("Charger ce tube"):
            with open(os.path.join(data_path, selected), 'rb') as f:
                data = pickle.load(f)
            T = Tube(filepath)
            if isinstance(data, dict):
                T.__dict__.update(data)
            st.session_state.tube = T
            st.success("Tube charge")

    if st.session_state.tube is not None and isinstance(st.session_state.tube, Tube):
        T = st.session_state.tube
        st.write("**Facteurs utilises :**")
        try:
            st.dataframe(T.describe())
        except Exception as e:
            st.warning(f"describe() non disponible : {e}")
        with st.spinner("Calcul des scores..."):
            scores = T.scores()
        st.dataframe(scores.head(20))
        for col in scores.columns[1:]:
            pct = scores[col] / scores['N'] * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(pct))), y=pct.values,
                                     mode='markers', name='% hors tube',
                                     marker=dict(size=4)))
            window = min(50, max(5, len(pct)//10))
            ma = pct.rolling(window=window).mean()
            fig.add_trace(go.Scatter(x=list(range(len(ma))), y=ma.values,
                                     mode='lines', name='Tendance',
                                     line=dict(color='red', width=2)))
            fig.update_layout(height=400, title=f"% hors tube - {col}",
                              xaxis_title="Numero de vol", yaxis_title="% hors tube")
            st.plotly_chart(fig, use_container_width=True)

# MODE DEGRADATION
elif mode == "Degradation":
    st.header("Analyse de la degradation moteur")
    if df_phys is None:
        st.warning("Chargez le fichier physique_montee.csv dans le menu lateral.")
        st.stop()

    # Diagnostic colonnes
    st.subheader("Colonnes disponibles dans df_phys")
    st.write(df_phys.columns.tolist())
    st.dataframe(df_phys.head(3))

    window = 50
    cols_num = df_phys.select_dtypes(include=[np.number]).columns.tolist()

    # Choix dynamique des colonnes a tracer
    col_x = 'vol' if 'vol' in df_phys.columns else df_phys.columns[0]

    col1, col2 = st.columns(2)
    plot_cols = [c for c in ['eta', 'TSFC', 'conso_kg', 'delta_atmo_K',
                              'TAS_moy_ms', 'Vz_moy_ms', 'F_moy_N', 'D_moy_N']
                 if c in df_phys.columns]

    if len(plot_cols) >= 1:
        with col1:
            y1 = plot_cols[0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_phys[col_x], y=df_phys[y1],
                                     mode='markers', name=y1,
                                     marker=dict(size=3, color='steelblue')))
            ma = df_phys[y1].rolling(window=window).mean()
            fig.add_trace(go.Scatter(x=df_phys[col_x], y=ma, mode='lines',
                                     name='Tendance', line=dict(color='red', width=2)))
            fig.update_layout(height=350, title=f"{y1} vs {col_x}",
                              xaxis_title=col_x, yaxis_title=y1)
            st.plotly_chart(fig, use_container_width=True)

    if len(plot_cols) >= 2:
        with col2:
            y2 = plot_cols[1]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_phys[col_x], y=df_phys[y2],
                                     mode='markers', name=y2,
                                     marker=dict(size=3, color='orange')))
            ma = df_phys[y2].rolling(window=window).mean()
            fig.add_trace(go.Scatter(x=df_phys[col_x], y=ma, mode='lines',
                                     name='Tendance', line=dict(color='red', width=2)))
            fig.update_layout(height=350, title=f"{y2} vs {col_x}",
                              xaxis_title=col_x, yaxis_title=y2)
            st.plotly_chart(fig, use_container_width=True)

    for y in plot_cols[2:]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_phys[col_x], y=df_phys[y],
                                 mode='markers', name=y,
                                 marker=dict(size=3, color='green')))
        ma = df_phys[y].rolling(window=window).mean()
        fig.add_trace(go.Scatter(x=df_phys[col_x], y=ma, mode='lines',
                                 name='Tendance', line=dict(color='red', width=2)))
        if y == 'delta_atmo_K':
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=300, title=f"{y} vs {col_x}",
                          xaxis_title=col_x, yaxis_title=y)
        st.plotly_chart(fig, use_container_width=True)