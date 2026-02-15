import streamlit as st
import pandas as pd
import numpy as np
import tabata as tbt 
import plotly.graph_objs as go

st.set_page_config(page_title="AppTube - Aircraft_01", layout="wide")
st.title("🛫 Analyse de Montée - Tubes de Confiance")

@st.cache_data
def load_data():
    # Import tabata seulement ici
    import sys
    sys.path.append('C:/Users/12412748/MACS2/Travaux Python')
    import tabata as tbt
    
    ds = tbt.Opset('../data/out/Aircraft_01_clean.h5')
    
    # Convertir en liste de DataFrames
    data = []
    for i in range(len(ds)):
        df = ds[i]
        data.append({
            'index': i,
            'name': df.index.name,
            'data': df
        })
    return data

try:
    vols = load_data()
    
    st.sidebar.header("Sélection")
    vol_idx = st.sidebar.slider("Vol", 0, len(vols)-1, 0)
    
    vol = vols[vol_idx]
    df = vol['data']
    
    st.header(f"Vol {vol_idx} - {vol['name']}")
    st.write(f"Durée : {len(df)} secondes")
    
    # Graphique
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(df))), y=df['ALT [ft]'], name='Altitude'))
    fig.update_layout(xaxis_title='Temps [s]', yaxis_title='Altitude [ft]')
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Altitude max", f"{df['ALT [ft]'].max():.0f} ft")
    col2.metric("Débit Q1 moyen", f"{df['Q_1 [lb/h]'].mean():.0f} lb/h")
    col3.metric("N1_1 moyen", f"{df['N1_1 [% rpm]'].mean():.1f} %")
    
except Exception as e:
    st.error(f"Erreur : {e}")
    import traceback
    st.code(traceback.format_exc())