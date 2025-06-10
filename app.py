import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# Título estilizado
st.set_page_config(layout="wide")
st.markdown("""
    <div style='margin-top: -65px; text-align: center;'>
        <h1 style='color: white; font-size: 42px;'>
             Aprendizado por Reforço - CliffWalking
        </h1>
        <p style='color: white; font-size: 16px; margin-top: -10px;'>
            Prof. Rodrigo Luna
        </p>
    </div>
""", unsafe_allow_html=True)

# Caminho dos arquivos
DATA_DIR = "data"
df_total = pd.read_excel(os.path.join(DATA_DIR, "Recompensas_Total.xlsx"))
erro_mc = np.load(os.path.join(DATA_DIR, "erro_monte_carlo.npy"))
erro_mc_every = np.load(os.path.join(DATA_DIR, "erro_monte_carlo_every.npy"))
erro_td = np.load(os.path.join(DATA_DIR, "erro_td.npy"))
erro_sarsa = np.load(os.path.join(DATA_DIR, "erro_sarsa.npy"))
erro_q = np.load(os.path.join(DATA_DIR, "erro_q_learning.npy"))
erro_dqn = np.load(os.path.join(DATA_DIR, "erro_dqn.npy"))

# Cores e nomes
cores_alg = {
    "Monte Carlo First-Visit": "#AB63FA",
    "Monte Carlo Every-Visit": "#FFA15A",
    "TD(0)": "#EF553B",
    "SARSA": "#00CC96",
    "Q-Learning": "#636EFA",
    "DQN": "#19D3F3"
}
ordem_alg = list(cores_alg.keys())

# Tabs na ordem solicitada
tabs = st.tabs([
    "Boxplot da Recompensa por Algoritmo",
    "Recompensa Média Suavizada por Episódio",
    "Boxplot do Erro Médio por Algoritmo",
    "Erro Médio Estimado por Episódio",
    "Episódio Greedy - Vídeos"
])

with tabs[0]:
    st.subheader("Boxplot da Recompensa por Algoritmo")
    fig1 = go.Figure()
    for alg in ordem_alg:
        df_alg = df_total[df_total['algoritmo'] == alg]
        fig1.add_trace(go.Box(y=df_alg['recompensa'], name=alg, boxpoints='outliers', marker_color=cores_alg[alg]))
    fig1.update_layout(
        template="plotly_dark",
        yaxis_title="Recompensa Acumulada",
        xaxis_title="Algoritmo",
        title="Boxplot da Recompensa por Algoritmo",
        updatemenus=[{
            "buttons": [
                {"label": alg, "method": "update",
                 "args": [{"visible": [alg == a for a in ordem_alg]},
                          {"title": f"Boxplot da Recompensa - {alg}"}]}
                for alg in ordem_alg
            ] + [{
                "label": "Todos", "method": "update",
                "args": [{"visible": [True] * len(ordem_alg)},
                         {"title": "Boxplot da Recompensa por Algoritmo"}]
            }],
            "direction": "down",
            "showactive": True,
            "x": 1.1, "y": 1.2
        }]
    )
    st.plotly_chart(fig1, use_container_width=True)

with tabs[1]:
    st.subheader("Recompensa Média Suavizada por Episódio")
    fig2 = go.Figure()
    tamanho_janela = 50
    for alg in ordem_alg:
        df_alg = df_total[df_total['algoritmo'] == alg]
        media = df_alg.groupby('episodio')['recompensa'].mean().reset_index()
        media['recompensa_suavizada'] = media['recompensa'].rolling(window=tamanho_janela, min_periods=1).mean()
        fig2.add_trace(go.Scatter(x=media['episodio'], y=media['recompensa_suavizada'],
                                  mode='lines', name=alg, line=dict(color=cores_alg[alg], width=2)))
    fig2.update_layout(
        template="plotly_dark",
        title="Recompensa Média Suavizada por Episódio",
        xaxis_title="Episódio",
        yaxis_title="Recompensa Média Suavizada",
        updatemenus=[{
            "buttons": [
                {"label": alg, "method": "update",
                 "args": [{"visible": [alg == a for a in ordem_alg]},
                          {"title": f"Recompensa Média Suavizada - {alg}"}]}
                for alg in ordem_alg
            ] + [{
                "label": "Todos", "method": "update",
                "args": [{"visible": [True] * len(ordem_alg)},
                         {"title": "Recompensa Média Suavizada por Episódio"}]
            }],
            "direction": "down",
            "showactive": True,
            "x": 1.1, "y": 1.2
        }]
    )
    st.plotly_chart(fig2, use_container_width=True)

with tabs[2]:
    st.subheader("Boxplot do Erro Médio por Algoritmo")
    df_erros = pd.DataFrame({
        "Monte Carlo First-Visit": erro_mc,
        "Monte Carlo Every-Visit": erro_mc_every,
        "TD(0)": erro_td,
        "SARSA": erro_sarsa,
        "Q-Learning": erro_q,
        "DQN": erro_dqn
    })
    df_melted = df_erros.melt(var_name="Algoritmo", value_name="Erro Médio")
    fig3 = go.Figure()
    for alg in ordem_alg:
        fig3.add_trace(go.Box(
            y=df_melted[df_melted["Algoritmo"] == alg]["Erro Médio"],
            name=alg,
            marker_color=cores_alg[alg],
            boxpoints='outliers'
        ))
    fig3.update_layout(
        template="plotly_dark",
        title="Boxplot do Erro Médio por Algoritmo",
        yaxis_title="Erro Médio (|Q - Q*|)",
        xaxis_title="Algoritmo",
        updatemenus=[{
            "buttons": [
                {"label": alg, "method": "update",
                 "args": [{"visible": [alg == a for a in ordem_alg]},
                          {"title": f"Boxplot do Erro Médio - {alg}"}]}
                for alg in ordem_alg
            ] + [{
                "label": "Todos", "method": "update",
                "args": [{"visible": [True] * len(ordem_alg)},
                         {"title": "Boxplot do Erro Médio por Algoritmo"}]
            }],
            "direction": "down",
            "showactive": True,
            "x": 1.1, "y": 1.2
        }]
    )
    st.plotly_chart(fig3, use_container_width=True)

with tabs[3]:
    st.subheader("Erro Médio Estimado por Episódio")
    nomes = ["Monte Carlo Every-Visit", "Monte Carlo First-Visit", "TD(0)", "SARSA", "Q-Learning", "DQN"]
    erros = [erro_mc_every, erro_mc, erro_td, erro_sarsa, erro_q, erro_dqn]
    fig4 = go.Figure()
    for nome in nomes:
        vetor = dict(zip(nomes, erros))[nome]
        fig4.add_trace(go.Scatter(y=vetor, mode='lines', name=nome, line=dict(color=cores_alg[nome], width=2)))
    fig4.update_layout(
        template="plotly_dark",
        title="Erro Médio Estimado por Episódio",
        xaxis_title="Episódios",
        yaxis_title="Erro Médio Estimado",
        updatemenus=[{
            "buttons": [
                {"label": nome, "method": "update",
                 "args": [{"visible": [nome == n for n in nomes]},
                          {"title": f"Erro Médio Estimado - {nome}"}]}
                for nome in nomes
            ] + [{
                "label": "Todos", "method": "update",
                "args": [{"visible": [True] * len(nomes)},
                         {"title": "Erro Médio Estimado por Episódio"}]
            }],
            "direction": "down",
            "showactive": True,
            "x": 1.1, "y": 1.2
        }]
    )
    st.plotly_chart(fig4, use_container_width=True)

with tabs[4]:
    st.subheader("Episódio Greedy - Vídeos")

    gifs = {
        "Monte Carlo First-Visit": "Monte_Carlo.gif",
        "Monte Carlo Every-Visit": "Monte_Carlo_Every.gif",
        "TD(0)": "TD(0).gif",
        "SARSA": "SARSA.gif",
        "Q-Learning": "Q-Learning.gif",
        "DQN": "DQN.gif"
    }

    for nome, arquivo in gifs.items():
        st.markdown(f"### {nome}")
        caminho_gif = os.path.join("data", arquivo)

        if os.path.exists(caminho_gif):
            st.image(caminho_gif)
        else:
            st.warning(f"GIF não encontrado: {arquivo}")


