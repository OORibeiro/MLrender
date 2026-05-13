from fastapi import FastAPI
import requests
import pandas as pd
import numpy as np
import skfuzzy as fuzzy
import google.generativeai as genai
import json

app = FastAPI()

import os

genai.configure(
    api_key="AIzaSyCoVrEe62FmPEaH4UO18Ft9eEKHfAC3Ok0"
)

model = genai.GenerativeModel("gemini-1.5-flash")

# =========================================
# FUNÇÃO INTERNA DO ML
# =========================================

def processar_grupos():

    skills = ['fisico', 'tecnico', 'experiencia']

    # LOGIN
    login_url = "https://grupo-escoteiro.onrender.com/login"

    dados_login = {
        "email": "luis@teste.com",
        "password": "123"
    }

    login_response = requests.post(
        login_url,
        json=dados_login
    )

    token = login_response.json()['token']

    # TOKEN JWT
    headers = {
        "Authorization": f"Bearer {token}"
    }

    # BUSCAR MEMBROS
    members_url = "https://grupo-escoteiro.onrender.com/members"

    response = requests.get(
        members_url,
        headers=headers
    )

    dados = response.json()

    # DATAFRAME
    df = pd.DataFrame(dados)

    # RENOMEAR
    df = df.rename(columns={
        'name': 'nome',
        'skill_fisico': 'fisico',
        'skill_tecnico': 'tecnico',
        'skill_experiencia': 'experiencia'
    })

    # NUMÉRICO
    df[skills] = df[skills].fillna(1).astype(int)

    # FUZZY
    data_fcm = df[skills].values.T

    cntr, u, u0, d, jm, p, fpc = fuzzy.cluster.cmeans(
        data_fcm,
        c=3,
        m=2,
        error=0.005,
        maxiter=1000,
        init=None
    )

    # ESPECIALIDADES
    mapa_clusters = {}

    for i, centro in enumerate(cntr):

        skill_nome = skills[np.argmax(centro)]

        mapa_clusters[i] = skill_nome

    probabilidades = u.T

    esp_1 = []
    esp_2 = []

    for prob in probabilidades:

        indices_ordenados = np.argsort(prob)[::-1]

        esp_1.append(
            mapa_clusters[indices_ordenados[0]]
        )

        esp_2.append(
            mapa_clusters[indices_ordenados[1]]
        )

    df['especialidade'] = esp_1
    df['segunda_especialidade'] = esp_2

    df['pontuacao_total'] = df[skills].sum(axis=1)

    # GRUPOS
    grupos_ids = [0, 1, 2]

    df['grupo_final'] = -1

    contador_membros = [0, 0, 0]

    # DISTRIBUIÇÃO
    restantes = df.sort_values(
        'pontuacao_total',
        ascending=False
    )

    for idx in restantes.index:

        menor_grupo = contador_membros.index(
            min(contador_membros)
        )

        df.loc[idx, 'grupo_final'] = menor_grupo

        contador_membros[menor_grupo] += 1

    # JSON FINAL
    resultado = []

    for g in grupos_ids:

        membros = df[df['grupo_final'] == g]

        grupo = {
            "grupo": int(g),
            "membros": membros[
                [
                    'nome',
                    'especialidade',
                    'segunda_especialidade',
                    'fisico',
                    'tecnico',
                    'experiencia'
                ]
            ].to_dict(orient='records')
        }

        resultado.append(grupo)

    return {
        "fpc": float(fpc),
        "grupos": resultado
    }

# =========================================
# ENDPOINT DOS GRUPOS
# =========================================

@app.get("/gerar-grupos")
def gerar_grupos():

    return processar_grupos()

# =========================================
# ENDPOINT DAS ATIVIDADES
# =========================================

@app.get("/atividades")
def gerar_atividades():

    grupos = processar_grupos()

    prompt = f"""
    Você é um especialista em atividades escoteiras.

    Analise os grupos abaixo e sugira:
    - 2 atividades ideais
    - dificuldade
    - objetivo

    Responda SOMENTE em JSON.

    Dados:
    {json.dumps(grupos, ensure_ascii=False)}
    """

    resposta = model.generate_content(prompt)

    texto = resposta.text

    texto = texto.replace("```json", "")
    texto = texto.replace("```", "")
    texto = texto.strip()

    return json.loads(texto)

