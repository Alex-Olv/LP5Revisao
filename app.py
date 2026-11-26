
import streamlit as st
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# ==========================================
# 1. CONFIGURAÇÃO DA APLICAÇÃO
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="IMDb Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================

@st.cache_data
def carregar_dados():
    caminho = BASE_DIR / "dados" / "imdb_tratado.csv"
    return pd.read_csv(caminho)


df = carregar_dados()


# ==========================================
# 3. DEFINIÇÃO DOS GÊNEROS
# ==========================================

generos = [
    'Action', 'Adventure', 'Animation', 'Biography',
    'Comedy', 'Crime', 'Documentary', 'Drama',
    'Family', 'Fantasy', 'Film-Noir', 'Game-Show',
    'History', 'Horror', 'Music', 'Musical',
    'Mystery', 'News', 'Reality-TV', 'Romance',
    'Sci-Fi', 'Short', 'Sport', 'Talk-Show',
    'Thriller', 'War', 'Western'
]


# ==========================================
# 4. PREPARAÇÃO DAS CARACTERÍSTICAS
# ==========================================

colunas_numericas = [
    'Date', 'Rate', 'Votes', 'Duration', 'Episodes'
]

dados_numericos = df[colunas_numericas].copy()

dados_numericos = dados_numericos.fillna(
    dados_numericos.median()
)

scaler = StandardScaler()

numericas_escaladas = scaler.fit_transform(
    dados_numericos
)

numericas_escaladas = pd.DataFrame(
    numericas_escaladas,
    columns=colunas_numericas,
    index=df.index
)

# Combinar gêneros e características numéricas
X = pd.concat(
    [
        df[generos].fillna(0),
        numericas_escaladas * 1.5
    ],
    axis=1
)


# ==========================================
# 5. CRIAÇÃO DO MODELO KNN
# ==========================================

modelo_knn = NearestNeighbors(
    n_neighbors=6,
    metric='cosine'
)

modelo_knn.fit(X)


# ==========================================
# 6. FUNÇÃO DE RECOMENDAÇÃO DIVERSIFICADA
# ==========================================

def selecionar_diversificados(
    indices_candidatos,
    distancias_candidatos,
    quantidade=5,
    diversidade=0.30
):

    candidatos = list(indices_candidatos)

    if len(candidatos) == 0:
        return []

    # Associar cada filme à sua distância
    distancias = dict(
        zip(candidatos, distancias_candidatos)
    )

    # Manter apenas os 100 filmes mais próximos
    candidatos = sorted(
        candidatos,
        key=lambda i: distancias[i]
    )[:100]

    # Selecionar primeiro o filme mais semelhante
    selecionados = [
        min(candidatos, key=lambda i: distancias[i])
    ]

    candidatos.remove(selecionados[0])

    # Escolher os demais buscando diversidade
    while candidatos and len(selecionados) < quantidade:

        melhor_filme = None
        melhor_pontuacao = float('-inf')

        for candidato in candidatos:

            relevancia = 1 - distancias[candidato]

            semelhanca_com_selecionados = cosine_similarity(
                X.loc[[candidato]],
                X.loc[selecionados]
            ).max()

            pontuacao = (
                (1 - diversidade) * relevancia
                - diversidade * semelhanca_com_selecionados
            )

            if pontuacao > melhor_pontuacao:
                melhor_pontuacao = pontuacao
                melhor_filme = candidato

        selecionados.append(melhor_filme)
        candidatos.remove(melhor_filme)

    return selecionados


# ==========================================
# 7. INTERFACE PRINCIPAL
# ==========================================

st.title("🎬 IMDb Movie Recommender")

st.write(
    "Descubra filmes e séries que combinam com seu gosto!"
)

st.divider()

modo = st.selectbox(
    "Como deseja receber recomendações?",
    [
        "Por filme",
        "Por gênero",
        "Por filme e gênero"
    ]
)

filmes = sorted(
    df["Name"].dropna().unique()
)


# ==========================================
# 8. RECOMENDAÇÃO POR FILME
# ==========================================

if modo == "Por filme":

    filme = st.selectbox(
        "Escolha um filme:",
        filmes
    )

    if st.button("Buscar recomendações"):

        indice = df.index[
            df["Name"] == filme
        ][0]

        distancias, indices = modelo_knn.kneighbors(
            X.loc[[indice]],
            n_neighbors=len(df)
        )

        candidatos = [
            i for i in indices[0]
            if i != indice
        ]

        distancias_candidatos = [
            distancias[0][pos]
            for pos, i in enumerate(indices[0])
            if i != indice
        ]

        melhores = selecionar_diversificados(
            candidatos,
            distancias_candidatos,
            quantidade=5
        )

        resultados = df.loc[melhores]

        st.subheader("🎥 Filmes semelhantes")

        if resultados.empty:
            st.warning(
                "Não encontramos filmes semelhantes."
            )
        else:
            st.dataframe(
                resultados[["Name", "Rate"]],
                hide_index=True,
                use_container_width=True
            )


# ==========================================
# 9. RECOMENDAÇÃO POR GÊNERO
# ==========================================

elif modo == "Por gênero":

    genero = st.selectbox(
        "Escolha um gênero:",
        generos
    )

    if st.button("Buscar recomendações"):

        resultados = df[df[genero] == 1].copy()

        resultados = resultados.sort_values(
            "Rate",
            ascending=False,
            na_position="last"
        ).head(5)

        st.subheader("🎭 Filmes do gênero escolhido")

        if resultados.empty:
            st.warning(
                "Não encontramos filmes para esse gênero."
            )
        else:
            st.dataframe(
                resultados[["Name", "Rate"]],
                hide_index=True,
                use_container_width=True
            )


# ==========================================
# 10. RECOMENDAÇÃO POR FILME + GÊNERO
# ==========================================

else:

    filme = st.selectbox(
        "Escolha um filme:",
        filmes
    )

    genero = st.selectbox(
        "Escolha um gênero:",
        generos
    )

    if st.button("Buscar recomendações"):

        indice = df.index[
            df["Name"] == filme
        ][0]

        distancias, indices = modelo_knn.kneighbors(
            X.loc[[indice]],
            n_neighbors=len(df)
        )

        candidatos = [
            i for i in indices[0]
            if i != indice and df.loc[i, genero] == 1
        ]

        distancias_candidatos = [
            distancias[0][pos]
            for pos, i in enumerate(indices[0])
            if i != indice and df.loc[i, genero] == 1
        ]

        melhores = selecionar_diversificados(
            candidatos,
            distancias_candidatos,
            quantidade=5
        )

        resultados = df.loc[melhores]

        st.subheader("⭐ Recomendações personalizadas")

        if resultados.empty:
            st.warning(
                "Não encontramos recomendações para essa combinação."
            )
        else:
            st.dataframe(
                resultados[["Name", "Rate"]],
                hide_index=True,
                use_container_width=True
            )


# ==========================================
# 11. RODAPÉ
# ==========================================

st.divider()

st.caption(
    "Projeto acadêmico de Machine Learning | Algoritmo KNN"
)

