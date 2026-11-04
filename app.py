# python -m streamlit run app.py

import json
import os
import time
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from algoritmos import agrupamiento, busqueda, similitud
from algoritmos.texto import cargar_corpus, limpiar

st.set_page_config(page_title="Análisis de algoritmos - Bibliometría", layout="wide")


@st.cache_data
def cargar_datos():
    corpus = cargar_corpus("data/corpus.json")
    palabras = {a["id"]: limpiar(a["abstract"]) for a in corpus}
    idf = similitud.calcular_idf(list(palabras.values()))
    embeddings = None
    if os.path.exists("data/embeddings.json"):
        with open("data/embeddings.json", encoding="utf-8") as f:
            embeddings = json.load(f)
    return corpus, palabras, idf, embeddings


corpus, palabras, idf, embeddings = cargar_datos()
por_id = {a["id"]: a for a in corpus}
ids = [a["id"] for a in corpus]


def nombre_corto(i):
    return f"{i}. {por_id[i]['titulo'][:60]}"


def vector(modelo, i):
    return embeddings[modelo]["vectores"][str(i)]


def inicio():
    st.title("Análisis de algoritmos en el contexto de la bibliometría")
    st.write("20 artículos sobre inteligencia artificial en educación. Todos los algoritmos "
             "trabajan sobre el abstract de cada artículo.")
    tabla = pd.DataFrame([{"Id": a["id"], "Título": a["titulo"], "Autores": ", ".join(a["autores"]),
                           "Palabras útiles": len(palabras[a["id"]])} for a in corpus])
    st.dataframe(tabla, hide_index=True)
    elegido = st.selectbox("Ver un abstract", ids, format_func=nombre_corto)
    st.write(por_id[elegido]["abstract"])
    st.caption("Palabras útiles: " + " ".join(palabras[elegido]))


def pagina_similitud():
    st.title("1 · Similitud textual")
    elegidos = st.multiselect("Elige dos o más artículos", ids, default=[11, 16], format_func=nombre_corto)
    if len(elegidos) < 2:
        st.info("Elige al menos dos artículos.")
        return

    filas = []
    for a, b in combinations(elegidos, 2):
        pa, pb = palabras[a], palabras[b]
        fila = {"Par": f"{a} y {b}",
                "Levenshtein": similitud.levenshtein(pa, pb)[0],
                "LCS": similitud.subsecuencia_comun(pa, pb)[0],
                "Jaccard": similitud.jaccard(pa, pb)[0],
                "Coseno TF-IDF": similitud.coseno_tfidf(pa, pb, idf)[0]}
        if embeddings:
            fila["GloVe (coseno)"] = similitud.coseno(vector("glove", a), vector("glove", b))
            fila["Sentence-BERT (coseno)"] = similitud.coseno(vector("sbert", a), vector("sbert", b))
            fila["Sentence-BERT (euclidiana)"] = similitud.euclidiana(vector("sbert", a), vector("sbert", b))
        filas.append(fila)
    st.write("Similitud de 0 (nada parecidos) a 1 (iguales). La euclidiana es una distancia: 0 = iguales.")
    st.dataframe(pd.DataFrame(filas).round(4), hide_index=True)
    if not embeddings:
        st.warning("Falta data/embeddings.json: ejecuta python generar_embeddings.py para ver los modelos de IA.")

    # detalle del primer par
    a, b = elegidos[0], elegidos[1]
    pa, pb = palabras[a], palabras[b]
    st.subheader(f"Cómo se calculó para {a} y {b}")
    sim, D = similitud.levenshtein(pa, pb)
    st.write(f"**Levenshtein:** distancia = {D[-1][-1]} operaciones; "
             f"similitud = 1 - {D[-1][-1]} / {max(len(pa), len(pb))} = {sim:.4f}. Esquina de la tabla:")
    st.dataframe(pd.DataFrame([fila[:8] for fila in D[:8]],
                              index=[f"{k}:{p}" for k, p in enumerate(["-"] + pa[:7])],
                              columns=[f"{k}:{p}" for k, p in enumerate(["-"] + pb[:7])]))
    sim, L = similitud.subsecuencia_comun(pa, pb)
    st.write(f"**LCS:** {L[-1][-1]} palabras en común en el mismo orden; "
             f"similitud = 2 · {L[-1][-1]} / ({len(pa)} + {len(pb)}) = {sim:.4f}")
    sim, comunes = similitud.jaccard(pa, pb)
    st.write(f"**Jaccard:** {len(comunes)} palabras en común: {', '.join(sorted(comunes))}")
    sim, u, v = similitud.coseno_tfidf(pa, pb, idf)
    pesos = [{"Palabra": p, "Peso en A": u[p], "Peso en B": v[p], "Producto": u[p] * v[p]}
             for p in u if p in v]
    st.write(f"**Coseno TF-IDF:** {sim:.4f}. Palabras compartidas y su aporte:")
    st.dataframe(pd.DataFrame(pesos).sort_values("Producto", ascending=False).round(5), hide_index=True)


def pagina_agrupamiento():
    st.title("2 · Agrupamiento jerárquico")
    opciones = ["TF-IDF"] + (["GloVe", "Sentence-BERT"] if embeddings else [])
    representacion = st.radio("Cómo representar cada abstract", opciones, horizontal=True)
    if representacion == "TF-IDF":
        D = agrupamiento.matriz_distancias(
            20, lambda i, j: similitud.coseno_tfidf(palabras[ids[i]], palabras[ids[j]], idf)[0])
    else:
        modelo = "glove" if representacion == "GloVe" else "sbert"
        D = agrupamiento.matriz_distancias(
            20, lambda i, j: similitud.coseno(vector(modelo, ids[i]), vector(modelo, ids[j])))

    nombres = [nombre_corto(i)[:45] for i in ids]
    resultados = []
    for criterio, titulo in agrupamiento.CRITERIOS.items():
        uniones = agrupamiento.agrupar(D, criterio)
        ccc = agrupamiento.correlacion_cofenetica(D, uniones)
        resultados.append({"Criterio": titulo, "Correlación cofenética": round(ccc, 4)})
        fig, ax = plt.subplots(figsize=(10, 6))
        agrupamiento.dibujar_dendrograma(uniones, nombres, f"{titulo} · {representacion}", ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("¿Cuál árbol es más coherente?")
    st.dataframe(pd.DataFrame(resultados), hide_index=True)
    mejor = max(resultados, key=lambda r: r["Correlación cofenética"])
    st.success(f"{mejor['Criterio']}: es el árbol que mejor respeta las distancias reales "
               f"entre los abstracts (más cerca de 1 es mejor).")


def pagina_busqueda():
    st.title("3 · Búsqueda de patrones")
    patron = st.text_input("Texto a buscar", "generative AI")
    exacta = st.checkbox("Distinguir mayúsculas y minúsculas (coincidencia exacta)", value=True)
    if not patron:
        return
    if exacta:
        textos = {a["id"]: a["abstract"] for a in corpus}
    else:
        patron = patron.lower()
        textos = {a["id"]: a["abstract"].lower() for a in corpus}

    filas = []
    for nombre, (preprocesar, buscar) in busqueda.ALGORITMOS.items():
        inicio = time.perf_counter()
        tabla = preprocesar(patron)
        t_pre = (time.perf_counter() - inicio) * 1000

        inicio = time.perf_counter()
        total, comparaciones = 0, 0
        for texto in textos.values():
            posiciones, comp = buscar(texto, patron, tabla)
            total += len(posiciones)
            comparaciones += comp
        t_bus = (time.perf_counter() - inicio) * 1000

        filas.append({"Algoritmo": nombre, "Ocurrencias": total, "Comparaciones": comparaciones,
                      "Preprocesamiento (ms)": round(t_pre, 4), "Búsqueda (ms)": round(t_bus, 3)})
    st.dataframe(pd.DataFrame(filas), hide_index=True)

    st.subheader("Dónde aparece")
    for i, texto in textos.items():
        posiciones, _ = busqueda.buscar("KMP", texto, patron)
        if posiciones:
            texto = por_id[i]["abstract"]
            with st.expander(f"{nombre_corto(i)} · {len(posiciones)} vez/veces · {', '.join(por_id[i]['autores'])}"):
                for p in posiciones:
                    st.markdown(f"- posición {p}: …{texto[max(0, p - 40):p]}**{texto[p:p + len(patron)]}**"
                                f"{texto[p + len(patron):p + len(patron) + 40]}…")

    st.subheader("Tablas de preprocesamiento del patrón")
    st.write("**KMP**, tabla de fallos:")
    st.dataframe(pd.DataFrame([list(patron), busqueda.kmp_preprocesar(patron)],
                              index=["carácter", "fallo"]).astype(str))
    st.write("**Boyer-Moore**, última posición de cada carácter:")
    st.dataframe(pd.DataFrame([busqueda.bm_preprocesar(patron)]).astype(str), hide_index=True)
    hash_patron, potencia, primo = busqueda.rk_preprocesar(patron)
    st.write(f"**Rabin-Karp:** hash del patrón = {hash_patron}, 256^(m-1) mod {primo} = {potencia}")


PAGINAS = {"Inicio": inicio, "1 · Similitud": pagina_similitud,
           "2 · Agrupamiento": pagina_agrupamiento, "3 · Búsqueda": pagina_busqueda}
eleccion = st.sidebar.radio("Secciones", list(PAGINAS))
PAGINAS[eleccion]()
