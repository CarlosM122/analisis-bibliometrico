# Genera data/embeddings.json con GloVe y Sentence-BERT (requiere requirements-ia.txt).

import json

from algoritmos.texto import cargar_corpus, limpiar


def promedio(vectores):
    resultado = [0.0] * len(vectores[0])
    for vector in vectores:
        for k in range(len(vector)):
            resultado[k] += vector[k] / len(vectores)
    return resultado


def main():
    import gensim.downloader
    from sentence_transformers import SentenceTransformer

    corpus = cargar_corpus("data/corpus.json")

    print("Descargando GloVe...")
    glove = gensim.downloader.load("glove-wiki-gigaword-100")
    vectores_glove = {}
    for articulo in corpus:
        conocidas = [glove[p].tolist() for p in limpiar(articulo["abstract"]) if p in glove.key_to_index]
        vectores_glove[articulo["id"]] = promedio(conocidas)

    print("Descargando Sentence-BERT...")
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectores_sbert = {}
    for articulo in corpus:
        vectores_sbert[articulo["id"]] = sbert.encode(articulo["abstract"]).tolist()

    datos = {
        "glove": {"dimension": 100, "vectores": vectores_glove},
        "sbert": {"dimension": 384, "vectores": vectores_sbert},
    }
    with open("data/embeddings.json", "w", encoding="utf-8") as f:
        json.dump(datos, f)
    print("Guardado data/embeddings.json")


if __name__ == "__main__":
    main()
