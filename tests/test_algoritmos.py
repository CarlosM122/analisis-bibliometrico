# Compara los algoritmos con SciPy, NumPy y str.find.
#   python tests/test_algoritmos.py

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import squareform

from algoritmos import agrupamiento, busqueda, similitud
from algoritmos.texto import cargar_corpus, limpiar

corpus = cargar_corpus("data/corpus.json")
documentos = [limpiar(a["abstract"]) for a in corpus]

assert similitud.levenshtein(list("gato"), list("pato"))[1][4][4] == 1
assert similitud.levenshtein(list("kitten"), list("sitting"))[1][6][7] == 3
assert similitud.subsecuencia_comun(list("ABCBDAB"), list("BDCABA"))[1][7][6] == 4
assert abs(similitud.jaccard(list("abc"), list("bcd"))[0] - 0.5) < 1e-12
u, v = np.random.rand(384), np.random.rand(384)
assert abs(similitud.coseno(list(u), list(v)) - u @ v / np.linalg.norm(u) / np.linalg.norm(v)) < 1e-12
print("Similitud: OK")

idf = similitud.calcular_idf(documentos)
D = agrupamiento.matriz_distancias(
    20, lambda i, j: similitud.coseno_tfidf(documentos[i], documentos[j], idf)[0])
nombres_scipy = {"simple": "single", "completo": "complete", "promedio": "average"}
for criterio, nombre in nombres_scipy.items():
    uniones = agrupamiento.agrupar(D, criterio)
    Z = linkage(squareform(D, checks=False), method=nombre)
    assert sorted(round(u[2], 9) for u in uniones) == sorted(round(x, 9) for x in Z[:, 2])
    nuestra = agrupamiento.correlacion_cofenetica(D, uniones)
    referencia = cophenet(Z, squareform(D, checks=False))[0]
    assert abs(nuestra - referencia) < 1e-9
print("Agrupamiento: OK (igual a SciPy)")

def con_find(texto, patron):
    posiciones, i = [], texto.find(patron)
    while i != -1:
        posiciones.append(i)
        i = texto.find(patron, i + 1)
    return posiciones

random.seed(1)
casos = 0
for _ in range(3000):
    letras = random.choice(["ab", "abc", "acgt"])
    texto = "".join(random.choice(letras) for _ in range(random.randint(0, 60)))
    patron = "".join(random.choice(letras) for _ in range(random.randint(1, 6)))
    for nombre in busqueda.ALGORITMOS:
        assert busqueda.buscar(nombre, texto, patron)[0] == con_find(texto, patron)
        casos += 1
for a in corpus:
    texto = a["abstract"].lower()
    for patron in ["ai", "learning", "generative ai", "e", "zzz"]:
        for nombre in busqueda.ALGORITMOS:
            assert busqueda.buscar(nombre, texto, patron)[0] == con_find(texto, patron)
            casos += 1
print(f"Búsqueda: OK ({casos} búsquedas iguales a str.find)")
