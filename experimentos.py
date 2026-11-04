# Mide tiempos con entradas crecientes y estima el exponente k (pendiente log-log).
#   python experimentos.py   -> resultados/

import csv
import math
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algoritmos import agrupamiento, busqueda, similitud
from algoritmos.texto import cargar_corpus, limpiar

random.seed(2026)


def medir(funcion, repeticiones=3):
    # mediana del tiempo en ms
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        funcion()
        tiempos.append((time.perf_counter() - inicio) * 1000)
    tiempos.sort()
    return tiempos[len(tiempos) // 2]


def pendiente(tamanos, tiempos):
    # mínimos cuadrados sobre log(n), log(tiempo)
    x = [math.log(t) for t in tamanos]
    y = [math.log(t) for t in tiempos]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    arriba = sum((x[i] - mx) * (y[i] - my) for i in range(len(x)))
    abajo = sum((x[i] - mx) ** 2 for i in range(len(x)))
    return arriba / abajo


def experimento(nombre, eje, tamanos, algoritmos, teorico, preparar):
    print(f"\n{nombre}")
    filas = []
    for n in tamanos:
        datos = preparar(n)
        fila = {eje: n}
        for alg, funcion in algoritmos.items():
            fila[alg] = medir(lambda: funcion(datos))
        filas.append(fila)

    with open(f"resultados/{nombre}.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)

    resumen = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for alg in algoritmos:
        tiempos = [fila[alg] for fila in filas]
        k = pendiente(tamanos, tiempos)
        resumen.append({"experimento": nombre, "algoritmo": alg,
                        "k_medido": round(k, 2), "k_teorico": teorico[alg]})
        print(f"  {alg:<18} k medido = {k:.2f}   k teórico = {teorico[alg]}")
        ax.plot(tamanos, tiempos, marker="o", label=f"{alg}: k = {k:.2f} (teórico {teorico[alg]})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(eje + " (escala log)")
    ax.set_ylabel("tiempo en ms (escala log)")
    ax.set_title(nombre.replace("_", " "))
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"resultados/{nombre}.png", dpi=150)
    plt.close(fig)
    return resumen


def main():
    corpus = cargar_corpus("data/corpus.json")
    vocabulario = sorted({p for a in corpus for p in limpiar(a["abstract"])})
    texto_corpus = " ".join(a["abstract"].lower() for a in corpus)
    idf = {p: 1.0 for p in vocabulario}
    resumen = []

    # R1: textos de L palabras al azar
    resumen += experimento(
        "similitud", "L (palabras por texto)", [100, 200, 400, 800, 1600],
        {"Levenshtein": lambda d: similitud.levenshtein(*d),
         "LCS": lambda d: similitud.subsecuencia_comun(*d),
         "Jaccard": lambda d: similitud.jaccard(*d),
         "Coseno TF-IDF": lambda d: similitud.coseno_tfidf(d[0], d[1], idf)},
        {"Levenshtein": 2, "LCS": 2, "Jaccard": 1, "Coseno TF-IDF": 1},
        lambda L: ([random.choice(vocabulario) for _ in range(L)],
                   [random.choice(vocabulario) for _ in range(L)]))

    # R2: n puntos al azar
    def matriz_al_azar(n):
        puntos = [[random.random() for _ in range(5)] for _ in range(n)]
        return agrupamiento.matriz_distancias(n, lambda i, j: similitud.coseno(puntos[i], puntos[j]))
    resumen += experimento(
        "agrupamiento", "n (documentos)", [20, 40, 80, 160],
        {nombre: (lambda D, c=c: agrupamiento.agrupar(D, c)) for c, nombre in agrupamiento.CRITERIOS.items()},
        {nombre: 3 for nombre in agrupamiento.CRITERIOS.values()},
        matriz_al_azar)

    # R3: corpus repetido hasta n caracteres
    patron = "artificial intelligence"
    resumen += experimento(
        "busqueda", "n (caracteres del texto)", [50_000, 100_000, 200_000, 400_000, 800_000],
        {nombre: (lambda t, nombre=nombre: busqueda.buscar(nombre, t, patron)) for nombre in busqueda.ALGORITMOS},
        {nombre: 1 for nombre in busqueda.ALGORITMOS},
        lambda n: (texto_corpus * (n // len(texto_corpus) + 1))[:n])

    # R3, peor caso: texto "aaa...a" y patrón "a" * m
    resumen += experimento(
        "busqueda_peor_caso", "m (largo del patrón)", [4, 8, 16, 32, 64],
        {nombre: (lambda m, nombre=nombre: busqueda.buscar(nombre, "a" * 20_000, "a" * m)) for nombre in busqueda.ALGORITMOS},
        {"KMP": 0, "Boyer-Moore": 1, "Rabin-Karp": 1},
        lambda m: m)

    with open("resultados/exponentes.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(resumen[0]))
        escritor.writeheader()
        escritor.writerows(resumen)
    print("\nResultados guardados en resultados/")


if __name__ == "__main__":
    main()
