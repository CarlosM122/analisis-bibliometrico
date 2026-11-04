# Requerimiento 1: similitud entre dos abstracts (listas de palabras).

import math


def levenshtein(a, b):
    m, n = len(a), len(b)
    D = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        D[i][0] = i
    for j in range(n + 1):
        D[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            costo = 0 if a[i - 1] == b[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1,          # borrar
                          D[i][j - 1] + 1,          # insertar
                          D[i - 1][j - 1] + costo)  # cambiar
    distancia = D[m][n]
    similitud = 1 - distancia / max(m, n)
    return similitud, D


def subsecuencia_comun(a, b):
    m, n = len(a), len(b)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    similitud = 2 * L[m][n] / (m + n)
    return similitud, L


def jaccard(a, b):
    conjunto_a = {}
    for palabra in a:
        conjunto_a[palabra] = True
    conjunto_b = {}
    for palabra in b:
        conjunto_b[palabra] = True

    comunes = []
    for palabra in conjunto_a:
        if palabra in conjunto_b:
            comunes.append(palabra)

    total_distintas = len(conjunto_a) + len(conjunto_b) - len(comunes)
    return len(comunes) / total_distintas, comunes


def calcular_idf(documentos):
    # idf = ln(N / df)
    df = {}
    for palabras in documentos:
        vistas = {}
        for palabra in palabras:
            vistas[palabra] = True
        for palabra in vistas:
            df[palabra] = df.get(palabra, 0) + 1
    N = len(documentos)
    idf = {}
    for palabra in df:
        idf[palabra] = math.log(N / df[palabra])
    return idf


def vector_tfidf(palabras, idf):
    conteo = {}
    for palabra in palabras:
        conteo[palabra] = conteo.get(palabra, 0) + 1
    vector = {}
    for palabra in conteo:
        vector[palabra] = conteo[palabra] / len(palabras) * idf[palabra]
    return vector


def coseno_tfidf(a, b, idf):
    u = vector_tfidf(a, idf)
    v = vector_tfidf(b, idf)
    producto = 0.0
    for palabra in u:
        if palabra in v:
            producto += u[palabra] * v[palabra]
    norma_u = math.sqrt(sum(peso * peso for peso in u.values()))
    norma_v = math.sqrt(sum(peso * peso for peso in v.values()))
    if norma_u == 0 or norma_v == 0:
        return 0.0, u, v
    return producto / (norma_u * norma_v), u, v


# Para los embeddings (listas de números)
def coseno(u, v):
    producto = 0.0
    norma_u = 0.0
    norma_v = 0.0
    for k in range(len(u)):
        producto += u[k] * v[k]
        norma_u += u[k] * u[k]
        norma_v += v[k] * v[k]
    return producto / (math.sqrt(norma_u) * math.sqrt(norma_v))


def euclidiana(u, v):
    # se normalizan los vectores a longitud 1 para que la distancia quede entre 0 y 2
    norma_u = math.sqrt(sum(x * x for x in u))
    norma_v = math.sqrt(sum(x * x for x in v))
    suma = 0.0
    for k in range(len(u)):
        diferencia = u[k] / norma_u - v[k] / norma_v
        suma += diferencia * diferencia
    return math.sqrt(suma)
