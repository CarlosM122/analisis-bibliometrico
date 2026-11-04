# Requerimiento 2: agrupamiento jerárquico aglomerativo.

import math

CRITERIOS = {"simple": "Enlace simple", "completo": "Enlace completo", "promedio": "Enlace promedio"}


def matriz_distancias(n, similitud_entre):
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distancia = max(0.0, 1 - similitud_entre(i, j))
            D[i][j] = distancia
            D[j][i] = distancia
    return D


def distancia_entre_grupos(D, grupo_a, grupo_b, criterio):
    distancias = []
    for x in grupo_a:
        for y in grupo_b:
            distancias.append(D[x][y])
    if criterio == "simple":
        return min(distancias)
    if criterio == "completo":
        return max(distancias)
    return sum(distancias) / len(distancias)


def agrupar(D, criterio):
    # devuelve la lista de uniones (grupo_a, grupo_b, distancia)
    grupos = [[i] for i in range(len(D))]
    uniones = []
    while len(grupos) > 1:
        mejor = None
        for i in range(len(grupos)):
            for j in range(i + 1, len(grupos)):
                d = distancia_entre_grupos(D, grupos[i], grupos[j], criterio)
                if mejor is None or d < mejor[0]:
                    mejor = (d, i, j)
        d, i, j = mejor
        uniones.append((grupos[i], grupos[j], d))
        nuevo = grupos[i] + grupos[j]
        grupos = [g for k, g in enumerate(grupos) if k != i and k != j]
        grupos.append(nuevo)
    return uniones


def correlacion_cofenetica(D, uniones):
    # Pearson entre las distancias reales y la altura a la que se unen en el árbol
    n = len(D)
    altura = [[0.0] * n for _ in range(n)]
    for grupo_a, grupo_b, d in uniones:
        for x in grupo_a:
            for y in grupo_b:
                altura[x][y] = d
                altura[y][x] = d

    reales, arbol = [], []
    for i in range(n):
        for j in range(i + 1, n):
            reales.append(D[i][j])
            arbol.append(altura[i][j])

    media_r = sum(reales) / len(reales)
    media_a = sum(arbol) / len(arbol)
    arriba = sum((reales[k] - media_r) * (arbol[k] - media_a) for k in range(len(reales)))
    abajo_r = sum((x - media_r) ** 2 for x in reales)
    abajo_a = sum((x - media_a) ** 2 for x in arbol)
    return arriba / math.sqrt(abajo_r * abajo_a)


def dibujar_dendrograma(uniones, nombres, titulo, ax):
    orden = uniones[-1][0] + uniones[-1][1]
    fila = {doc: posicion for posicion, doc in enumerate(orden)}

    lugar = {}    # grupo -> (altura, fila) donde se dibuja
    for doc in orden:
        lugar[str([doc])] = (0.0, fila[doc])
    for grupo_a, grupo_b, d in uniones:
        xa, ya = lugar[str(grupo_a)]
        xb, yb = lugar[str(grupo_b)]
        ax.plot([xa, d, d, xb], [ya, ya, yb, yb], color="#2f5d8a")
        lugar[str(grupo_a + grupo_b)] = (d, (ya + yb) / 2)

    ax.set_yticks(range(len(orden)))
    ax.set_yticklabels([nombres[doc] for doc in orden], fontsize=8)
    ax.set_xlabel("Distancia (1 - coseno)")
    ax.set_title(titulo)
