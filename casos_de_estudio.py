# Caso de estudio del requerimiento 3: "generative AI" en el abstract del artículo 16.
# Guarda resultados/boyer_moore_caso.png

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from algoritmos import busqueda
from algoritmos.texto import cargar_corpus

PATRON = "generative AI"
ARTICULO = 16


def pasos_boyer_moore(texto, patron, ultima):
    # igual que bm_buscar, pero guarda (s, j, carácter, salto) de cada intento
    pasos, s, m = [], 0, len(patron)
    while s <= len(texto) - m:
        j = m - 1
        while j >= 0 and patron[j] == texto[s + j]:
            j -= 1
        if j < 0:
            pasos.append((s, -1, "", 1))
            s += 1
        else:
            salto = max(1, j - ultima.get(texto[s + j], -1))
            pasos.append((s, j, texto[s + j], salto))
            s += salto
    return pasos


def dibujar_boyer_moore(texto, patron, pasos, cuantos, ruta):
    ancho = pasos[cuantos - 1][0] + len(patron) + 1
    fig, ax = plt.subplots(figsize=(14, 0.55 * (cuantos + 2)))
    for x in range(ancho):
        ax.add_patch(Rectangle((x, 0), 1, 1, fill=False, lw=0.3))
        ax.text(x + 0.5, 0.5, texto[x], ha="center", va="center", family="monospace", fontsize=8)
        if x % 10 == 0:
            ax.text(x + 0.5, 1.3, str(x), ha="center", fontsize=6, color="gray")
    for fila, (s, j, caracter, salto) in enumerate(pasos[:cuantos], start=1):
        y = -fila
        for k in range(len(patron)):
            if j == -1 or k > j:
                color = "#bfe5bf"
            elif k == j:
                color = "#f4b6b6"
            else:
                color = "white"
            ax.add_patch(Rectangle((s + k, y), 1, 1, facecolor=color, lw=0.3, edgecolor="gray"))
            ax.text(s + k + 0.5, y + 0.5, patron[k], ha="center", va="center", family="monospace", fontsize=8)
        nota = "coincide todo" if j == -1 else f"falla con '{caracter}', salto {salto}"
        ax.text(s + len(patron) + 0.5, y + 0.5, f"s = {s}: {nota}", va="center", fontsize=7)
    ax.set_xlim(0, ancho + 18)
    ax.set_ylim(-cuantos - 0.5, 2)
    ax.axis("off")
    ax.set_title(f"Boyer-Moore: primeros {cuantos} intentos de '{patron}'", fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    corpus = {a["id"]: a for a in cargar_corpus("data/corpus.json")}
    texto = corpus[ARTICULO]["abstract"]
    print(f"Texto: abstract del artículo {ARTICULO}, n = {len(texto)}. Patrón: '{PATRON}', m = {len(PATRON)}\n")

    fallo = busqueda.kmp_preprocesar(PATRON)
    posiciones, comparaciones = busqueda.kmp_buscar(texto, PATRON, fallo)
    print("KMP  tabla de fallos:", fallo)
    print("     posiciones:", posiciones, " comparaciones:", comparaciones, "\n")

    ultima = busqueda.bm_preprocesar(PATRON)
    pasos = pasos_boyer_moore(texto, PATRON, ultima)
    print("Boyer-Moore  última posición:", ultima)
    for s, j, caracter, salto in pasos[:12]:
        print(f"     s = {s:4}  " + ("coincide todo" if j == -1 else f"falla en j = {j:2} con '{caracter}'  salto {salto}"))
    print("     intentos:", len(pasos), " comparaciones:", busqueda.bm_buscar(texto, PATRON, ultima)[1], "\n")
    dibujar_boyer_moore(texto, PATRON, pasos, 10, "resultados/boyer_moore_caso.png")

    for primo in [101, busqueda.PRIMO]:
        hash_patron, potencia, _ = busqueda.rk_preprocesar(PATRON, primo)
        print(f"Rabin-Karp  q = {primo}: hash del patrón = {hash_patron}, 256^(m-1) mod q = {potencia}")
        if primo == 101:
            h = 0
            for caracter in PATRON:
                print(f"     '{caracter}' código {ord(caracter):3}: ({h} · 256 + {ord(caracter)}) mod {primo} = "
                      f"{(h * 256 + ord(caracter)) % primo}")
                h = (h * 256 + ord(caracter)) % primo
        posiciones, comparaciones = busqueda.rk_buscar(texto, PATRON, (hash_patron, potencia, primo))
        h, colisiones = 0, []
        for k in range(len(PATRON)):
            h = (h * 256 + ord(texto[k])) % primo
        for s in range(len(texto) - len(PATRON) + 1):
            if h == hash_patron and texto[s:s + len(PATRON)] != PATRON:
                colisiones.append((s, texto[s:s + len(PATRON)]))
            if s < len(texto) - len(PATRON):
                h = ((h - ord(texto[s]) * potencia) * 256 + ord(texto[s + len(PATRON)])) % primo
        print("     colisiones:", colisiones)
        print("     posiciones:", posiciones, " comparaciones:", comparaciones, "\n")


if __name__ == "__main__":
    main()
