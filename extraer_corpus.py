# Extrae título, autores y abstract de los PDF de articulos/ y guarda data/corpus.json.
#   python extraer_corpus.py              genera data/corpus.json
#   python extraer_corpus.py --verificar  compara con data/corpus.json y data/metadatos.json

import json
import logging
import os
import sys
import unicodedata

import pdfplumber

logging.getLogger("pdfminer").setLevel(logging.ERROR)

PALABRAS_DE_FIN = ("keywords", "key words", "k e y w o r d s", "1. introduction", "introduction", "©")


def agrupar_en_renglones(palabras, tolerancia=3):
    palabras = sorted(palabras, key=lambda p: (p["top"], p["x0"]))
    renglones = []
    for p in palabras:
        if renglones and abs(renglones[-1][0]["top"] - p["top"]) < tolerancia:
            renglones[-1].append(p)
        else:
            renglones.append([p])
    return [sorted(r, key=lambda p: p["x0"]) for r in renglones]


def parte_de_la_columna(renglon, x_columna):
    # separa el renglón en columnas (huecos de más de 15 puntos) y devuelve la del abstract
    pedazos = [[renglon[0]]]
    for p in renglon[1:]:
        if p["x0"] - pedazos[-1][-1]["x1"] > 15:
            pedazos.append([p])
        else:
            pedazos[-1].append(p)
    for pedazo in pedazos:
        if abs(pedazo[0]["x0"] - x_columna) <= 25:
            return pedazo
    return None


def buscar_titulo_abstract(pdf):
    for pagina in pdf.pages[:2]:
        for renglon in agrupar_en_renglones(pagina.extract_words(x_tolerance=1)):
            for k, p in enumerate(renglon):
                siguientes = "".join(x["text"] for x in renglon[k:k + 8]).upper()
                if p["text"][0] in "Aa" and siguientes.startswith("ABSTRACT"):
                    return pagina, p
    raise ValueError("No se encontró la palabra Abstract")


def leer_renglones_del_abstract(ruta_pdf):
    with pdfplumber.open(ruta_pdf) as pdf:
        pagina, encabezado = buscar_titulo_abstract(pdf)
        izquierda, _, derecha, abajo = pagina.bbox
        recorte = pagina.crop((max(izquierda, encabezado["x0"] - 2), encabezado["top"] - 1, derecha, abajo))

        renglones, alturas = [], []
        for renglon in agrupar_en_renglones(recorte.extract_words(x_tolerance=1)):
            pedazo = parte_de_la_columna(renglon, encabezado["x0"])
            if pedazo is None:
                continue
            texto = " ".join(p["text"] for p in pedazo)
            if renglones and texto.lower().startswith(PALABRAS_DE_FIN):
                break
            # un espacio entre renglones mucho mayor que el normal indica que terminó el abstract
            if len(alturas) >= 3:
                espacios = sorted(alturas[i + 1] - alturas[i] for i in range(len(alturas) - 1))
                espacio_normal = espacios[len(espacios) // 2]
                if pedazo[0]["top"] - alturas[-1] > 1.6 * espacio_normal:
                    break
            renglones.append(texto)
            alturas.append(pedazo[0]["top"])

    # quitar la palabra "Abstract" del primer renglón
    primero = renglones[0]
    letras = 0
    while letras < 8:
        if primero[0].isalpha():
            letras += 1
        primero = primero[1:]
    renglones[0] = primero.lstrip(": ").strip()
    return [r for r in renglones if r]


def texto_del_renglon(palabras):
    # sin espacio si las palabras están pegadas (ligaduras como "fi")
    texto = palabras[0]["text"]
    for anterior, p in zip(palabras, palabras[1:]):
        if p["x0"] - anterior["x1"] < 0.5:
            texto += p["text"]
        else:
            texto += " " + p["text"]
    return texto


def tamano(renglon):
    return max(p["size"] for p in renglon)


def agrupar_en_bloques(renglones):
    # renglones seguidos con el mismo tamaño de letra
    bloques = []
    for renglon in renglones:
        if (bloques and abs(tamano(bloques[-1][-1]) - tamano(renglon)) < 0.3
                and renglon[0]["top"] - bloques[-1][-1][0]["top"] < 2.2 * tamano(renglon)):
            bloques[-1].append(renglon)
        else:
            bloques.append([renglon])
    return bloques


def texto_del_bloque(bloque):
    # ignora la letra pequeña (marcas de afiliación, notas al margen)
    grande = tamano(bloque[0])
    renglones = [[p for p in renglon if p["size"] >= 0.8 * grande] for renglon in bloque]
    return " ".join(texto_del_renglon(r) for r in renglones if r)


def leer_titulo_y_autores(ruta_pdf):
    # título: último bloque de letra grande antes de "Abstract"; autores: el bloque siguiente
    with pdfplumber.open(ruta_pdf) as pdf:
        pagina, abstract = buscar_titulo_abstract(pdf)
        limite = abstract["top"] if pagina.page_number == 1 else pdf.pages[0].height
        palabras = [p for p in pdf.pages[0].extract_words(x_tolerance=1, extra_attrs=["size"])
                    if p["top"] < limite]
    bloques = agrupar_en_bloques(agrupar_en_renglones(palabras, tolerancia=10))

    k = None
    for i, bloque in enumerate(bloques):
        if tamano(bloque[0]) >= 13 and len(texto_del_bloque(bloque).split()) >= 4:
            k = i
    titulo = texto_del_bloque(bloques[k]).replace("‑", "-")

    texto = texto_del_bloque(bloques[k + 1])
    texto = unicodedata.normalize("NFC", texto.replace("´", "́"))
    for separador in [" and ", "|", "·", ";"]:
        texto = texto.replace(separador, ",")
    autores = []
    for nombre in texto.split(","):
        partes = [w for w in nombre.split() if w.strip("*0123456789") != ""]
        if partes:
            autores.append(" ".join(partes))
    return " ".join(titulo.split()), autores


def palabras_completas(todos_los_renglones):
    # palabras que aparecen sin cortar en algún abstract
    vistas = {}
    for renglones in todos_los_renglones:
        for i, renglon in enumerate(renglones):
            palabras = renglon.split()
            for k, palabra in enumerate(palabras):
                cortada_al_final = k == len(palabras) - 1 and palabra.endswith("-")
                cortada_al_inicio = k == 0 and i > 0 and renglones[i - 1].endswith("-")
                if not cortada_al_final and not cortada_al_inicio:
                    for parte in palabra.replace("-", " ").split():
                        vistas[parte.strip(".,;:()").lower()] = True
    return vistas


def unir_renglones(renglones, completas):
    # "AI-" + "driven" -> AI-driven, pero "edu-" + "cation" -> education
    texto = renglones[0]
    for renglon in renglones[1:]:
        if texto.endswith("-"):
            siguiente = renglon.split()[0].strip(".,;:()").lower()
            if siguiente in completas:
                texto = texto + renglon
            else:
                texto = texto[:-1] + renglon
        elif texto.endswith("/") or texto.endswith("—"):
            texto = texto + renglon
        else:
            texto = texto + " " + renglon
    texto = texto.replace("well- defined", "well-defined")
    return " ".join(texto.split())


def main():
    archivos = sorted(f for f in os.listdir("articulos") if f.endswith(".pdf"))

    articulos, todos = [], []
    for numero, archivo in enumerate(archivos, start=1):
        titulo, autores = leer_titulo_y_autores("articulos/" + archivo)
        articulos.append({"id": numero, "archivo": archivo, "titulo": titulo, "autores": autores})
        todos.append(leer_renglones_del_abstract("articulos/" + archivo))
    completas = palabras_completas(todos)

    corpus = []
    for articulo, renglones in zip(articulos, todos):
        corpus.append({**articulo, "abstract": unir_renglones(renglones, completas)})

    if "--verificar" in sys.argv:
        with open("data/corpus.json", encoding="utf-8") as f:
            actual = json.load(f)
        with open("data/metadatos.json", encoding="utf-8") as f:
            a_mano = json.load(f)
        for campo, referencia in [("abstract", actual), ("titulo", a_mano), ("autores", a_mano)]:
            iguales = sum(1 for a, b in zip(corpus, referencia) if a[campo] == b[campo])
            print(f"{campo}: {iguales} de {len(corpus)} iguales")
            for a, b in zip(corpus, referencia):
                if a[campo] != b[campo]:
                    print("  Diferente: artículo", a["id"], a[campo])
        return

    with open("data/corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Guardado data/corpus.json con {len(corpus)} artículos")


if __name__ == "__main__":
    main()
