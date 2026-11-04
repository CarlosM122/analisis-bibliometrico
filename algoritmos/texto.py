import json

# stopwords del inglés
PALABRAS_VACIAS = set("""
a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for from
further had has have having he her here hers him his how i if in into is it its itself just
me more most my no nor not now of off on once only or other our ours out over own same she
should so some such than that the their theirs them then there these they this those through
to too under until up very was we were what when where which while who whom why will with
would you your however thus therefore whether may might must via within without among upon
et al one two three four five six seven eight nine ten
""".split())


def cargar_corpus(ruta):
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def limpiar(texto):
    # minúsculas, solo letras a-z, sin stopwords ni palabras de una letra
    texto = texto.lower()
    solo_letras = ""
    for caracter in texto:
        if "a" <= caracter <= "z":
            solo_letras += caracter
        else:
            solo_letras += " "
    palabras = []
    for palabra in solo_letras.split():
        if palabra not in PALABRAS_VACIAS and len(palabra) > 1:
            palabras.append(palabra)
    return palabras
