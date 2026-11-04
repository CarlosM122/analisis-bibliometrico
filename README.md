# Análisis de algoritmos en el contexto de la bibliometría

Proyecto final de Análisis de Algoritmos 2026-2. Una página web que analiza los abstracts de 20 artículos sobre inteligencia artificial en educación.

## Qué hay en cada archivo

```
analisis_bibliometrico/
├── app.py                  la página web (Streamlit)
├── extraer_corpus.py       paso 1: saca título, autores y abstract de los PDF -> data/corpus.json
├── generar_embeddings.py   paso 2: vectores de los modelos de IA -> data/embeddings.json
├── experimentos.py         mide los tiempos y los compara con el Big O -> resultados/
├── casos_de_estudio.py     caso de estudio del Requerimiento 3, paso a paso -> resultados/boyer_moore_caso.png
├── algoritmos/
│   ├── texto.py            leer el corpus y limpiar el texto
│   ├── similitud.py        Requerimiento 1: Levenshtein, LCS, Jaccard, coseno TF-IDF, coseno y euclidiana
│   ├── agrupamiento.py     Requerimiento 2: enlace simple, completo y promedio, dendrograma
│   └── busqueda.py         Requerimiento 3: KMP, Boyer-Moore, Rabin-Karp
├── tests/test_algoritmos.py  compara nuestros resultados con SciPy, NumPy y str.find
├── articulos/              los 20 PDF
├── data/                   corpus.json (lo genera extraer_corpus.py), embeddings.json,
│                           metadatos.json (títulos y autores copiados a mano, solo para comprobar)
└── resultados/             tablas y gráficos de experimentos.py
```

Todos los algoritmos están escritos a mano con listas, diccionarios y ciclos. No se usan `find()`, `index()`, SciPy ni scikit-learn dentro de los algoritmos; esas librerías solo aparecen en las pruebas.

## Cómo ejecutarlo (en la terminal, dentro de la carpeta del proyecto)

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Otros comandos:

| Comando | Qué hace |
| --- | --- |
| `python extraer_corpus.py` | Saca título, autores y abstract de los 20 PDF y escribe `data/corpus.json` |
| `python extraer_corpus.py --verificar` | Vuelve a sacar todo y lo compara: abstracts contra `data/corpus.json`, títulos y autores contra `data/metadatos.json` (debe decir 20 de 20 en los tres) |
| `python generar_embeddings.py` | Recalcula los vectores de IA (antes: `python -m pip install -r requirements-ia.txt`) |
| `python experimentos.py` | Mide los tiempos y guarda tablas y gráficos en `resultados/` |
| `python casos_de_estudio.py` | Imprime las tablas del caso de estudio de búsqueda ("generative AI" en el abstract 16) y guarda el gráfico de Boyer-Moore |
| `python tests/test_algoritmos.py` | Pruebas (antes: `python -m pip install -r requirements-pruebas.txt`) |