# Requerimiento 3: KMP, Boyer-Moore y Rabin-Karp.
# Cada algoritmo se divide en preprocesar(patron) y buscar(texto, patron, tabla).
# buscar devuelve (posiciones, comparaciones).


def kmp_preprocesar(patron):
    # fallo[q]: largo del prefijo propio más largo de patron[0..q] que también es sufijo
    fallo = [0] * len(patron)
    k = 0
    for q in range(1, len(patron)):
        while k > 0 and patron[k] != patron[q]:
            k = fallo[k - 1]
        if patron[k] == patron[q]:
            k += 1
        fallo[q] = k
    return fallo


def kmp_buscar(texto, patron, fallo):
    posiciones, comparaciones, q = [], 0, 0
    for i in range(len(texto)):
        comparaciones += 1
        while q > 0 and patron[q] != texto[i]:
            q = fallo[q - 1]
            comparaciones += 1
        if patron[q] == texto[i]:
            q += 1
        if q == len(patron):
            posiciones.append(i - len(patron) + 1)
            q = fallo[q - 1]
    return posiciones, comparaciones


def bm_preprocesar(patron):
    # última posición de cada carácter en el patrón (regla del carácter malo)
    ultima = {}
    for j in range(len(patron)):
        ultima[patron[j]] = j
    return ultima


def bm_buscar(texto, patron, ultima):
    n, m = len(texto), len(patron)
    posiciones, comparaciones, s = [], 0, 0
    while s <= n - m:
        j = m - 1
        while j >= 0:
            comparaciones += 1
            if patron[j] != texto[s + j]:
                break
            j -= 1
        if j < 0:
            posiciones.append(s)
            s += 1
        else:
            s += max(1, j - ultima.get(texto[s + j], -1))
    return posiciones, comparaciones


BASE = 256
PRIMO = 1_000_000_007


def rk_preprocesar(patron, primo=PRIMO):
    hash_patron = 0
    potencia = 1    # 256^(m-1) mod primo
    for k in range(len(patron)):
        hash_patron = (hash_patron * BASE + ord(patron[k])) % primo
        if k > 0:
            potencia = (potencia * BASE) % primo
    return hash_patron, potencia, primo


def rk_buscar(texto, patron, tabla):
    hash_patron, potencia, primo = tabla
    n, m = len(texto), len(patron)
    posiciones, comparaciones = [], 0
    if m > n:
        return posiciones, comparaciones

    hash_ventana = 0
    for k in range(m):
        hash_ventana = (hash_ventana * BASE + ord(texto[k])) % primo

    for s in range(n - m + 1):
        if hash_ventana == hash_patron:
            # mismo hash: se comparan los caracteres porque puede ser una colisión
            iguales = True
            for k in range(m):
                comparaciones += 1
                if texto[s + k] != patron[k]:
                    iguales = False
                    break
            if iguales:
                posiciones.append(s)
        if s < n - m:
            # rolling hash: quitar texto[s] y agregar texto[s + m]
            sin_primero = hash_ventana - ord(texto[s]) * potencia
            hash_ventana = (sin_primero * BASE + ord(texto[s + m])) % primo
    return posiciones, comparaciones


ALGORITMOS = {
    "KMP": (kmp_preprocesar, kmp_buscar),
    "Boyer-Moore": (bm_preprocesar, bm_buscar),
    "Rabin-Karp": (rk_preprocesar, rk_buscar),
}


def buscar(nombre, texto, patron):
    preprocesar, buscar_en_texto = ALGORITMOS[nombre]
    return buscar_en_texto(texto, patron, preprocesar(patron))
