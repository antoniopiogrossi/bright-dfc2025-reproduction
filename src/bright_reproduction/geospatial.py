"""Controlli delle griglie geografiche dei raster."""

import rasterio


def verifica_griglie(percorsi, nome_riferimento, tolleranza=1e-6):
    """Verifica forma, CRS e trasformazioni rispetto a un raster.

    Restituisce i risultati per ciascun file.
    Solleva ValueError se una griglia non è compatibile.
    La tolleranza è espressa in pixel.
    """
    if tolleranza < 0:
        raise ValueError("La tolleranza deve essere non negativa.")

    if nome_riferimento not in percorsi:
        raise ValueError("Il raster di riferimento non è nei percorsi.")

    metadati = {}

    for nome, percorso in percorsi.items():
        with rasterio.open(percorso) as dataset:
            if dataset.crs is None:
                raise ValueError(f"{nome}: CRS assente.")

            metadati[nome] = {
                "shape": dataset.shape,
                "crs": dataset.crs,
                "transform": dataset.transform,
            }

    riferimento = metadati[nome_riferimento]
    altezza, larghezza = riferimento["shape"]

    angoli = [
        (0, 0),
        (larghezza, 0),
        (0, altezza),
        (larghezza, altezza),
    ]

    trasformazione_inversa = ~riferimento["transform"]
    risultati = {}

    for nome, dati in metadati.items():
        stessa_forma = dati["shape"] == riferimento["shape"]
        stesso_crs = dati["crs"] == riferimento["crs"]

        if not stessa_forma or not stesso_crs:
            raise ValueError(
                f"{nome}: griglia incompatibile con {nome_riferimento}. "
                f"Stessa forma: {stessa_forma}; stesso CRS: {stesso_crs}."
            )

        scarti = []

        for colonna, riga in angoli:
            x, y = dati["transform"] * (colonna, riga)
            col_rif, riga_rif = trasformazione_inversa * (x, y)

            scarti.append(
                max(abs(col_rif - colonna), abs(riga_rif - riga))
            )

        scarto_massimo = max(scarti)

        if scarto_massimo > tolleranza:
            raise ValueError(
                f"{nome}: scarto di {scarto_massimo:.3e} pixel, "
                f"superiore alla tolleranza di {tolleranza:.3e}."
            )

        risultati[nome] = {
            "stessa_forma": stessa_forma,
            "stesso_crs": stesso_crs,
            "scarto_massimo_pixel": scarto_massimo,
        }

    return risultati