"""Lettura delle caratteristiche dei raster BRIGHT."""

import rasterio
import numpy as np

def descrivi_raster(percorso):
    """Restituisce metadati e intervalli dei valori validi per banda."""
    with rasterio.open(percorso) as dataset:
        valori = dataset.read(masked=True)

        statistiche_bande = []

        for indice, banda in enumerate(valori, start=1):
            numero_validi = int(banda.count())

            statistiche_bande.append({
                "banda": indice,
                "pixel_validi": numero_validi,
                "minimo": (
                    banda.min().item() if numero_validi else None
                ),
                "massimo": (
                    banda.max().item() if numero_validi else None
                ),
            })

        return {
            "forma": valori.shape,
            "tipi": dataset.dtypes,
            "crs": dataset.crs,
            "risoluzione": dataset.res,
            "nodata": dataset.nodata,
            "bande": statistiche_bande,
        }

def leggi_maschera(percorso):
    """Legge una maschera a una banda, conservando i pixel non validi."""
    with rasterio.open(percorso) as dataset:
        if dataset.count != 1:
            raise ValueError(
                f"Attesa una maschera a una banda: {percorso}"
            )

        return dataset.read(1, masked=True)


def conta_classi(maschera, codici=(0, 1, 2, 3)):
    """Conta le classi sui soli pixel validi della maschera."""
    pixel_validi = np.ma.asarray(maschera).compressed()

    if pixel_validi.size == 0:
        raise ValueError("La maschera non contiene pixel validi.")

    valori, conteggi = np.unique(pixel_validi, return_counts=True)

    inattesi = set(valori.tolist()) - set(codici)

    if inattesi:
        raise ValueError(f"Etichette non previste: {inattesi}")

    conteggi_presenti = dict(zip(valori.tolist(), conteggi.tolist()))

    return {
        "totale_pixel_validi": int(pixel_validi.size),
        "conteggi": {
            codice: int(conteggi_presenti.get(codice, 0))
            for codice in codici
        },
    }