"""Disponibilità dei file locali e statistiche sui pixel validi."""
import numpy as np
import pandas as pd
from pathlib import Path
from .geospatial import verifica_griglie
from .raster import leggi_maschera
from .raster import descrivi_raster
from .raster import maschera_validita_congiunta
from .geospatial import (
    estensione_geografica,
    sovrapposizione_geografica
)


def controlla_disponibilita(cartella, split):
    """Verifica la presenza dei tre file per ogni voce nelle liste."""
    if not cartella.is_dir():
        raise FileNotFoundError(
            f"Cartella dei dati non trovata: {cartella}"
        )

    controlli = []

    for nome_split, identificativi in split.items():
        for identificativo in identificativi:

            percorso_ottico = (
                cartella / "pre-event"
                / f"{identificativo}_pre_disaster.tif"
            )
            percorso_sar = (
                cartella / "post-event"
                / f"{identificativo}_post_disaster.tif"
            )
            percorso_maschera = (
                cartella / "target"
                / f"{identificativo}_building_damage.tif"
            )

            controlli.append({
                "split": nome_split,
                "id": identificativo,
                "ottico_presente": percorso_ottico.is_file(),
                "sar_presente": percorso_sar.is_file(),
                "maschera_presente": percorso_maschera.is_file(),
            })

    disponibilita = pd.DataFrame(controlli)

    colonne_file = [
        "ottico_presente",
        "sar_presente",
        "maschera_presente",
    ]

    disponibilita["completo"] = disponibilita[colonne_file].all(axis=1)

    riepilogo = (
        disponibilita.groupby("split")
        .agg(
            esempi_in_lista=("id", "size"),
            ottici_presenti=("ottico_presente", "sum"),
            sar_presenti=("sar_presente", "sum"),
            maschere_presenti=("maschera_presente", "sum"),
            esempi_completi=("completo", "sum"),
        )
        .reindex(["train", "holdout", "val", "test"])
    )
    return disponibilita, riepilogo


CLASSI = {
    0: "Sfondo",
    1: "Edificio integro",
    2: "Edificio danneggiato",
    3: "Edificio distrutto",
}


def statistiche_classi(cartella, id_locali):
    """Conta i pixel delle quattro classi per esempio e nel campione."""
    if not id_locali:
        raise ValueError("Nessun esempio completo disponibile.")

    righe_statistiche = []

    for identificativo in id_locali:
        percorso = (
            cartella / "target"
            / f"{identificativo}_building_damage.tif"
        )

        maschera = leggi_maschera(percorso)

        # Conserviamo soltanto i pixel validi.
        # Lo sfondo 0 resta incluso se non è dichiarato NoData.
        pixel_validi = maschera.compressed()

        valori, conteggi = np.unique(pixel_validi, return_counts=True)

        inattesi = set(valori.tolist()) - set(CLASSI)
        if inattesi:
            raise ValueError(
                f"{identificativo}: etichette inattese {inattesi}"
            )

        conteggi_per_classe = dict(zip(valori.tolist(), conteggi.tolist()))

        riga = {"id": identificativo}

        for codice, descrizione in CLASSI.items():
            riga[descrizione] = conteggi_per_classe.get(codice, 0)

        righe_statistiche.append(riga)

    statistiche_locali = pd.DataFrame(righe_statistiche).set_index("id")


    # Sommiamo i conteggi di tutte le maschere analizzate
    totali_classi = statistiche_locali.sum(axis=0)
    totale_pixel = int(totali_classi.sum())

    if totale_pixel == 0:
        raise ValueError("Le maschere non contengono pixel validi.")

    riepilogo_classi = pd.DataFrame({
        "pixel": totali_classi,
        "percentuale": 100 * totali_classi / totale_pixel,
    })
    return statistiche_locali, riepilogo_classi, totale_pixel


def statistiche_edifici(conteggi):
    """Raggruppa i conteggi delle classi 1–3 come edificio."""
    tabella = pd.DataFrame(index=conteggi.index)
    tabella["pixel_sfondo"] = conteggi[CLASSI[0]]
    tabella["pixel_edificio"] = conteggi[[CLASSI[i] for i in (1, 2, 3)]].sum(axis=1)
    totale = tabella.sum(axis=1)
    tabella["percentuale_edificio"] = 100 * tabella["pixel_edificio"] / totale.where(totale > 0)
    return tabella


def controlla_griglie_campione(
    cartella,
    identificativi,
    tolleranza=1e-6
):
    """Verifica la compatibilità delle griglie di ogni esempio."""

    cartella = Path(cartella)
    risultati = []

    for identificativo in identificativi:

        percorsi = {
            "ottico": (
                cartella / "pre-event"
                / f"{identificativo}_pre_disaster.tif"
            ),
            "sar": (
                cartella / "post-event"
                / f"{identificativo}_post_disaster.tif"
            ),
            "maschera": (
                cartella / "target"
                / f"{identificativo}_building_damage.tif"
            )
        }

        try:
            controlli = verifica_griglie(
                percorsi,
                nome_riferimento="ottico",
                tolleranza=tolleranza
            )

            scarto = max(
                dati["scarto_massimo_pixel"]
                for dati in controlli.values()
            )

            risultati.append({
                "id": identificativo,
                "compatibile": True,
                "scarto_massimo_pixel": scarto,
                "errore": None
            })

        except ValueError as errore:

            risultati.append({
                "id": identificativo,
                "compatibile": False,
                "scarto_massimo_pixel": None,
                "errore": str(errore)
            })

    return pd.DataFrame(risultati).set_index("id")

def controlla_sovrapposizioni(cartella, campione):
    """Cerca sovrapposizioni geografiche tra train e val."""

    cartella = Path(cartella)
    estensioni = {}

    for split in ("train", "val"):

        estensioni[split] = {}

        for id_ in campione[split]:

            percorso = (
                cartella / split / "pre-event"
                / f"{id_}_pre_disaster.tif"
            )

            estensioni[split][id_] = (
                estensione_geografica(percorso)
            )

    risultati = []

    for id_train, limiti_train in estensioni["train"].items():

        for id_val, limiti_val in estensioni["val"].items():

            if sovrapposizione_geografica(
                limiti_train, limiti_val
            ):

                risultati.append({
                    "id_train": id_train,
                    "id_val": id_val
                })

    return pd.DataFrame(
        risultati,
        columns=["id_train", "id_val"]
    )


def controlla_raster_campione(cartella, identificativi):
    """Controlla dimensioni, bande e pixel validi dei raster."""

    cartella = Path(cartella)
    risultati = []

    suffissi = {
        "pre-event": "_pre_disaster.tif",
        "post-event": "_post_disaster.tif",
        "target": "_building_damage.tif"
    }

    for id_ in identificativi:

        for categoria, suffisso in suffissi.items():

            percorso = (
                cartella / categoria
                / f"{id_}{suffisso}"
            )

            info = descrivi_raster(percorso)

            bande, h, w = info["forma"]

            invalidi = [
                h * w - banda["pixel_validi"]
                for banda in info["bande"]
            ]

            risultati.append({
                "id": id_,
                "categoria": categoria,
                "bande": bande,
                "altezza": h,
                "larghezza": w,
                "dtype": str(info["tipi"]),
                "nodata": info["nodata"],
                "invalidi_per_banda": invalidi,
                "totale_valori_invalidi": sum(invalidi)
            })

    return pd.DataFrame(risultati)


def statistiche_validita_congiunta(cartella, identificativi):
    """Confronta i criteri di validità multimodale."""

    cartella = Path(cartella)
    risultati = []

    for id_ in identificativi:

        ottico = (
            cartella / "pre-event"
            / f"{id_}_pre_disaster.tif"
        )

        sar = (
            cartella / "post-event"
            / f"{id_}_post_disaster.tif"
        )

        target = (
            cartella / "target"
            / f"{id_}_building_damage.tif"
        )

        rigorosa = maschera_validita_congiunta(
            ottico, sar, target,
            ottico_rigoroso=True
        )

        alternativa = maschera_validita_congiunta(
            ottico, sar, target,
            ottico_rigoroso=False
        )

        risultati.append({
            "id": id_,
            "pixel_totali": rigorosa.size,
            "validi_rigoroso": int(rigorosa.sum()),
            "validi_alternativo": int(alternativa.sum()),
            "percentuale_rigoroso": 100 * rigorosa.mean(),
            "percentuale_alternativo": 100 * alternativa.mean()
        })

    return pd.DataFrame(risultati).set_index("id")
