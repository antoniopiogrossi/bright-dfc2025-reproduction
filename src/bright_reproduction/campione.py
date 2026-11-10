"""Disponibilità dei file locali e statistiche sui pixel validi."""
import numpy as np
import pandas as pd
from .raster import leggi_maschera


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
