"""Calcoli indipendenti e confronto con il valutatore della copia locale di BRIGHT."""
import importlib.util

import numpy as np
import pandas as pd

from .raster import leggi_maschera


def matrice_confusione(riferimento, previsione, n_classi):
    """Righe: classe reale; colonne: classe prevista."""
    riferimento, previsione = np.asarray(riferimento), np.asarray(previsione)
    if riferimento.shape != previsione.shape:
        raise ValueError("Le due maschere devono avere la stessa forma.")
    for v in (riferimento, previsione):
        if v.dtype.kind not in "iu" or np.any((v < 0) | (v >= n_classi)):
            raise ValueError("Attesi indici interi tra 0 e n_classi - 1.")
    matrice = np.zeros((n_classi, n_classi), dtype=np.int64)

    for classe_vera, classe_prevista in zip(
        riferimento.ravel(),
        previsione.ravel(),
    ):
        matrice[classe_vera, classe_prevista] += 1
    return matrice


def metriche_binarie(matrice):
    """Metriche della classe positiva; un rapporto 0/0 rimane NaN."""
    TN = int(matrice[0, 0])
    FP = int(matrice[0, 1])
    FN = int(matrice[1, 0])
    TP = int(matrice[1, 1])

    def rapporto(n, d):
        return n / d if d else np.nan

    risultati = {
        "Accuratezza complessiva": rapporto(TP + TN, matrice.sum()),
        "Precisione edificio": rapporto(TP, TP + FP),
        "Recall edificio": rapporto(TP, TP + FN),
        "IoU edificio": rapporto(TP, TP + FP + FN),
        "F1/Dice edificio": rapporto(2 * TP, 2 * TP + FP + FN),
    }
    return risultati


def carica_evaluator(percorso):
    """Importa la classe originale dal metrics.py del benchmark."""
    specifica = importlib.util.spec_from_file_location(
        "bright_metrics",
        percorso,
    )
    modulo_metriche = importlib.util.module_from_spec(specifica)
    specifica.loader.exec_module(modulo_metriche)
    return modulo_metriche.Evaluator


def f1_edificio_da_matrice(matrice_confusione):
    TP = int(matrice_confusione[1, 1])
    FP = int(matrice_confusione[0, 1])
    FN = int(matrice_confusione[1, 0])

    denominatore = 2 * TP + FP + FN

    if denominatore == 0:
        return np.nan

    return 2 * TP / denominatore


def metriche_per_classe(matrice_danno):
    """Conteggi, IoU e F1 diretti per classe; NaN se il denominatore è zero."""
    TP = np.diag(matrice_danno)
    FN = matrice_danno.sum(axis=1) - TP
    FP = matrice_danno.sum(axis=0) - TP

    with np.errstate(divide="ignore", invalid="ignore"):
        iou_diretta = TP / (TP + FP + FN)
        f1_diretto = 2 * TP / (2 * TP + FP + FN)
    return TP, FP, FN, iou_diretta, f1_diretto


def carica_maschere_controllo(cartella):
    """Legge le maschere a banda singola, esclude NoData e verifica le classi."""
    percorsi_maschere = sorted(
        cartella.glob("*_building_damage.tif")
    )

    if not percorsi_maschere:
        raise FileNotFoundError(
            f"Nessuna maschera trovata in: {cartella}"
        )

    maschere_locali = {}
    righe_controllo = []

    for percorso in percorsi_maschere:
        identificativo = percorso.name.removesuffix(
            "_building_damage.tif"
        )

        maschera = leggi_maschera(percorso)

        pixel_validi = maschera.compressed()

        if pixel_validi.size == 0:
            raise ValueError(
                f"{identificativo}: nessun pixel valido."
            )

        classi_presenti = np.unique(pixel_validi)

        if not np.all(np.isin(classi_presenti, [0, 1, 2, 3])):
            raise ValueError(
                f"{identificativo}: classi inattese {classi_presenti}"
            )

        maschere_locali[identificativo] = maschera

        righe_controllo.append({
            "id": identificativo,
            "righe": maschera.shape[0],
            "colonne": maschera.shape[1],
            "pixel_validi": pixel_validi.size,
            "classi_presenti": classi_presenti.tolist(),
        })

    controllo_maschere = pd.DataFrame(
        righe_controllo
    ).set_index("id")
    return maschere_locali, controllo_maschere


def previsioni_di_controllo(maschere_locali, evaluator):
    """Valuta previsioni artificiali sui soli pixel validi."""
    valutatori_controllo = {
        "Perfetta": evaluator(num_class=4),
        "Sempre sfondo": evaluator(num_class=4),
    }

    for identificativo, maschera in maschere_locali.items():
        # Sequenza delle etichette dei soli pixel validi.
        riferimento_valido = maschera.compressed().astype(np.int64)

        previsioni_controllo = {
            "Perfetta": riferimento_valido.copy(),
            "Sempre sfondo": np.zeros_like(riferimento_valido),
        }

        for caso, previsione_controllo in previsioni_controllo.items():
            valutatori_controllo[caso].add_batch(
                riferimento_valido,
                previsione_controllo,
            )

    # Calcoliamo le metriche dopo aver accumulato tutte le immagini.
    righe_risultati = []

    for caso, valutatore_controllo in valutatori_controllo.items():
        iou = valutatore_controllo.Intersection_over_Union()

        righe_risultati.append({
            "caso": caso,
            "accuratezza": valutatore_controllo.Pixel_Accuracy(),
            "IoU sfondo": iou[0],
            "IoU integro": iou[1],
            "IoU danneggiato": iou[2],
            "IoU distrutto": iou[3],
            "mIoU quattro classi": (
                valutatore_controllo.Mean_Intersection_over_Union()
            ),
        })

    risultati_controllo = pd.DataFrame(
        righe_risultati
    ).set_index("caso")
    return valutatori_controllo, risultati_controllo


def valuta_perfetta_per_immagine(maschere_locali, evaluator):
    """Valuta previsioni artificiali sui soli pixel validi."""
    righe_per_immagine = []

    for identificativo, maschera in maschere_locali.items():
        riferimento_valido = maschera.compressed().astype(np.int64)

        valutatore_singolo = evaluator(num_class=4)
        valutatore_singolo.add_batch(
            riferimento_valido,
            riferimento_valido.copy(),
        )

        iou = valutatore_singolo.Intersection_over_Union()

        righe_per_immagine.append({
            "id": identificativo,
            "IoU sfondo": iou[0],
            "IoU integro": iou[1],
            "IoU danneggiato": iou[2],
            "IoU distrutto": iou[3],
            "mIoU": valutatore_singolo.Mean_Intersection_over_Union(),
        })

    metriche_perfette_per_immagine = pd.DataFrame(
        righe_per_immagine
    ).set_index("id")
    return metriche_perfette_per_immagine
