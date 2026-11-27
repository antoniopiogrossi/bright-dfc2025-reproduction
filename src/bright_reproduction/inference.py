
"""Inferenza e ricostruzione delle mappe di danno BRIGHT."""

import numpy as np
import torch

from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def predici_immagine(
    modello,
    dataset,
    identificativo,
    device,
    batch_size=4
):
    """Ricostruisce la previsione completa da patch non sovrapposte."""

    if dataset.training:
        raise ValueError(
            "Utilizzare un Dataset con copertura completa."
        )

    indici = [
        i
        for i, (id_, x, y) in enumerate(dataset.patch)
        if id_ == identificativo
    ]

    if not indici:
        raise ValueError(
            f"Identificativo non trovato: {identificativo}"
        )

    dimensione = dataset.dimensione

    larghezza = max(
        dataset.patch[i][1] for i in indici
    ) + dimensione

    altezza = max(
        dataset.patch[i][2] for i in indici
    ) + dimensione

    previsione = np.full(
        (altezza, larghezza),
        dataset.ignore_index,
        dtype=np.int64
    )

    riferimento = previsione.copy()

    copertura = np.zeros(
        (altezza, larghezza),
        dtype=bool
    )

    loader = DataLoader(
        Subset(dataset, indici),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    modello.eval()

    for batch in loader:

        inputs = batch["input"].to(device)

        logits = modello(inputs)

        predizioni = logits.argmax(dim=1).cpu().numpy()

        targets = batch["target"].numpy()

        for i in range(len(targets)):

            x = int(batch["x"][i])
            y = int(batch["y"][i])

            regione = np.s_[
                y:y+dimensione,
                x:x+dimensione
            ]

            if copertura[regione].any():
                raise RuntimeError("Patch sovrapposte.")

            pred = predizioni[i].copy()

            pred[
                targets[i] == dataset.ignore_index
            ] = dataset.ignore_index

            previsione[regione] = pred
            riferimento[regione] = targets[i]
            copertura[regione] = True

    if not copertura.all():
        raise RuntimeError("Copertura incompleta.")

    return {
        "id": identificativo,
        "previsione": previsione,
        "riferimento": riferimento,
        "validita": riferimento != dataset.ignore_index
    }
