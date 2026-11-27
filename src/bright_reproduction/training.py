
"""Funzioni per l'addestramento dei modelli."""

import torch
import numpy as np
import time
import shutil
import pandas as pd

from pathlib import Path
from .metriche import riepilogo_da_matrice
from .metriche import matrice_confusione


def training_step(modello, inputs, target, criterio, optimizer):
    """Esegue una singola iterazione di addestramento."""

    modello.train()
    optimizer.zero_grad(set_to_none=True)

    # Forward
    logits = modello(inputs)

    # Calcolo della loss
    loss = criterio(logits, target)

    if not torch.isfinite(loss).item():
        raise RuntimeError("Loss non finita.")

    # Backward
    loss.backward()

    # Aggiornamento dei parametri
    optimizer.step()

    return loss.detach().item()

def addestra_su_patch(
    modello,
    inputs,
    target,
    criterio,
    optimizer,
    num_iterazioni
):
    """Ripete l'addestramento sulla stessa patch."""

    storico_loss = []

    for _ in range(num_iterazioni):

        loss = training_step(
            modello,
            inputs,
            target,
            criterio,
            optimizer
        )

        storico_loss.append(loss)

    return storico_loss

def salva_checkpoint(
    percorso,
    modello,
    optimizer,
    configurazione,
    storico_loss
):
    """Salva lo stato del modello e dell'addestramento."""

    from pathlib import Path

    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": modello.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "configurazione": configurazione,
        "storico_loss": storico_loss
    }, percorso)

    return percorso


def addestra_epoca(
    modello,
    loader,
    criterio,
    optimizer,
    device
):
    """Esegue un'epoca di addestramento."""

    modello.train()

    loss_totale = 0.0
    pixel_totali = 0
    batch_eseguiti = 0

    for batch in loader:

        inputs = batch["input"].to(device)
        target = batch["target"].to(device)

        pixel_validi = (
            target != criterio.ignore_index
        ).sum().item()

        if pixel_validi == 0:
            continue

        loss = training_step(
            modello,
            inputs,
            target,
            criterio,
            optimizer
        )

        loss_totale += loss * pixel_validi
        pixel_totali += pixel_validi
        batch_eseguiti += 1

    if pixel_totali == 0:
        raise RuntimeError("Nessun pixel valido nell'epoca.")

    return {
        "loss": loss_totale / pixel_totali,
        "pixel_validi": pixel_totali,
        "batch_eseguiti": batch_eseguiti
    }


def valida_epoca(modello, loader, criterio, device):
    """Valuta il modello senza aggiornare i parametri."""

    modello.eval()

    loss_totale = 0.0
    pixel_totali = 0
    batch_eseguiti = 0
    batch_ignorati = 0

    matrice_totale = np.zeros((4, 4), dtype=np.int64)

    with torch.no_grad():

        for batch in loader:

            inputs = batch["input"].to(device)
            target = batch["target"].to(device)

            validita = target != criterio.ignore_index

            pixel_validi = validita.sum().item()

            if pixel_validi == 0:
                batch_ignorati += 1
                continue

            logits = modello(inputs)

            loss = criterio(logits, target).item()

            if not np.isfinite(loss):
                raise RuntimeError(
                    "Validation loss non finita."
                )

            previsione = logits.argmax(dim=1)

            riferimento = target[validita].cpu().numpy()
            predizione = previsione[validita].cpu().numpy()

            matrice_totale += matrice_confusione(
                riferimento,
                predizione,
                n_classi=4
            )

            loss_totale += loss * pixel_validi
            pixel_totali += pixel_validi
            batch_eseguiti += 1

    if pixel_totali == 0:
        raise RuntimeError(
            "Nessun pixel valido nella validation."
        )

    return {
        "loss": loss_totale / pixel_totali,
        "pixel_validi": pixel_totali,
        "batch_eseguiti": batch_eseguiti,
        "batch_ignorati": batch_ignorati,
        "matrice_confusione": matrice_totale
    }


def addestra_modello(
    modello,
    loader_train,
    loader_val,
    criterio,
    optimizer,
    device,
    epoche_totali,
    storico,
    configurazione,
    cartella_checkpoint,
    cartella_risultati
):
    """Addestra e valuta il modello fino al numero di epoche richiesto."""

    cartella_checkpoint = Path(cartella_checkpoint)
    cartella_risultati = Path(cartella_risultati)

    cartella_checkpoint.mkdir(parents=True, exist_ok=True)
    cartella_risultati.mkdir(parents=True, exist_ok=True)

    percorso_ultimo = (
        cartella_checkpoint / "unet_baseline_ultimo.pt"
    )

    percorso_migliore = (
        cartella_checkpoint / "unet_baseline_migliore.pt"
    )

    migliore_miou = max(
        (
            riga["mIoU diagnostica"]
            for riga in storico
        ),
        default=float("-inf")
    )

    for epoca in range(len(storico) + 1, epoche_totali + 1):

        print(f"\nEPOCA {epoca}/{epoche_totali}", flush=True)

        # Training
        inizio = time.perf_counter()

        risultato_train = addestra_epoca(
            modello,
            loader_train,
            criterio,
            optimizer,
            device
        )

        tempo_train = time.perf_counter() - inizio

        # Validation
        inizio = time.perf_counter()

        risultato_val = valida_epoca(
            modello,
            loader_val,
            criterio,
            device
        )

        tempo_val = time.perf_counter() - inizio

        matrice = risultato_val["matrice_confusione"]

        if matrice.sum() != risultato_val["pixel_validi"]:
            raise RuntimeError("Conteggio validation incoerente.")

        tabella, riepilogo = riepilogo_da_matrice(matrice)

        miou = float(riepilogo["mIoU diagnostica"])

        if not np.isfinite(miou):
            raise RuntimeError("mIoU non finita.")

        # Aggiornamento dello storico
        storico.append({
            "epoca": epoca,
            "train_loss": risultato_train["loss"],
            "val_loss": risultato_val["loss"],
            "tempo_train": tempo_train,
            "tempo_val": tempo_val,
            **riepilogo
        })

        nuovo_migliore = miou > migliore_miou

        if nuovo_migliore:
            migliore_miou = miou

        config_checkpoint = {
            **configurazione,
            "epoche_completate": epoca,
            "migliore_miou": migliore_miou,
            "stato_generatore_train": (
                loader_train.generator.get_state()
            ),
            "stato_rng_torch": torch.get_rng_state(),
            "stato_rng_mps": (
                torch.mps.get_rng_state()
                if device.type == "mps"
                else None
            )
        }

        # Salvataggio dell'ultimo modello
        temporaneo = percorso_ultimo.with_suffix(".tmp")

        salva_checkpoint(
            temporaneo,
            modello,
            optimizer,
            config_checkpoint,
            storico
        )

        temporaneo.replace(percorso_ultimo)

        # Salvataggio del modello migliore
        if nuovo_migliore:
            temporaneo_migliore = (
                percorso_migliore.with_suffix(".tmp")
            )

            shutil.copyfile(
                percorso_ultimo,
                temporaneo_migliore
            )

            temporaneo_migliore.replace(percorso_migliore)

        # Salvataggio dei risultati numerici
        pd.DataFrame(storico).to_csv(
            cartella_risultati / "storico_baseline.csv",
            index=False
        )

        tabella.to_csv(
            cartella_risultati
            / f"metriche_epoca_{epoca:02d}.csv"
        )

        pd.DataFrame(matrice).to_csv(
            cartella_risultati
            / f"matrice_epoca_{epoca:02d}.csv",
            index=False
        )

        # Riepilogo dell'epoca
        print(f"Training loss: {risultato_train['loss']:.4f}")
        print(f"Validation loss: {risultato_val['loss']:.4f}")
        print(f"Accuracy: {riepilogo['Accuracy'] * 100:.2f}%")
        print(f"mIoU: {miou * 100:.2f}%")
        print(f"F1 edificio: {riepilogo['F1 edificio'] * 100:.2f}%")

        print(
            f"Tempo totale: "
            f"{(tempo_train + tempo_val) / 60:.2f} minuti"
        )

        if nuovo_migliore:
            print("Nuovo checkpoint migliore.")

        print("Checkpoint e risultati salvati.", flush=True)

    return storico
