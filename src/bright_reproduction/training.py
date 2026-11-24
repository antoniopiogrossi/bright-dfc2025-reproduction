
"""Funzioni per l'addestramento dei modelli."""

import torch


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