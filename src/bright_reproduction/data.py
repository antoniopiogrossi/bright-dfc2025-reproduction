"""Preparazione delle immagini e Dataset per gli esempi BRIGHT."""

from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import Dataset


def prepara_immagine(percorso):
    """Legge un'immagine e restituisce un array float32 in formato CHW.

    Converte l'immagine in RGB e normalizza i valori da 0–255 a -1–1.
    Questa procedura riproduce quella verificata nel notebook 01
    per le immagini del campione BRIGHT.
    """
    with Image.open(percorso) as immagine:
        array = np.array(
            immagine.convert("RGB"),
            dtype=np.float32,
        )

    array = (array / 255.0 - 0.5) / 0.5

    return np.moveaxis(array, -1, 0)


class BrightEsempiDataset(Dataset):
    """Carica ottico, SAR, maschera del danno e identificativo.

    Dataset esplorativo per il campione BRIGHT verificato:
    immagini uint8 e maschere a una banda con classi 0–3.

    Non applica augmentation e non gestisce pixel da ignorare
    nella loss. Non fornisce la mappa degli edifici aggiuntiva
    richiesta dalla procedura completa dei vincitori.
    """

    def __init__(self, cartella, identificativi):
        self.cartella = Path(cartella)
        self.identificativi = list(identificativi)

    def __len__(self):
        return len(self.identificativi)

    def __getitem__(self, indice):
        nome = self.identificativi[indice]

        percorso_ottico = (
            self.cartella
            / "pre-event"
            / f"{nome}_pre_disaster.tif"
        )
        percorso_sar = (
            self.cartella
            / "post-event"
            / f"{nome}_post_disaster.tif"
        )
        percorso_maschera = (
            self.cartella
            / "target"
            / f"{nome}_building_damage.tif"
        )

        ottico = prepara_immagine(percorso_ottico)
        sar = prepara_immagine(percorso_sar)

        with rasterio.open(percorso_maschera) as file:
            etichette = file.read(1).astype(np.int64)

        return {
            "ottico": torch.from_numpy(np.ascontiguousarray(ottico)),
            "sar": torch.from_numpy(np.ascontiguousarray(sar)),
            "maschera": torch.from_numpy(etichette),
            "id": nome,
        }

def prepara_patch_centrale(batch, dimensione, device):
    """Estrae una patch centrale e concatena ottico e SAR."""

    import torch

    h, w = batch["maschera"].shape[-2:]

    if not 0 < dimensione <= min(h, w):
        raise ValueError("Dimensione della patch non valida.")

    y = (h - dimensione) // 2
    x = (w - dimensione) // 2

    ottico = batch["ottico"][
        :, :, y:y+dimensione, x:x+dimensione
    ]

    sar = batch["sar"][
        :, :, y:y+dimensione, x:x+dimensione
    ]

    target = batch["maschera"][
        :, y:y+dimensione, x:x+dimensione
    ].to(device)

    inputs = torch.cat(
        [ottico, sar], dim=1
    ).to(device)

    return inputs, target, x, y

def prepara_target_loss(
    target,
    percorso_maschera,
    x,
    y,
    ignore_index=255
):
    """Esclude dalla loss le etichette non valide."""

    import torch
    import rasterio
    from rasterio.windows import Window

    h, w = target.shape[-2:]

    with rasterio.open(percorso_maschera) as file:
        validita = file.read_masks(
            1,
            window=Window(x, y, w, h)
        ) > 0

        nodata = file.nodata

    validita = torch.from_numpy(validita).unsqueeze(0)
    validita = validita.to(target.device)

    target_loss = target.clone().long()

    target_loss[
        ~validita | (target_loss < 0) | (target_loss > 3)
    ] = ignore_index

    if not (target_loss != ignore_index).any().item():
        raise ValueError("La patch non contiene pixel validi.")

    return target_loss, nodata