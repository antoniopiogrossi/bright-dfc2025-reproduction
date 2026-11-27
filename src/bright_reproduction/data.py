"""Preparazione delle immagini e Dataset per gli esempi BRIGHT."""

from pathlib import Path

import numpy as np
import rasterio
import torch
import pandas as pd
from PIL import Image
from .raster import maschera_validita_congiunta
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


def normalizza_imagenet(immagine):
    """Normalizzazione ImageNet in formato CHW, come in BRIGHT."""

    immagine = np.asarray(immagine, dtype=np.float32)

    if immagine.ndim != 3 or immagine.shape[0] != 3:
        raise ValueError("Attesa un'immagine con 3 canali CHW.")

    media = np.array(
        [123.675, 116.28, 103.53],
        dtype=np.float32
    )[:, None, None]

    std = np.array(
        [58.395, 57.12, 57.375],
        dtype=np.float32
    )[:, None, None]

    return (immagine - media) / std


def carica_patch_multimodale(
    cartella,
    identificativo,
    x,
    y,
    dimensione=128,
    ignore_index=255
):
    """Carica e prepara una patch ottico-SAR con target valido."""

    from rasterio.windows import Window

    cartella = Path(cartella)

    finestra = Window(x, y, dimensione, dimensione)

    percorso_ottico = (
        cartella / "pre-event"
        / f"{identificativo}_pre_disaster.tif"
    )

    percorso_sar = (
        cartella / "post-event"
        / f"{identificativo}_post_disaster.tif"
    )

    percorso_target = (
        cartella / "target"
        / f"{identificativo}_building_damage.tif"
    )

    with rasterio.open(percorso_ottico) as file:
        ottico = file.read(
            [1, 2, 3], window=finestra
        ).astype(np.float32)

        valido_ottico = (
            file.read_masks([1, 2, 3], window=finestra) > 0
        ).all(axis=0)

    with rasterio.open(percorso_sar) as file:
        sar = file.read(
            1, window=finestra
        ).astype(np.float32)

        valido_sar = file.read_masks(
            1, window=finestra
        ) > 0

    with rasterio.open(percorso_target) as file:
        target = file.read(
            1, window=finestra
        ).astype(np.int64)

        valido_target = (
            (file.read_masks(1, window=finestra) > 0)
            & np.isin(target, [0, 1, 2, 3])
        )

    if target.shape != (dimensione, dimensione):
        raise ValueError("Patch fuori dai limiti del raster.")

    validita = (
        valido_ottico
        & valido_sar
        & valido_target
    )

    sar = np.repeat(sar[None, :, :], 3, axis=0)

    ottico = normalizza_imagenet(ottico)
    sar = normalizza_imagenet(sar)

    # Riempimento neutro nello spazio normalizzato
    ottico[:, ~valido_ottico] = 0.0
    sar[:, ~valido_sar] = 0.0

    target[~validita] = ignore_index

    inputs = np.concatenate([ottico, sar], axis=0)

    return {
        "input": torch.from_numpy(
            np.ascontiguousarray(inputs)
        ),
        "target": torch.from_numpy(target),
        "validita": torch.from_numpy(validita),
        "id": identificativo,
        "x": x,
        "y": y
    }


class BrightPatchDataset(Dataset):
    """Dataset multimodale BRIGHT basato su patch non sovrapposte."""

    def __init__(
        self,
        cartella,
        identificativi,
        dimensione=128,
        training=True,
        ignore_index=255
    ):
        self.cartella = Path(cartella)
        self.identificativi = list(identificativi)
        self.dimensione = dimensione
        self.training = training
        self.ignore_index = ignore_index

        self.patch = []

        if dimensione <= 0:
            raise ValueError("Dimensione non valida.")

        for id_ in self.identificativi:

            ottico = (
                self.cartella / "pre-event"
                / f"{id_}_pre_disaster.tif"
            )

            sar = (
                self.cartella / "post-event"
                / f"{id_}_post_disaster.tif"
            )

            target = (
                self.cartella / "target"
                / f"{id_}_building_damage.tif"
            )

            validita = maschera_validita_congiunta(
                ottico,
                sar,
                target,
                ottico_rigoroso=True
            )

            h, w = validita.shape

            if h % dimensione or w % dimensione:
                raise ValueError(
                    f"Dimensioni non divisibili: {id_}"
                )

            for y in range(0, h, dimensione):
                for x in range(0, w, dimensione):

                    regione = validita[
                        y:y+dimensione,
                        x:x+dimensione
                    ]

                    if self.training and not regione.any():
                        continue

                    self.patch.append((id_, x, y))

        if not self.patch:
            raise ValueError("Nessuna patch disponibile.")

    def __len__(self):
        return len(self.patch)

    def __getitem__(self, indice):

        id_, x, y = self.patch[indice]

        return carica_patch_multimodale(
            self.cartella,
            id_,
            x,
            y,
            dimensione=self.dimensione,
            ignore_index=self.ignore_index
        )

def statistiche_patch(dataset):
    """Conta i pixel validi per classe in ogni patch."""

    risultati = []

    for indice in range(len(dataset)):

        esempio = dataset[indice]
        target = esempio["target"].numpy()

        validi = target != dataset.ignore_index

        conteggi = np.bincount(
            target[validi],
            minlength=4
        )

        risultati.append({
            "indice": indice,
            "id": esempio["id"],
            "x": esempio["x"],
            "y": esempio["y"],
            "pixel_validi": int(validi.sum()),
            "pixel_ignorati": int((~validi).sum()),
            "sfondo": int(conteggi[0]),
            "integro": int(conteggi[1]),
            "danneggiato": int(conteggi[2]),
            "distrutto": int(conteggi[3])
        })

    return pd.DataFrame(risultati)
