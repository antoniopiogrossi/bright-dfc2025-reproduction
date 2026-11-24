"""Visualizzazione degli esempi BRIGHT."""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from .raster import leggi_maschera


def visualizza_esempio(
    percorso_ottico,
    percorso_sar,
    percorso_maschera,
    titolo=None,
):
    """Mostra ottico RGB, SAR uint8 e maschera del danno.

    Restituisce figura e assi, senza chiamare plt.show().
    I pixel non validi vengono rappresentati in grigio.
    """
    with rasterio.open(percorso_ottico) as dataset:
        if dataset.count != 3:
            raise ValueError("La visualizzazione richiede un ottico RGB.")

        ottico = dataset.read()
        validi_ottico = np.all(dataset.read_masks() > 0, axis=0)

    rgb = np.moveaxis(ottico, 0, -1)

    with rasterio.open(percorso_sar) as dataset:
        if dataset.count != 1:
            raise ValueError("La visualizzazione richiede un SAR a una banda.")

        sar = dataset.read(1, masked=True)

    maschera = leggi_maschera(percorso_maschera)

    colori = ["#303030", "#4CAF50", "#FFA726", "#E53935"]
    nomi_classi = [
        "Sfondo",
        "Edificio integro",
        "Edificio danneggiato",
        "Edificio distrutto",
    ]

    mappa_classi = ListedColormap(colori)
    mappa_classi.set_bad("#808080")
    norma_classi = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5, 3.5],
        ncolors=mappa_classi.N,
    )

    mappa_sar = plt.get_cmap("gray").copy()
    mappa_sar.set_bad("#808080")

    fig, assi = plt.subplots(1, 3, figsize=(18, 6))

    assi[0].set_facecolor("#808080")
    assi[0].imshow(rgb, alpha=validi_ottico.astype(float))
    assi[0].set_title("Ottico — prima del disastro")

    assi[1].imshow(sar, cmap=mappa_sar, vmin=0, vmax=255)
    assi[1].set_title("SAR — dopo il disastro")

    assi[2].imshow(
        maschera,
        cmap=mappa_classi,
        norm=norma_classi,
        interpolation="nearest",
    )
    assi[2].set_title("Maschera del danno — riferimento")

    for asse in assi:
        asse.axis("off")

    legenda = [
        Patch(facecolor=colore, label=nome)
        for colore, nome in zip(colori, nomi_classi)
    ]

    fig.legend(
        handles=legenda,
        loc="lower center",
        ncol=4,
        frameon=False,
    )

    if titolo is not None:
        fig.suptitle(titolo)

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    return fig, assi


def visualizza_previsione_patch(inputs, target, previsione):
    """Confronta immagini, maschera reale e previsione."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    inputs = inputs.detach().cpu()
    target = target.detach().cpu()
    previsione = previsione.detach().cpu()

    # Inversione della normalizzazione [-1, 1]
    ottico = (
        inputs[0, :3].permute(1, 2, 0) + 1
    ) / 2

    sar = (inputs[0, 3] + 1) / 2

    colori = [
        "#222222",
        "#329b72",
        "#e8b848",
        "#d94d4d"
    ]

    classi = [
        "Sfondo",
        "Integro",
        "Danneggiato",
        "Distrutto"
    ]

    cmap = ListedColormap(colori)
    cmap.set_bad("#bbbbbb")

    norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5, 3.5],
        cmap.N
    )

    maschera = target[0].float()
    predetta = previsione[0].float()

    non_validi = (maschera < 0) | (maschera > 3)

    maschera[non_validi] = float("nan")
    predetta[non_validi] = float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(ottico.clamp(0, 1))
    axes[0, 0].set_title("Ottico pre-evento")

    axes[0, 1].imshow(sar, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("SAR post-evento")

    axes[1, 0].imshow(
        maschera, cmap=cmap, norm=norm,
        interpolation="nearest"
    )
    axes[1, 0].set_title("Ground truth")

    axes[1, 1].imshow(
        predetta, cmap=cmap, norm=norm,
        interpolation="nearest"
    )
    axes[1, 1].set_title("Previsione U-Net")

    for ax in axes.flat:
        ax.axis("off")

    legenda = [
        Patch(facecolor=c, label=f"{i} - {nome}")
        for i, (c, nome) in enumerate(zip(colori, classi))
    ]

    legenda.append(
        Patch(facecolor="#bbbbbb", label="Pixel ignorati")
    )

    fig.legend(
        handles=legenda,
        loc="lower center",
        ncol=3
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    return fig
