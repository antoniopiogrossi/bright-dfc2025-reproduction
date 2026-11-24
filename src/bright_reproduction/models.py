
"""Modelli per la segmentazione dei danni agli edifici."""

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path


def crea_unet(cartella_bright):
    """Inizializza la U-Net originale del benchmark BRIGHT."""

    percorso = (
        Path(cartella_bright)
        / "bda_benchmark"
        / "model"
        / "UNet.py"
    )

    if not percorso.is_file():
        raise FileNotFoundError(percorso)

    spec = spec_from_file_location("bright_unet", percorso)
    modulo = module_from_spec(spec)
    spec.loader.exec_module(modulo)

    return modulo.UNet(in_channels=6, num_classes=4)
