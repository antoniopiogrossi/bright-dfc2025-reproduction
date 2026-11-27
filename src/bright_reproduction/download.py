"""Download selettivo degli esempi BRIGHT da un archivio ZIP remoto."""

from pathlib import Path, PurePosixPath
import zlib

from remotezip import RemoteZip


def cerca_file_esempi(
    url_archivio,
    identificativi,
    split="train",
    solo_target=False
):
    """Cerca i file degli esempi negli archivi DFC2025 Track 2."""

    identificativi = list(identificativi)

    if not identificativi:
        raise ValueError("Nessun identificativo specificato.")

    if len(set(identificativi)) != len(identificativi):
        raise ValueError("Gli identificativi contengono duplicati.")

    if split not in {"train", "val"}:
        raise ValueError(f"Split non supportato: {split}")

    suffissi = {
        "pre-event": "_pre_disaster.tif",
        "post-event": "_post_disaster.tif",
        "target": "_building_damage.tif",
    }

    if split == "train":
        if solo_target:
            raise ValueError(
                "solo_target è previsto per la validation."
            )

        prefisso = "dfc25_track2_trainval/train"
        categorie = ["pre-event", "post-event", "target"]

    elif solo_target:
        prefisso = "dfc25_track2_val_labels"
        categorie = ["target"]

    else:
        prefisso = "dfc25_track2_trainval/val"
        categorie = ["pre-event", "post-event"]

    percorsi_attesi = [
        f"{prefisso}/{categoria}/{nome}{suffissi[categoria]}"
        for nome in identificativi
        for categoria in categorie
    ]

    with RemoteZip(
        url_archivio,
        initial_buffer_size=2 * 1024 * 1024,
        timeout=60,
    ) as archivio:
        nomi_presenti = {
            elemento.filename
            for elemento in archivio.infolist()
            if not elemento.is_dir()
        }

    mancanti = sorted(set(percorsi_attesi) - nomi_presenti)

    if mancanti:
        raise FileNotFoundError(
            "File attesi non trovati nell'archivio:\n"
            + "\n".join(mancanti)
        )

    return sorted(percorsi_attesi)


def scarica_file_esempi(url_archivio, percorsi_remoti, cartella_dati):
    """Scarica i file selezionati, verificandone dimensione e CRC.

    I file locali già integri vengono riutilizzati.
    Restituisce una lista con percorso locale e stato di ogni file.
    """
    cartella_dati = Path(cartella_dati)
    risultati = []

    with RemoteZip(
        url_archivio,
        initial_buffer_size=2 * 1024 * 1024,
        timeout=60,
    ) as archivio:
        for percorso_remoto in sorted(percorsi_remoti):
            informazioni = archivio.getinfo(percorso_remoto)
            percorso = PurePosixPath(percorso_remoto)
            categoria = percorso.parent.name

            if categoria not in {"pre-event", "post-event", "target"}:
                raise ValueError(f"Cartella inattesa: {categoria}")

            destinazione = cartella_dati / categoria / percorso.name
            destinazione.parent.mkdir(parents=True, exist_ok=True)

            if destinazione.is_file():
                contenuto_locale = destinazione.read_bytes()

                integro = (
                    len(contenuto_locale) == informazioni.file_size
                    and zlib.crc32(contenuto_locale) == informazioni.CRC
                )

                if integro:
                    risultati.append({
                        "percorso": destinazione,
                        "stato": "già presente e verificato",
                    })
                    continue

            contenuto = archivio.read(percorso_remoto)

            if (
                len(contenuto) != informazioni.file_size
                or zlib.crc32(contenuto) != informazioni.CRC
            ):
                raise RuntimeError(
                    f"Controllo di integrità fallito: {percorso.name}"
                )

            temporaneo = destinazione.with_suffix(".tif.part")
            temporaneo.write_bytes(contenuto)
            temporaneo.replace(destinazione)

            risultati.append({
                "percorso": destinazione,
                "stato": "scaricato e verificato",
            })

    return risultati


def scarica_campione_operativo(
    url_dati,
    url_labels,
    campione,
    cartella_dati
):
    """Scarica gli esempi di training e validation separatamente."""

    cartella_dati = Path(cartella_dati)

    train_ids = campione["train"]
    val_ids = campione["val"]

    if set(train_ids) & set(val_ids):
        raise ValueError(
            "Sovrapposizione tra training e validation."
        )

    # Ricerca dei file negli archivi
    file_train = cerca_file_esempi(
        url_dati,
        train_ids,
        split="train"
    )

    file_val = cerca_file_esempi(
        url_dati,
        val_ids,
        split="val"
    )

    labels_val = cerca_file_esempi(
        url_labels,
        val_ids,
        split="val",
        solo_target=True
    )

    # Download del training
    risultati_train = scarica_file_esempi(
        url_dati,
        file_train,
        cartella_dati / "train"
    )

    # Download delle immagini di validation
    risultati_val = scarica_file_esempi(
        url_dati,
        file_val,
        cartella_dati / "val"
    )

    # Download delle maschere di validation
    risultati_val += scarica_file_esempi(
        url_labels,
        labels_val,
        cartella_dati / "val"
    )

    return {
        "train": risultati_train,
        "val": risultati_val
    }
