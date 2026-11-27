"""Lettura e confronto delle quattro liste ufficiali BRIGHT."""
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from itertools import combinations

import random
import pandas as pd


@dataclass
class SuddivisioniBright:
    """Conserva le liste in ordine, incluse eventuali ripetizioni."""
    cartella: Path
    liste: dict[str, list[str]]

    @classmethod
    def da_cartella(cls, cartella):
        cartella = Path(cartella)
        liste = {}
        for nome in ("train", "holdout", "val", "test"):
            righe = (cartella / f"{nome}_set.txt").read_text(encoding="utf-8").splitlines()
            liste[nome] = [r.strip() for r in righe if r.strip()]
        return cls(cartella, liste)

    def sovrapposizioni(self):
        righe = [(a, b, id_) for a, b in combinations(self.liste, 2)
                 for id_ in sorted(set(self.liste[a]).intersection(self.liste[b]))]
        return pd.DataFrame(righe, columns=["split_a", "split_b", "id"])

    def posizioni(self, identificativi):
        """Numeri di riga nei file originali, comprese eventuali righe vuote."""
        cercati = set(identificativi)
        for nome in self.liste:
            file = self.cartella / f"{nome}_set.txt"
            for n, riga in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                if riga.strip() in cercati:
                    yield file.name, n, riga.strip()

    def eventi(self):
        conteggi = {nome: Counter(id_.rsplit("_", 1)[0] for id_ in lista)
                    for nome, lista in self.liste.items()}
        return pd.DataFrame(conteggi).fillna(0).astype(int).sort_index()

    def liste_operative(self):
        """Costruisce split operativi senza sovrapposizioni di ID."""

        train = list(dict.fromkeys(
            self.liste["train"] + self.liste["holdout"]
        ))

        val = list(self.liste["val"])
        test = list(self.liste["test"])

        esclusi = set(val) | set(test)

        train = [
            id_ for id_ in train
            if id_ not in esclusi
        ]

        operative = {
            "train": train,
            "val": val,
            "test": test
        }

        for nome, lista in operative.items():
            if len(lista) != len(set(lista)):
                raise ValueError(
                    f"Duplicati nello split {nome}"
                )

        for a, b in combinations(operative, 2):
            comuni = set(operative[a]) & set(operative[b])

            if comuni:
                raise ValueError(
                    f"Sovrapposizione tra {a} e {b}: {comuni}"
                )

        return operative

    def salva_liste_operative(self, destinazione):
        """Salva le liste operative senza modificare quelle ufficiali."""

        destinazione = Path(destinazione).resolve()

        if destinazione == self.cartella.resolve():
            raise ValueError(
                "Non sovrascrivere le liste ufficiali."
            )

        operative = self.liste_operative()

        destinazione.mkdir(parents=True, exist_ok=True)

        for nome, lista in operative.items():
            percorso = destinazione / f"{nome}_operativo.txt"

            percorso.write_text(
                "\n".join(lista) + "\n",
                encoding="utf-8"
            )

        return destinazione

    def campione_per_evento(
        self,
        n_train=2,
        n_val=1,
        seed=42
    ):
        """Seleziona un campione riproducibile per evento."""

        operative = self.liste_operative()
        rng = random.Random(seed)

        campione = {}

        for split, n in [
            ("train", n_train),
            ("val", n_val)
        ]:
            gruppi = {}

            for id_ in operative[split]:
                evento = id_.rsplit("_", 1)[0]

                gruppi.setdefault(evento, []).append(id_)

            selezionati = []

            for evento, lista in sorted(gruppi.items()):

                if len(lista) < n:
                    raise ValueError(
                        f"Esempi insufficienti: {split}, {evento}"
                    )

                selezionati.extend(
                    rng.sample(sorted(lista), n)
                )

            campione[split] = selezionati

        return campione