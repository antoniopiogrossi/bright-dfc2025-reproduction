"""Lettura e confronto delle quattro liste ufficiali BRIGHT."""
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from itertools import combinations

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
