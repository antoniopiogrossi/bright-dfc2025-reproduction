# BRIGHT · Riproduzione del DFC 2025, Track 2

È possibile riconoscere il danno a un edificio confrontando un’immagine ottica acquisita **prima** del disastro con un’immagine radar (SAR) acquisita **dopo**? Questo progetto di tesi studia il problema sul dataset BRIGHT. L’uscita attesa è una mappa in cui ogni pixel appartiene a una delle quattro classi: sfondo, edificio integro, danneggiato o distrutto.

L’obiettivo sperimentale è riprodurre la soluzione vincitrice del **2025 IEEE GRSS Data Fusion Contest, Track 2**, misurare quanto i risultati si avvicinano a quelli pubblicati e usare la riproduzione come riferimento per valutare eventuali miglioramenti.

**Stato attuale:** sono disponibili l’esplorazione di tre esempi, il controllo delle liste ufficiali e la verifica delle metriche. Non sono ancora state prodotte previsioni da una rete. Le tabelle presenti nel repository descrivono i dati e il comportamento della valutazione.

## Percorso di lettura

| Notebook | Domanda a cui risponde | Risultato |
|---|---|---|
| [01 · Esplorazione BRIGHT](01_esplorazione_bright.ipynb) | Che cosa entra nella rete e qual è la risposta di riferimento? | Recupera tre esempi di Bata, confronta ottico, SAR e maschera del primo esempio, ne verifica la griglia geografica e prepara i tre esempi in batch PyTorch. |
| [02 · Dati e split](02_dati_e_split.ipynb) | Quali esempi compaiono nelle quattro liste e quali abbiamo davvero sul disco? | Confronta gli identificativi, verifica la disponibilità locale e conta i pixel per classe nelle maschere degli esempi completi distinti. [Tabelle CSV](risultati/02_dati_e_split/) |
| [03 · Metriche](03_metriche.ipynb) | Come viene valutata una mappa del danno? | Confronta i calcoli diretti con la classe originale `Evaluator` di BRIGHT e verifica i casi limite con previsioni artificiali. [Tabelle CSV](risultati/03_metriche/) |

## Organizzazione del codice

I notebook restano nella radice del progetto: scelgono gli esempi, espongono il ragionamento e mostrano figure e tabelle. La logica riutilizzabile è nel package [`src/bright_reproduction/`](src/bright_reproduction/).

| Moduli | Responsabilità |
|---|---|
| `download.py`, `raster.py`, `geospatial.py` | Cercare i file nell’archivio remoto, controllarne l’integrità al recupero, leggere GeoTIFF e maschere, confrontare le griglie dichiarate. |
| `data.py`, `visualization.py` | Preparare gli ingressi, caricare esempi con `BrightEsempiDataset` e visualizzare ottico, SAR e annotazione. |
| `splits.py`, `campione.py` | Leggere e confrontare le liste con `SuddivisioniBright`, controllare i file locali e calcolare le statistiche del campione. |
| `metriche.py` | Calcolare metriche indipendenti e caricare il valutatore dal codice originale BRIGHT per il confronto. |

## Riprodurre le analisi

L’ambiente di riferimento è Python 3.11.9 su macOS arm64, con VS Code e Jupyter. [`requirements.txt`](requirements.txt) riporta le versioni dell’ambiente di analisi; l’installazione da zero su altre configurazioni resta da verificare.

**1. Preparare l’ambiente.** Dalla radice di questo repository, per un nuovo ambiente:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m ipykernel install --user --name bright-dfc2025 --display-name "Python (BRIGHT DFC2025)"
```

Se si usa già `venv-tesi` con le dipendenze e il package installati, basta selezionare il kernel **Python (venv-tesi)**. L’installazione con `-e` permette ai notebook di importare il codice direttamente da `src/`.

**2. Recuperare il riferimento BRIGHT.** I notebook leggono le liste e il codice delle metriche dalla copia locale. Per crearla:

```bash
git clone https://github.com/ChenHongruixuan/BRIGHT.git BRIGHT
git -C BRIGHT checkout 59269142f3a3550320513e362692732f46486985
```

**3. Eseguire 01 → 02 → 03**, mantenendo la radice del progetto come directory di lavoro. Il notebook 01 recupera nove GeoTIFF dall’[archivio dati BRIGHT](https://huggingface.co/datasets/Kullervo/BRIGHT): tre ottici, tre SAR e tre maschere. Li salva in `dati/bright_esempi/{pre-event,post-event,target}/`. La ricerca e la verifica dell’integrità richiedono Internet anche se i file sono già presenti. I notebook 02 e 03 usano i dati locali e rigenerano i CSV in `risultati/`.

Per studiare anche il codice dei vincitori, la versione presa come riferimento si recupera separatamente. Non serve per eseguire i tre notebook:

```bash
git clone https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping.git soluzione_vincitori
git -C soluzione_vincitori checkout fe8a8580949dc4e1e38a5a1e3901cee7e20f806b
```

I dati e le due copie dei repository esterni non sono inclusi qui; per il loro utilizzo valgono le condizioni delle rispettive fonti.

## Cosa hanno mostrato i primi controlli

- **Dati.** Nel primo esempio la griglia dichiarata coincide per ottico, SAR e maschera. Il SAR ha una sola banda: replicarla su tre canali serve a preparare l’ingresso nel formato usato dal caricatore dei vincitori. La maschera conserva gli indici interi delle classi.
- **Split.** Train, holdout, validation e test contengono rispettivamente 2.229, 661, 349 e 156 voci: **3.395 voci, 3.393 identificativi distinti**. `bata-explosion_00000049` compare in train e holdout, entrambi destinati al training nel protocollo originario; `libya-flood_00000024` compare in train e validation. Le liste originali sono conservate. Prima di addestrare occorre documentare come si evita di valutare un esempio già visto.
- **Metriche.** Sulle tre maschere locali, prevedere sempre «sfondo» dà **88,91% di accuratezza** ma **0% di IoU per ciascuna classe edificio**. La mIoU sulle quattro classi, calcolata dopo aver accumulato i pixel, è **22,23%**. Il caso mostra perché l’accuratezza da sola descrive male la capacità di riconoscere il danno. Questi numeri riguardano tre esempi dello stesso evento.
- **Protocollo.** Il notebook 03 distingue la localizzazione binaria, l’F1 del danno sui veri pixel edificio nello script BRIGHT esaminato e la mIoU sull’intera mappa. Prima del confronto con la classifica va verificato quale calcolo riproduca il punteggio ufficiale.

## Esperimenti previsti

| Fase | Lavoro previsto | Risultato da documentare |
|---|---|---|
| 1 · Primo modello di riferimento | Definire split operativi e protocollo di valutazione. Provare una U-Net su pochi esempi, poi estendere l’addestramento su GPU. | Una prima mappa predetta confrontata con la maschera corretta; successivamente, metriche su dati separati dal training. |
| 2 · Localizzazione degli edifici | Riprodurre la fase `BuildingExtraction` dei vincitori, che produce la mappa degli edifici usata dalla fase successiva. | Qualità dei footprint predetti, misurata rispetto al riferimento binario edificio/sfondo. |
| 3 · Classificazione del danno | Collegare `ChangeDetection` alla mappa degli edifici predetta; ricostruire l’uso delle pseudo-etichette e il post-processing. | Mappe del danno e risultati per classe, indicando quali immagini e annotazioni sono state usate in ogni fase. |
| 4 · Confronto e miglioramenti | Confrontare la riproduzione con i risultati pubblicati a parità di protocollo; analizzare gli errori e provare una modifica alla volta. | Una tabella di confronto che riporti split, configurazione e metriche; ripetizioni con semi diversi per misurare la variabilità. |

Nei materiali dei vincitori esaminati non sono stati individuati i pesi finali e gli script contengono chiamate esplicite a CUDA. Prima di avviare una riproduzione completa occorre quindi verificare l’ambiente GPU e quali componenti sia necessario riaddestrare.
