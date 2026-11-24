# BRIGHT · Riproduzione del DFC 2025, Track 2

È possibile riconoscere il danno a un edificio confrontando un’immagine ottica acquisita **prima** del disastro con un’immagine radar (SAR) acquisita **dopo**? Questo progetto di tesi studia il problema sul dataset BRIGHT. L’uscita attesa è una mappa in cui ogni pixel appartiene a una delle quattro classi: sfondo, edificio integro, danneggiato o distrutto.

L’obiettivo sperimentale è riprodurre la soluzione vincitrice del **2025 IEEE GRSS Data Fusion Contest, Track 2**, misurare quanto i risultati si avvicinano a quelli pubblicati e usare la riproduzione come riferimento per valutare eventuali miglioramenti.

**Stato attuale:** sono disponibili l’esplorazione di tre esempi, il controllo delle liste ufficiali e la verifica delle metriche. È stato inoltre completato il primo esperimento tecnico con la U-Net originale di BRIGHT: forward, backward, addestramento su una singola patch, previsione, metriche diagnostiche e verifica del checkpoint. La baseline su dati indipendenti è ancora in corso; la soluzione vincitrice non è ancora stata riprodotta.

## Percorso di lettura

| Notebook | Domanda a cui risponde | Risultato |
|---|---|---|
| [01 · Esplorazione BRIGHT](01_esplorazione_bright.ipynb) | Che cosa entra nella rete e qual è la risposta di riferimento? | Recupera tre esempi di Bata, confronta ottico, SAR e maschera del primo esempio, ne verifica la griglia geografica e prepara i tre esempi in batch PyTorch. |
| [02 · Dati e split](02_dati_e_split.ipynb) | Quali esempi compaiono nelle quattro liste e quali abbiamo davvero sul disco? | Confronta gli identificativi, verifica la disponibilità locale e conta i pixel per classe nelle maschere degli esempi completi distinti. [Tabelle CSV](risultati/02_dati_e_split/) |
| [03 · Metriche](03_metriche.ipynb) | Come viene valutata una mappa del danno? | Confronta i calcoli diretti con la classe originale `Evaluator` di BRIGHT e verifica i casi limite con previsioni artificiali. [Tabelle CSV](risultati/03_metriche/) |
| [04 · Baseline U-Net](04_baseline_unet.ipynb) | La rete riesce ad apprendere dai dati reali? | Verifica il ciclo di addestramento su una patch, mostra loss e previsione, confronta le metriche con il controllo «sempre sfondo» e verifica il checkpoint. In corso: valutazione indipendente. |

## Organizzazione del codice

I notebook restano nella radice del progetto: scelgono gli esempi, espongono il ragionamento e mostrano figure e tabelle. La logica riutilizzabile è nel package [`src/bright_reproduction/`](src/bright_reproduction/).

| Moduli | Responsabilità |
|---|---|
| `download.py`, `raster.py`, `geospatial.py` | Cercare i file nell’archivio remoto, controllarne l’integrità al recupero, leggere GeoTIFF e maschere, confrontare le griglie dichiarate. |
| `data.py`, `visualization.py` | Preparare gli ingressi, caricare esempi con `BrightEsempiDataset`, estrarre patch, gestire le etichette non valide e visualizzare ottico, SAR, annotazione e previsione. |
| `splits.py`, `campione.py` | Leggere e confrontare le liste con `SuddivisioniBright`, controllare i file locali e calcolare le statistiche del campione. |
| `metriche.py` | Calcolare metriche indipendenti, valutare le previsioni sui pixel validi e caricare il valutatore originale BRIGHT per il confronto. |
| `models.py` | Inizializzare la U-Net originale dalla copia locale di BRIGHT, senza duplicarne il codice. |
| `training.py` | Eseguire i passi di addestramento, ripeterli su una patch e salvare i checkpoint. |

## Riprodurre le analisi e il primo esperimento

L’ambiente di riferimento è Python 3.11.9 su macOS arm64, con VS Code e Jupyter. Il primo esperimento è stato eseguito con PyTorch 2.14.1, Rasterio 1.4.4 e accelerazione MPS sulla GPU Apple. CUDA non è disponibile sul Mac; un ambiente NVIDIA separato resta da verificare. [`requirements.txt`](requirements.txt) riporta le versioni dell’ambiente di analisi; l’installazione da zero su altre configurazioni resta da verificare.

**1. Preparare l’ambiente.** Dalla radice di questo repository, per un nuovo ambiente:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m ipykernel install --user --name bright-dfc2025 --display-name "Python (BRIGHT DFC2025)"
```

Se si usa già `venv-tesi` con le dipendenze e il package installati, basta selezionare il kernel **Python (venv-tesi)**. L’installazione con `-e` permette ai notebook di importare il codice direttamente da `src/`.

**2. Recuperare il riferimento BRIGHT.** I notebook leggono le liste, il codice delle metriche e la U-Net dalla copia locale. Per crearla:

```bash
git clone https://github.com/ChenHongruixuan/BRIGHT.git BRIGHT
git -C BRIGHT checkout 59269142f3a3550320513e362692732f46486985
```

**3. Eseguire 01 → 02 → 03 → 04**, mantenendo la radice del progetto come directory di lavoro. Il notebook 01 recupera nove GeoTIFF dall’[archivio dati BRIGHT](https://huggingface.co/datasets/Kullervo/BRIGHT): tre ottici, tre SAR e tre maschere. Li salva in `dati/bright_esempi/{pre-event,post-event,target}/`. La ricerca e la verifica dell’integrità richiedono Internet anche se i file sono già presenti. I notebook 02 e 03 usano i dati locali e rigenerano i CSV in `risultati/`.

Il notebook 04 usa gli stessi dati locali, seleziona CUDA, MPS o CPU in base alla disponibilità e salva il checkpoint in `checkpoints/unet_overfit_patch.pt`. La riesecuzione dell’esperimento sovrascrive tale file. Il checkpoint (355,44 MiB nell’esecuzione verificata) è escluso da Git; contiene stato del modello, ottimizzatore, configurazione e storico delle 30 iterazioni aggiuntive. Il caricamento ha restituito previsioni identiche e logits equivalenti. La ripresa dell’addestramento dall’ottimizzatore non è ancora stata verificata.

Per studiare anche il codice dei vincitori, la versione presa come riferimento si recupera separatamente. Non serve per eseguire i quattro notebook:

```bash
git clone https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping.git soluzione_vincitori
git -C soluzione_vincitori checkout fe8a8580949dc4e1e38a5a1e3901cee7e20f806b
```

I dati e le due copie dei repository esterni non sono inclusi qui; per il loro utilizzo valgono le condizioni delle rispettive fonti.

## Cosa hanno mostrato i primi controlli

- **Dati.** Nel primo esempio la griglia dichiarata coincide per ottico, SAR e maschera. Il SAR ha una sola banda: replicarla su tre canali serve a preparare l’ingresso nel formato usato dal caricatore dei vincitori. La maschera conserva gli indici interi delle classi.
- **Split.** Train, holdout, validation e test contengono rispettivamente 2.229, 661, 349 e 156 voci: **3.395 voci, 3.393 identificativi distinti**. `bata-explosion_00000049` compare in train e holdout, entrambi destinati al training nel protocollo originario; `libya-flood_00000024` compare in train e validation. Le liste originali sono conservate. Prima di estendere l’addestramento e introdurre la validation indipendente occorre documentare come si evita di valutare un esempio già visto.
- **Metriche.** Sulle tre maschere locali, prevedere sempre «sfondo» dà **88,91% di accuratezza** ma **0% di IoU per ciascuna classe edificio**. La mIoU sulle quattro classi, calcolata dopo aver accumulato i pixel, è **22,23%**. Il caso mostra perché l’accuratezza da sola descrive male la capacità di riconoscere il danno. Questi numeri riguardano tre esempi dello stesso evento.
- **Protocollo.** Il notebook 03 distingue la localizzazione binaria, l’F1 del danno sui veri pixel edificio nello script BRIGHT esaminato e la mIoU sull’intera mappa. Prima del confronto con la classifica va verificato quale calcolo riproduca il punteggio ufficiale.

## Primo esperimento U-Net

Il notebook 04 utilizza la U-Net originale del benchmark BRIGHT (31.045.444 parametri), inizializzata senza pesi preaddestrati. L’ottico RGB e il SAR replicato sono concatenati in un tensore a sei canali; la rete restituisce quattro logits per pixel, da cui si ricava la classe predetta con `argmax`. La maschera reale è utilizzata come target, non come ingresso.

Il test usa la patch centrale da 128 × 128 pixel di `bata-explosion_00000049`, contenente soltanto sfondo e classe distrutto. Sono stati effettuati un primo aggiornamento dei pesi e 30 iterazioni aggiuntive, con batch size 1, seed 42, AdamW (learning rate 0,0001) e Cross Entropy (`ignore_index=255`). Tutti i 16.384 pixel della maschera sono risultati validi. La validità degli ingressi ottico e SAR non è ancora gestita completamente.

La loss nelle 30 iterazioni aggiuntive è scesa da 1,42167 a 0,59883; la loss in modalità `eval`, dopo l’ultimo aggiornamento, è 1,02844. Il notebook mostra il grafico della loss e il confronto fra ottico, SAR, annotazione e previsione.

| Metrica | U-Net | Sempre sfondo |
|---|---:|---:|
| Accuracy | 67,03% | 50,02% |
| mIoU diagnostica | 24,97% | 25,01% |
| mIoU 4 classi | 24,97% | 12,50% |
| mIoU classi presenti | 49,95% | 25,01% |
| F1 edificio | 64,29% | 0,00% |
| IoU distrutto | 34,65% | 0,00% |

La rete riconosce parzialmente gli edifici, ma tende a sottostimarne l’estensione. La mIoU diagnostica esclude le classi con IoU indefinita (assenti sia nel riferimento sia nella previsione); la mIoU a quattro classi assegna loro zero, mentre quella sulle classi presenti usa soltanto le classi del riferimento. Quest’ultima non penalizza direttamente nella media le classi assenti erroneamente predette. Le tre convenzioni sono riportate separatamente.

**Limite dell’esperimento:** queste metriche sono calcolate sulla stessa patch utilizzata per il training. Il test verifica il funzionamento della pipeline e la capacità di adattamento locale, non la generalizzazione. Non costituisce una riproduzione della soluzione vincitrice né un punteggio ufficiale della competizione.

## Stato delle fasi sperimentali

| Fase | Lavoro previsto | Risultato da documentare |
|---|---|---|
| 1 · Primo modello di riferimento (in corso) | Test su una patch completato. Definire split operativi, estendere Dataset e downloader, completare la gestione NoData e addestrare su più esempi. Verificare il protocollo e l’ambiente NVIDIA. | Prima mappa e metriche sulla patch già disponibili; restano da ottenere metriche su dati separati dal training. |
| 2 · Localizzazione degli edifici | Riprodurre la fase `BuildingExtraction` dei vincitori, che produce la mappa degli edifici usata dalla fase successiva. | Qualità dei footprint predetti, misurata rispetto al riferimento binario edificio/sfondo. |
| 3 · Classificazione del danno | Collegare `ChangeDetection` alla mappa degli edifici predetta; ricostruire l’uso delle pseudo-etichette e il post-processing. | Mappe del danno e risultati per classe, indicando quali immagini e annotazioni sono state usate in ogni fase. |
| 4 · Confronto e miglioramenti | Confrontare la riproduzione con i risultati pubblicati a parità di protocollo; analizzare gli errori e provare una modifica alla volta. | Una tabella di confronto che riporti split, configurazione e metriche; ripetizioni con semi diversi per misurare la variabilità. |

Per gli split operativi si propone di unire train e holdout, deduplicare gli identificativi ed escludere dal training quelli della validation: 2.888 ID di training, 349 di validation e 156 di test. Le liste ufficiali restano inalterate. La proposta non è ancora implementata; resta da verificare anche l’eventuale sovrapposizione geografica tra ID differenti. Il downloader attuale ricerca soltanto nella cartella train dell’archivio.

Nei materiali dei vincitori esaminati non sono stati individuati i pesi finali e gli script contengono chiamate esplicite a CUDA. Prima di avviare una riproduzione completa occorre quindi verificare l’ambiente GPU e quali componenti sia necessario riaddestrare.
