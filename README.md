# BRIGHT · Preparazione alla riproduzione del DFC2025 Track 2

Progetto di tesi dedicato allo studio della mappatura del danno agli edifici
mediante immagini ottiche pre-evento e SAR post-evento.

L'obiettivo è riprodurre una soluzione di riferimento del
2025 IEEE GRSS Data Fusion Contest, Track 2, e successivamente
studiare possibili miglioramenti attraverso esperimenti controllati.

## Stato del progetto

È stata completata una prima fase di esplorazione dei dati,
controllo delle suddivisioni e verifica delle metriche.

Non è ancora stato addestrato o eseguito un modello per produrre
previsioni. I risultati presenti documentano controlli sui dati
e prove con previsioni artificiali.

## Notebook

| Notebook | Contenuto |
|---|---|
| `01_esplorazione_bright.ipynb` | Download di tre esempi, lettura di ottico/SAR/maschere, controllo della griglia geografica, normalizzazione e caricamento in batch PyTorch |
| `02_dati_e_split.ipynb` | Controllo delle liste ufficiali, sovrapposizioni, distribuzione per evento, disponibilità locale e statistiche delle maschere |
| `03_metriche.ipynb` | Verifica di matrice di confusione, IoU e F1, confronto con il codice BRIGHT, casi limite e prove sulle maschere locali |

I notebook includono spiegazioni e output salvati.
Le tabelle esportate sono disponibili in:

- `risultati/02_dati_e_split/`
- `risultati/03_metriche/`

## Materiali di riferimento

- BRIGHT: https://github.com/ChenHongruixuan/BRIGHT
- Dataset BRIGHT: https://huggingface.co/datasets/Kullervo/BRIGHT
- Soluzione dei vincitori:
  https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping

Versioni del codice utilizzate:

| Repository | Commit |
|---|---|
| BRIGHT | `59269142f3a3550320513e362692732f46486985` |
| Soluzione dei vincitori | `fe8a8580949dc4e1e38a5a1e3901cee7e20f806b` |

Questi repository sono materiali esterni: il loro codice non è
incluso in questo repository. Restano applicabili le rispettive
licenze e condizioni di utilizzo.

## Ambiente utilizzato

I notebook sono stati eseguiti localmente con:

- Python 3.11.9;
- macOS su architettura arm64;
- VS Code con supporto ai notebook Jupyter.

`requirements.txt` riporta le versioni delle dipendenze rilevate
nell'ambiente utilizzato. L'installazione in un ambiente pulito
non è ancora stata verificata.

Queste dipendenze riguardano i notebook 01–03 e non costituiscono
ancora l'ambiente per l'addestramento della soluzione dei vincitori.

## Preparazione dell'ambiente

Dalla cartella principale del progetto, utilizzando Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name bright-dfc2025 --display-name "Python (BRIGHT DFC2025)"
```

In VS Code selezionare il kernel `Python (BRIGHT DFC2025)`.

## Recupero dei repository esterni

Eseguire questi comandi dalla cartella principale del progetto,
se le cartelle BRIGHT e soluzione_vincitori non sono già presenti:

```bash
git clone https://github.com/ChenHongruixuan/BRIGHT.git BRIGHT
git -C BRIGHT checkout 59269142f3a3550320513e362692732f46486985
```

```bash
git clone https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping.git soluzione_vincitori
git -C soluzione_vincitori checkout fe8a8580949dc4e1e38a5a1e3901cee7e20f806b
```

Il checkout fissa la versione del codice utilizzata nelle analisi.

## Dati ed esecuzione

I dati non sono inclusi nel repository.

Eseguire i notebook nell'ordine 01, 02 e 03, mantenendoli nella
cartella principale del progetto e utilizzando tale cartella
come directory di lavoro.

Il notebook 01 contiene la procedura per recuperare tre esempi
dall'archivio pubblico `dfc25_track2_trainval.zip`, scaricando
i file selezionati mediante RemoteZip.

I file vengono salvati in:

- `dati/bright_esempi/pre-event/`
- `dati/bright_esempi/post-event/`
- `dati/bright_esempi/target/`

Il download richiede accesso a Internet e la disponibilità della
sorgente remota.

Per ciascun notebook, eseguire tutte le celle dall'inizio.
I notebook 02 e 03 utilizzano i dati scaricati dal notebook 01.

## Principali osservazioni

### Suddivisioni

Nella versione analizzata, le liste contengono:

- train: 2.229 identificativi;
- holdout: 661;
- validation: 349;
- test: 156.

Sono presenti due identificativi condivisi:

- `bata-explosion_00000049`: train e holdout;
- `libya-flood_00000024`: train e validation.

Secondo il README ufficiale, train e holdout sono destinati
all'addestramento. Le liste originali non sono state modificate.

Prima degli esperimenti saranno definite e documentate le liste
operative, evitando duplicazioni e condivisioni inappropriate
tra addestramento e valutazione.

### Metriche

Le verifiche hanno evidenziato che:

- l'F1 binario di BRIGHT può restituire NaN quando precisione
  e recall sono entrambe zero;
- una classe assente sia nel riferimento sia nella previsione
  riceve IoU zero nell'implementazione analizzata;
- la media delle metriche per immagine può differire dalla metrica
  calcolata sulla matrice di confusione accumulata;
- la selezione dei pixel valutati cambia il significato del punteggio.

Il notebook 03 distingue il comportamento del codice ispezionato
dal protocollo ufficiale della competizione, ancora da confermare
prima del confronto con i risultati pubblicati.

## Limiti attuali

Le statistiche locali riguardano tre esempi di Bata e non descrivono
l'intero dataset.

Le prove sulle metriche utilizzano previsioni artificiali:
non misurano le prestazioni di una rete.

Nei materiali pubblici controllati non sono stati individuati
i pesi finali della soluzione dei vincitori.
Il relativo codice contiene chiamate esplicite a CUDA.

## Prossimo passo

Eseguire una prima prova con un modello su un campione limitato,
verificando caricamento dei dati, produzione delle previsioni
e valutazione, prima di affrontare esperimenti più estesi.