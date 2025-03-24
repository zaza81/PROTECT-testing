# Testing generazione Modello AI con Gestione delle Licenze

Questo progetto implementa un modello di AI che include la gestione delle licenze attraverso i metadati EXIF. E' un dimostrativo di PROTECT del progetto Holmes, che garantisce un corretto tracciamento delle licenze per ogni immagine - dimonstrando come si possa generare dei modelli di AI in rispetto del Copyright e delle licenze per ciascun file.

## Caratteristiche

- Classificazione di immagini utilizzando un modello CNN
- Gestione delle licenze attraverso metadati EXIF
- Supporto per immagini riproducibili e protette da copyright
- Filtraggio del dataset basato sullo stato della licenza

## Struttura del Progetto

```
.
├── license_checker.py    # Funzioni per la gestione delle licenze delle immagini
├── model_builder.py      # Codice per la costruzione e l'addestramento del modello
├── main.py              # Script principale per eseguire la pipeline
├── requirements.txt     # Dipendenze del progetto
└── README.md           # Questo file
```

## Installazione

1. Clona questo repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Crea un ambiente virtuale (consigliato):
```bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
```

3. Installa le dipendenze richieste:
```bash
pip install -r requirements.txt
```

## Utilizzo

### Preparazione del Dataset

Posiziona le tue immagini di input nella seguente struttura:
```
intel-image-classification/
└── seg_train/
    └── seg_train/
        ├── buildings/
        │   └── [file immagine]
        └── forest/
            └── [file immagine]
```

### Esecuzione della Pipeline

1. Esegui lo script principale:
```bash
python main.py
```

Questo:
- Elaborerà tutte le immagini e aggiungerà i metadati delle licenze
- Filtrerà per mantenere solo le immagini riproducibili
- Costruirà e addestrerà il modello
- Salverà il modello addestrato
- Testerà il modello con un'immagine di esempio

### Personalizzazione

Puoi modificare i seguenti parametri in `main.py`:
- `input_dir`: Directory contenente le immagini di input
- `output_dir`: Directory per le immagini elaborate
- `classes`: Lista delle categorie di immagini
- `reproducible_limit`: Numero di immagini da contrassegnare come riproducibili per classe

## Gestione delle Licenze

Il sistema supporta due tipi di licenze:
- "riproducibile": Immagini che possono essere riprodotte
- "diritto d'autore: copia negata": Immagini protette da copyright

Le immagini vengono automaticamente etichettate con le licenze in base alla loro posizione nel dataset, e solo le immagini riproducibili vengono utilizzate per l'addestramento.

## Architettura del Modello

Il modello CNN è composto da:
- Layer di input con ridimensionamento dell'immagine
- Tre blocchi convolutivi (Conv2D + MaxPooling2D)
- Layer di appiattimento
- Due layer densi (128 unità + layer di output)

## Output

Il modello addestrato viene salvato nella directory `saved_model` come `my_model.keras`.

## Dipendenze

- TensorFlow 2.15.0 o superiore
- TensorFlow Datasets 4.9.4 o superiore
- Pillow 10.2.0 o superiore
- piexif 1.1.3 o superiore
- exifread 3.0.0 o superiore
- NumPy 1.24.0 o superiore
- Matplotlib 3.8.0 o superiore
