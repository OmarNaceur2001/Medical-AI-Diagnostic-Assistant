# Breast Cancer UI — Diagnostic Assisté par IA

Application web de démonstration basée sur **FastAPI** pour l’analyse d’images médicales avec **PyTorch**.  
Le projet expose deux modèles de classification pré-entraînés, avec génération d’explications visuelles via **Grad-CAM** :

- **Breast histopathology** : classification binaire `Benign / Malignant`
- **Brain MRI** : classification multi-classes `Glioma / Meningioma / No Tumor / Pituitary`

> ⚠️ Ce projet est destiné à la recherche, au prototypage et à la démonstration. Il ne remplace pas un avis médical ni un dispositif validé en production clinique.

## Fonctionnalités

- Interface web légère pour téléverser une image et obtenir une prédiction instantanée
- Deux parcours de prédiction distincts : sein et cerveau
- Retour des probabilités de classe
- Visualisation Grad-CAM pour comprendre les zones les plus influentes dans la décision
- API simple compatible avec des intégrations futures côté front-end ou services tiers

## Architecture

Le serveur charge deux modèles au démarrage :

- `model/final_model.pth` pour le cas **breast**
- `model/brain_model.pth` pour le cas **brain**

Le pipeline de traitement comprend :

1. Chargement de l’image au format RGB
2. Prétraitement standardisé avec `Resize(160, 160)` et normalisation ImageNet
3. Inférence avec **EfficientNet-B0**
4. Calcul des probabilités de sortie
5. Génération d’une carte **Grad-CAM**
6. Retour de la prédiction au format JSON

## Stack technique

- **Backend** : FastAPI, Uvicorn
- **Deep Learning** : PyTorch, TorchVision
- **Image processing** : Pillow, OpenCV, NumPy
- **Visualisation** : Matplotlib
- **Interface** : HTML/CSS/JS servie par FastAPI

## Prérequis

- Python 3.10+ recommandé
- Les fichiers de modèles doivent être présents dans le dossier `model/`
- Les images doivent être fournies au format `multipart/form-data`

## Installation

### 1. Créer un environnement virtuel

```bash
python -m venv .venv
```

### 2. Activer l’environnement

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install fastapi uvicorn torch torchvision opencv-python matplotlib pillow python-multipart numpy
```

## Lancement

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Puis ouvrir :

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Utilisation

1. Ouvrir l’interface web
2. Choisir le mode :
   - **Breast histopathology**
   - **Brain MRI**
3. Importer une image
4. Lancer la prédiction
5. Consulter :
   - la classe prédite
   - le niveau de confiance
   - la heatmap Grad-CAM

## API

### `GET /`

Retourne l’interface web principale.

### `POST /predict/breast`

Prédit `Benign` ou `Malignant` à partir d’une image histopathologique.

**Champ attendu**

- `image` : fichier image

### `POST /predict/brain`

Prédit l’une des 4 classes cérébrales :

- `Glioma`
- `Meningioma`
- `No Tumor`
- `Pituitary`

**Champ attendu**

- `image` : fichier image

## Réponses API

Les endpoints de prédiction renvoient un JSON contenant :

- `prediction` : classe prédite
- `confidence` : confiance en pourcentage
- `gradcam_image` : image Grad-CAM encodée en base64

Pour le modèle cerveau, la réponse inclut aussi :

- `all_probs` : distribution complète des probabilités par classe

## Structure du projet

```text
.
├── app.py
├── model/
│   ├── final_model.pth
│   └── brain_model.pth
├── static/
├── templates/
│   └── index.html
└── README.md
```

## Notes de conception

- Le chargement des modèles est effectué au démarrage du serveur.
- Le front-end utilise une seule interface pour commuter entre les deux modes.
- Les cartes Grad-CAM sont générées côté serveur et renvoyées directement à l’interface.
- Le projet peut servir de base pour une future intégration PACS / DICOM ou pour exposer d’autres modèles médicaux.

## Limites

- Le système n’est pas conçu pour un usage clinique sans validation réglementaire.
- Les prédictions dépendent fortement de la qualité, de la modalité et de la cohérence des images fournies.
- Les seuils et les modèles fournis doivent être recalibrés avant tout déploiement réel.

## Licence

À définir selon l’usage du projet et les contraintes de diffusion.

