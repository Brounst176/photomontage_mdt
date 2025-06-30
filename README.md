
# Manuel

Le code a été développé avec la version **3.10** du langage de programmation **Python**. Il est disponible sur GitHub.

Les codes suivants ont été repris pour les traitements avec l’intelligence artificielle :

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)  
- [LightGlue](https://github.com/cvg/LightGlue)

---

## Installation

1. Cloner le dépôt GitHub :
    ```bash
    git clone https://github.com/cvg/LightGlue.git
    ```

2. Créer un environnement virtuel et installer les dépendances :
    ```bash
    python -m venv virtual
    virtual\Scripts\activate
    python -m pip install -r requirements.txt
    ```

3. Télécharger le modèle pré-entraîné pour la création des cartes de profondeur IA :  
   [depth_anything_v2_vitl.pth (HuggingFace)](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)

   Place-le dans le dossier `checkpoints`.

---

## Manuel utilisateur

### Démarrer l'application web

Lancer l’application depuis l’invite de commande après avoir activé l’environnement virtuel :

```bash
python flask_run.py
```

L'application web sera accessible à l'adresse : [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### Tutoriel

Une vidéo tutorielle est disponible :

- directement sur le serveur web
- ou dans le fichier : `appviewer/static/image/tutoriel.mp4`

---

### Création d’un nouveau projet de photomontage

Des exemples de dossiers de projet sont disponibles dans le répertoire `data_projet`.

#### Structure du dossier

```
ProjetName/
│   emprise.obj
│   CalibExemple.xml
│   image.png
│   pointcloud.las
│   position_orientation.txt
│   projet.obj
│
├── image/
│   ├── image1.JPG
│   ├── image2.JPG
│   └── ...
│
└── Output/
    └── ...
```

#### Contenu requis

- `emprise.obj` : Emprise du projet (zone considérée comme démolie)
- `pointcloud.las` : Nuage de points de l’existant
- `image.png` : Image du projet
- `position_orientation.txt` : Positions et orientations des caméras exportées depuis Agisoft (avec modifications)
- `projet.obj` : Maquette 3D du projet
- `image/` : Images du modèle photogrammétrique
- `Output/` : Résultats des photomontages
- `CalibExemple.xml` : Fichier de calibration exporté depuis Agisoft (plusieurs fichiers possibles)

#### Exemple de `position_orientation.txt`

Export depuis Agisoft avec format **Omega Phi Kappa**, puis ajout de deux éléments :

- le **format de l’image** (ex: `.JPG`)
- le **nom de la calibration**

```txt
# Cameras (35)
# PhotoID, X, Y, Z, Omega, Phi, Kappa, r11, ..., r33
_DSC7918.JPG  585.159  291.145  445.99  -97.12  27.89  -186.07  ...  CalibExemple
DJI_1564.JPG  578.179  282.249  446.09  -98.22  26.79  -176.07  ...  CalibExemple2
...
```

---

### Enregistrement d’un projet dans l'application

Modifier le fichier `appviewer/projet.json` pour ajouter un nouveau projet.

Exemple :

```json
{
  "ProjetName": {
    "objet": "Description du projet de construction",
    "coordinate": "POINT(2528496 1159644)",
    "calibration": ["CalibExemple", "CalibExemple2"],
    "projet": "MultiPolygon (((2540574.18 1181272.83, 2540575.59 1181271.19, 2540577.105 1181272.49, 2540575.71 1181274.11, 2540574.18 1181272.83)))"
  }
}
```

- `coordinate` : point central du projet au format **WKT**
- `calibration` : liste des fichiers de calibration associés
- `projet` : géométrie du projet au format **WKT MultiPolygon**

Pour plus d’infos sur le format WKT : [libgeos.org/specifications/wkt](https://libgeos.org/specifications/wkt/)

---

## Manuel développeur

### Dossiers du code

- `appviewer/` : Interface web (Flask)
- `data_projet/` : Données des projets de photomontage
- `module_python/` : Modules Python pour les traitements (photomontage, IA, etc.)
- `checkpoints/` : Modèles pré-entraînés IA (cartes de profondeur)
