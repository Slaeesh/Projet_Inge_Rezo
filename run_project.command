#!/bin/bash

# Se déplacer dans le dossier où se trouve le script (pour que ça marche au double-clic)
cd "$(dirname "$0")"

echo "Activation de l'environnement virtuel..."
source .venv/bin/activate

echo "Lancement du programme..."
python main.py
