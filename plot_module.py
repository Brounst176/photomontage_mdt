# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 08:36:23 2025

@author: Bruno
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Fonction pour créer et mettre à jour l'histogramme
def plot_histogram(values, interval=10, title="titre"):
    """
    Crée un diagramme en barres des valeurs dans des intervalles.

    Args:
        values (list or np.ndarray): Liste des valeurs à afficher.
        interval (int): Taille de l'intervalle  (par défaut 10).
    """
    # Calcul des intervalles (en divisant les valeurs par l'intervalle)
    min_value = min(values)
    max_value = max(values)
    
    # Création des bornes des intervalles
    bins = np.arange(min_value, max_value + interval, interval)
    
    # Création de l'histogramme
    plt.hist(values, bins=bins, edgecolor='black', alpha=0.7)
    
    plt.title(title)
    
    # Affichage
    plt.show()

def plot_mesure_calcule(val_x, val_mesure, val_calcul, title, x_dep=None, y_dep=None):
    """
    Crée graphiques des valeurs observées et des valeurs calculés

    Args:
        val_x (list or np.ndarray): Liste des valeurs de l'axe des x
        val_mesure (list or np.ndarray): Liste des valeurs observées
        val_calcul (list or np.ndarray): Liste des valeurs calculés par moindre carré
    """
    val_x=check_is_array_to_list(val_x)
    val_mesure=check_is_array_to_list(val_mesure)
    val_calcul=check_is_array_to_list(val_calcul)
    
    
    fig, ax = plt.subplots()
    
    
    
    
    ax.plot(val_x, val_calcul, '.', label='Moindre carré')
    ax.plot(val_x, val_mesure, '.',label='Monoplotting', alpha=0.3)
    if x_dep!=None and y_dep!=None:
        ax.plot(x_dep, y_dep, '.', label='Pts non utilisé', alpha=0.5)
    
    plt.xlabel('Valeurs Depth IA', fontsize=12)
    plt.ylabel('Valeurs terrain', fontsize=12)
    # ax.plot(inc, B)
    plt.legend()
    plt.title(title)
    plt.show()
    plt.close()
    
def plot_from_liste_prof(liste_prof):
    x=[]
    y=[]
    for mesure in liste_prof:
        x.append(mesure[4])
        y.append(mesure[2])
    fig, ax = plt.subplots()
    
    ax.plot(x, y, '.', alpha=0.3)

    
    plt.xlabel('Valeurs Depth IA', fontsize=12)
    plt.ylabel('Valeurs terrain', fontsize=12)
    # ax.plot(inc, B)
    # plt.legend()
    plt.title("Valeur des profondeurs IA et monoplotting")
    plt.show()
    plt.close()
    

def check_is_array_to_list(arrayliste):
    if type(arrayliste)=="list":
        return arrayliste
    else:
        return arrayliste.flatten().tolist()
    
    



def input_10_wi_to_image(path_image, liste_uv_obs, val_mesure, val_calcul, wi, nb_wi=10):
    """
    Insert dans un image les x plus grand Wi et les valeurs observées et les valeurs calculés

    Args:
        path_image: chemin de l'image
        liste_uv_obs (list): Liste des observations
        val_mesure (np.ndarray): Liste des valeurs observées
        val_calcul (np.ndarray): Liste des valeurs calculés par moindre carré
        wi (np.ndarray): liste des WI
    """
    
    
    #Sort les 10 plus grande valeurs
    indices = np.argsort(np.abs(wi)[:, 0])[-nb_wi:][::-1]

    image_origine=Image.open(path_image)
    draw = ImageDraw.Draw(image_origine)
    font = ImageFont.load_default() 
    for index in indices:
        u=liste_uv_obs[index][0]
        v=liste_uv_obs[index][1]
        centre = (u, v)  # Position du centre du cercle
        rayon = np.abs(wi[index])*2           # Rayon du cercle
        largeur_bordure = 1  # Largeur de la bordure du cercle
    
        # # Dessiner le cercle (contour seulement)
        draw.ellipse(
            [centre[0] - rayon, centre[1] - rayon, centre[0] + rayon, centre[1] + rayon],
            outline="red",   # Couleur de la bordure
            width=largeur_bordure  # Largeur de la bordure
        )
        
        
        draw.text((u, v), "Obs:"+"{:.1f}".format(val_mesure[index,0])+"/"+"C:"+"{:.1f}".format(val_calcul[index,0]), font=font, fill=(0, 0, 0))
    
    
    image_origine.show()