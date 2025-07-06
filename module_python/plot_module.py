# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 08:36:23 2025

@author: Bruno
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from matplotlib.colors import ListedColormap, BoundaryNorm
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
    
def plot_from_liste_prof(liste_prof, title="Valeur des profondeurs IA et monoplotting", equal=False):
    x=[]
    y=[]
    for mesure in liste_prof:
        x.append(mesure[4])
        y.append(mesure[2])
    fig, ax = plt.subplots()
    
    ax.plot(x, y, '.', alpha=0.3)

    
    plt.xlabel('Valeurs Depth transformées (approx.)', fontsize=12)
    plt.ylabel('Valeurs terrain (dproj)', fontsize=12)
    if equal:
        ax.set_aspect('equal')
    plt.title(title)
    plt.show()
    plt.close()
    

def check_is_array_to_list(arrayliste):
    if type(arrayliste)=="list":
        return arrayliste
    else:
        return arrayliste.flatten().tolist()
    
    
def plot_cluster(X, labels):
    unique_labels = np.unique(labels)
    print(unique_labels)
    for i in range(len(unique_labels)):
    # Extraire les points appartenant au cluster i
        cluster_points = X[labels == i]
        plt.plot(cluster_points[:, 0], cluster_points[:, 1], 'o', label=f'Cluster {i+1}')
        
    plt.show()
    plt.close()


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
    
def show_point_in_image(pathimage, uv , corresp=None, show_plot=True):
    img=Image.open(pathimage)
    max_size = 650
    w, h = img.size
    scale = min(max_size / w, max_size / h)
    new_size = (int(w * scale), int(h * scale))
    # image_origine.show()
    img = img.resize(new_size, Image.LANCZOS)
    
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(0.3)
    
    img_array=np.asarray(img)
    fig, ax = plt.subplots()
    if show_plot:
        ax.imshow(img_array)
    ax.axis('off') 
    
    if corresp is None:
        corresp = np.ones((uv.shape[0],))*2
    
    cmap = ListedColormap(['blue', 'lime', 'orange', 'red'])  # 2, 3, 4
    bounds = [0.5,1.5, 2.5, 3.5, 4.5]  # bornes entre les valeurs
    norm = BoundaryNorm(bounds, cmap.N)
    
    u=uv[:,0]*scale
    v=uv[:,1]*scale
    centre = (u, v)
    rayon = 5
    largeur_bordure = 1 
    
    sc=ax.scatter(u, v, c=corresp, cmap=cmap, vmin=1, vmax=5,s=10, linewidths=0, alpha=0.8)
    cbar = plt.colorbar(sc, ticks=[1,2, 3, 4])
    cbar.ax.set_yticklabels(['1','2', '3', '4'])
    # plt.gca().invert_yaxis()
    
    
def show_only_point_in_image(pathimage, uv , uv_out=None, show_plot=True):
    img=Image.open(pathimage)
    max_size = 650
    w, h = img.size
    scale = min(max_size / w, max_size / h)
    new_size = (int(w * scale), int(h * scale))
    # image_origine.show()
    img = img.resize(new_size, Image.LANCZOS)
    
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(0.3)
    
    img_array=np.asarray(img)
    fig, ax = plt.subplots()

    ax.imshow(img_array)
    ax.axis('off') 

    
    u=uv[:,0]*scale
    v=uv[:,1]*scale
    centre = (u, v)
    rayon = 5
    largeur_bordure = 1 
    
    sc=ax.scatter(u, v, c="lime", alpha=0.8, label="Points avec un wi inférieur à 2.5")
    if uv_out is not None and uv_out.shape[0]>0:
        u_out=uv_out[:,0]*scale
        v_out=uv_out[:,1]*scale
        scout=ax.scatter(u_out, v_out, c="red", alpha=0.8, label="Points avec un wi supérieur à 2.5")
    # plt.legend(loc="lower left")
    
    return fig


def show_only_point_in_image_pillow(pathimage, uv, uv_out=None, point_radius=5):
    # Ouvrir l'image
    img = Image.open(pathimage).convert("RGB")
    
    # Redimensionner
    max_size = 650
    w, h = img.size
    scale = min(max_size / w, max_size / h)
    new_size = (int(w * scale), int(h * scale))
    img = img.resize(new_size, Image.LANCZOS)

    # Réduire la saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(0.3)

    # Dessiner dessus
    draw = ImageDraw.Draw(img)

    # Convertir coordonnées et dessiner les points "verts"
    u = uv[:, 0] * scale
    v = uv[:, 1] * scale
    for x, y in zip(u, v):
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            fill="lime", outline="black"
        )

    # Si des points rouges sont fournis
    if uv_out is not None and uv_out.shape[0] > 0:
        u_out = uv_out[:, 0] * scale
        v_out = uv_out[:, 1] * scale
        for x, y in zip(u_out, v_out):
            draw.ellipse(
                (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
                fill="red", outline="black"
            )

    return img

def save_fig(fig, pathname):
    fig.savefig(pathname)
    
def save_image(image_pil, path):
    image_pil.save(path)
