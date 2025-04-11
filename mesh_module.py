# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 14:12:29 2025

@author: Bruno
"""

import trimesh
import numpy as np
from PIL import Image
from numba import njit #OPTIMISATION DES CALCULS

def intersection_obj_vecteur(point, direction, mesh):
    # mesh = obj_trimesh
    
   
    # Normaliser la direction du rayon
    direction = direction / np.linalg.norm(direction)
    
    # Convertir le point et la direction en tableaux 2D (1 rayon)
    point = np.reshape(point, (1, 3)) # Forme (1, 3)
    direction = np.reshape(direction, (1, 3)) # Forme (1, 3)
    # Calculer les intersections
    locations, _, index_tri = mesh.ray.intersects_location(
        ray_origins=point, 
        ray_directions=direction
    )

    # Si des intersections existent
    if len(locations) == 0:
        return None
        # Trouver l'intersection la plus proche (la plus petite distance)
    deltas=locations - point
    distances = np.einsum('ij,ij->i', deltas, deltas)
    closest_idx = np.argmin(distances)
    closest_location = locations[closest_idx]
    closest_triangle_idx = index_tri[closest_idx]
    
    # Obtenir la normale du triangle d'intersection
    normal = mesh.face_normals[closest_triangle_idx]
    
    # Obtenir les indices des sommets du triangle d'intersection
    vertex_indices = mesh.faces[closest_triangle_idx]
    
    # Récupérer les coordonnées des sommets et les coordonnées UV associées
    # vertices = mesh.vertices[vertex_indices]
    uvs = mesh.visual.uv[vertex_indices]  
    uv=np.array([[uvs[0,0], uvs[0,1]]])
    color=mesh.visual.material.to_color(uv)
    
    face_color=tuple(color[0])
    # if mesh.visual.face_colors is not None:
    #     face_color = mesh.visual.face_colors[closest_triangle_idx]
    # else:
    # face_color = None 
    
    return [closest_location, distances[closest_idx], normal, face_color]
    # print(f"Intersection la plus proche: {closest_location}")
        # print(f"Normale du triangle d'intersection: {normal}")

        
        
        
        
        
#GESTION DES COULEURS
        
        
def adjust_brightness(color, factor):
    # Assurer que les composantes sont dans la plage [0, 255]
    r, g, b = color
    r = min(max(int(r * factor), 0), 255)
    g = min(max(int(g * factor), 0), 255)
    b = min(max(int(b * factor), 0), 255)
    return (r, g, b)

# Fonction pour ajuster la couleur de base en fonction des normales
def color_from_normal(base_color, normal, light_direction):
    # Normaliser les vecteurs
    light_dir = np.array(light_direction)
    light_dir = light_dir / np.linalg.norm(light_dir)
    

    
    
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # Normaliser la normale
    
    # Calculer le produit scalaire entre la normale et la direction de la lumière
    intensity = np.dot(normal, light_dir)
    
    # Limiter l'intensité à la plage [0, 1]
    intensity = max(0.2, min(1.0, intensity))
    
    # Ajuster la luminosité de la couleur de base
    adjusted_color = adjust_brightness(base_color, intensity)

    
    return adjusted_color




# Fonction pour normaliser un vecteur
@njit
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

# Fonction pour calculer l'éclairage (diffus + spéculaire + ambiant)
@njit
def phong_lighting(normal, light_dir, view_dir, surface_color, ambient_intensity, light_intensity, specular_intensity, shininess):
    # Normaliser les vecteurs
    normal = normalize(normal)
    light_dir = normalize(light_dir)
    view_dir = normalize(view_dir)
    
    # Calcul de l'éclairement diffus (Lambertien)
    diffuse = max(np.dot(normal, light_dir), 0)  # Diffuse component
    
    # Calcul de l'éclairement spéculaire (Phong)
    reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir  # Direction de la réflexion
    specular = max(np.dot(reflect_dir, view_dir), 0) ** shininess  # Specular component
    
    # Calcul de l'éclairement ambiant
    ambient = ambient_intensity  # Lumière ambiante constante

    # Combinaison des trois composantes
    color = np.array(surface_color) * (ambient + light_intensity * diffuse + specular_intensity * specular)
    color = np.clip(color, 0, 255)  # Limiter les valeurs à [0, 255] pour les couleurs RGB

    # Conversion des valeurs en entiers sans utiliser astype
    color_int = np.round(color).astype(np.int32)
    
    return [color_int[0],color_int[1],color_int[2]]

# Exemple de vecteurs et paramètres
normal = np.array([0, 0, 1])  # Normale de la surface (ex. normale vers le haut)
light_dir = np.array([1, 1,0.5])  # Direction de la lumière
view_dir = np.array([0, 0, 1])  # Direction de l'observateur (par exemple, vers l'avant)
surface_color = (255, 0, 0)  # Couleur de la surface (rouge)
ambient_intensity = 0.1  # Intensité de la lumière ambiante
light_intensity = 1.0  # Intensité de la lumière diffusée
specular_intensity = 0.5  # Intensité de la lumière spéculaire
shininess = 25  # Paramètre de brillance (plus il est grand, plus l'éclat est petit)

# mesh = trimesh.load_mesh("Input/cabane/cabane_texture.obj")
# print(mesh.visual)
# mesh.show()

# # if mesh.visual.material is not None:
# #     # Accéder au matériau unique
# #     for i, material in enumerate(mesh.visual.material):
# #         # material = mesh.visual.material
# #         print(f"Nom du matériau : {material.name}")
# #         print(f"  Couleur ambiante (Ka) : {material.ambient}")
# #         print(f"  Couleur diffuse (Kd) : {material.diffuse}")
# #         print(f"  Couleur spéculaire (Ks) : {material.specular}")
# #     # print(f"  Transparence (Tr) : {material.transparency}")
# # else:
# #     print("Aucun matériau trouvé dans le maillage.")

# # # if mesh.visual.material is not None:
# # #     print("Matériaux définis :")

# # #     # Parcourir les matériaux
# # #     for material in mesh.visual.material:
# # #         print(f"Nom du matériau : {material.name}")
# # #         print(f"  Couleur ambiante (Ka) : {material.ambient}")
# # #         print(f"  Couleur diffuse (Kd) : {material.diffuse}")
# # #         print(f"  Couleur spéculaire (Ks) : {material.specular}")
# # #         print(f"  Transparence (Tr) : {material.transparency}")
# # # else:
# # #     print("Aucun matériau trouvé dans le maillage.")
# # # Vérifier si le maillage a des couleurs associées
# print(mesh.visual.face_subset(0).uv)

# print(mesh.visual.material.to_color(mesh.visual.uv))
# if mesh.visual.uv is not None:
#     # Récupérer les coordonnées UV de la première face
#     print("Coordonnées UV du maillage :")
#     # Afficher les coordonnées UV de la première face (par exemple)
#     for i, uv in enumerate(mesh.visual.uv):
#         uv_array=np.array([[uv[0], uv[1]]])
#         # uv_array
#         color=mesh.visual.material.to_color(uv_array)
#         print(f"Face {i}: UV = {uv}")
# else:
#     print("Aucune coordonnée UV trouvée.")