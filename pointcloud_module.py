# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 08:12:58 2025

@author: Bruno
"""
import numpy as np
import laspy
from scipy.spatial import cKDTree
import open3d as o3d;

       
        
def array_TO_o3d( pointcloudarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloudarray)
    
    return pcd
def o3d_TO_array(pcd_o3d):
    return np.asarray(pcd_o3d.points)

def readlas_to_numpy(path):
    
    las = laspy.read(path)

    # Extraire les coordonnées des points x, y, z
    point_cloud = np.vstack((las.x, las.y, las.z)).transpose()
    
    return point_cloud


def readlas_TO_o3d(path):
    array=readlas_to_numpy(path)
    return array_TO_o3d(array)

def points_inside_mesh(point_cloud, mesh):

    #NECESSITE pip install rtree
    inside_mask = mesh.contains(point_cloud)
    
    points_inside = point_cloud[inside_mask]
    print(f"Nombre de points à l'intérieur du maillage : {len(points_inside)}")

    return points_inside

def save_point_cloud_las(points, output_file):
    """
    Sauvegarde les points filtrés dans un fichier .las.
    
    points_inside: tableau numpy des points à l'intérieur du maillage (n, 3)
    output_file: chemin du fichier de sortie (ex: "output_file.las")
    """
    # Créer un header pour le fichier .laz
    header = laspy.header.LasHeader(version="1.2", point_format=3)
    
    # Créer un objet LasData
    las_data = laspy.LasData(header)
    print(f"Nombre de points réduit : {len(points)}")
    # Ajouter les coordonnées des points à l'objet LasData
    las_data.x = points[:, 0]
    las_data.y = points[:, 1]
    las_data.z = points[:, 2]

    # Enregistrer le fichier .laz
    las_data.write(output_file)
    print(f"Nuage de points enregistré dans {output_file}")
    
def reduction_nuage_nbre(points, nbre=1000):
    """
    Sous-échantillonnage aléatoire des points.

    points: tableau numpy des points du nuage de points (n, 3)
    fraction: pourcentage des points à garder (0.1 = 10%)
    
    Retourne un sous-ensemble des points sélectionnés aléatoirement.
    """
    num_points = points.shape[0]
    num_samples = nbre
    
    # Sélectionner aléatoirement les indices
    indices = np.random.choice(num_points, num_samples, replace=False)
    
    # Sélectionner les points correspondants
    sampled_points = points[indices]
    
    return sampled_points




def reduction_nuage_voxel(points, voxel=2):
    """
    Sous-échantillonnage aléatoire des points.

    points: tableau numpy des points du nuage de points (n, 3)
    fraction: pourcentage des points à garder (0.1 = 10%)
    
    Retourne un sous-ensemble des points sélectionnés aléatoirement.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel)
    
    return np.asarray(voxel_down_pcd.points)

def suppression_point_isole(point_cloud, k=2):
    """
    Suppression des points isolés d'un nuage de points en fonction de la proximité avec les autres points.

    points: tableau numpy des points du nuage de points (n, 3)
    fraction: pourcentage des points à garder (0.1 = 10%)
    
    Retourne un sous-ensemble des points sélectionnés aléatoirement.
    """

    tree = cKDTree(point_cloud)

    # Tableau pour garder les indices des points non isolés
    non_isolated_indices = []
    isolated_indices = []
    
    d_proche1=[]
    diff12=[]
    fact=[]
    for i, point in enumerate(point_cloud):
        distances, indices = tree.query(point, k=k+1)  # Rechercher k+1 voisins (incluant le point lui-même)
        d_proche1.append(distances[1])
        diff12.append(distances[2]-distances[1])
        
    mean_d=np.mean(d_proche1)
    std_d=np.std(d_proche1)
    mean_diff12=np.mean(diff12)
    std_diff12=np.std(diff12)

    
    for i, point in enumerate(point_cloud):
        if d_proche1[i]>1 and diff12[i]>0.2:
            isolated_indices.append(i)
        else:
            non_isolated_indices.append(i)
    
    non_isolated_point=point_cloud[non_isolated_indices]
    isolated_point=point_cloud[isolated_indices]
    # save_point_cloud_las(isolated_point, "isolated_point.las")
    # save_point_cloud_las(non_isolated_point, "non_isolated_point.las")
    
    return non_isolated_point
    
def remove_statistical_outlier_pcd(point_cloud):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    print("Statistical oulier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=0.1)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    
    # Appliquer des couleurs
    inlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # Vert
    outlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # Rouge

    return np.asarray(inlier_cloud.points)
        

    
        
    