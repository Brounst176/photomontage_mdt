# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 08:12:58 2025

@author: Bruno
"""
import numpy as np
import laspy
from scipy.spatial import cKDTree
import open3d as o3d;
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

        
def array_TO_o3d(pointcloudarray, normales=None, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloudarray)
    if normales is not None:
        pcd.normals = o3d.utility.Vector3dVector(normales)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    
    return pcd
def o3d_TO_array(pcd_o3d):
    
    return np.asarray(pcd_o3d.points), np.asarray(pcd_o3d.normals)

def readlas_to_numpy(path):
    
    las = laspy.read(path)
    # print(las.point_format)
    # print(las.point_records)
    # Extraire les coordonnées des points x, y, z

    point_cloud = np.vstack((las.x, las.y, las.z)).transpose()
    normal_pcd = np.vstack((las.points["normal x"], las.points["normal y"], las.points["normal z"])).transpose()
    normal_rgb = np.vstack((las.points["red"], las.points["green"], las.points["blue"])).transpose()
    
    

    # Afficher les formats du nuage de points
    # for dimension in las.point_format.dimensions:
    #     print(dimension.name)
    
    return point_cloud, normal_pcd


def readlas_TO_o3d(path):
    array=readlas_to_numpy(path)
    return array_TO_o3d(array)

def points_inside_mesh(point_cloud, normal, mesh):

    #NECESSITE pip install rtree
    inside_mask = mesh.contains(point_cloud)
    
    points_inside = point_cloud[inside_mask]
    normals_inside=normal[inside_mask]
    print(f"Nombre de points à l'intérieur du maillage : {len(points_inside)}")

    return points_inside, normals_inside

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
    print(f"Nombre de points : {len(points)}")
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




def reduction_nuage_voxel(o3d_point, voxel=2):
    """
    Sous-échantillonnage aléatoire des points.

    points: tableau numpy des points du nuage de points (n, 3)
    fraction: pourcentage des points à garder (0.1 = 10%)
    
    Retourne un sous-ensemble des points sélectionnés aléatoirement.
    """

    voxel_down_pcd = o3d_point.voxel_down_sample(voxel_size=voxel)
    
    return voxel_down_pcd

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
    
def remove_statistical_outlier_pcd(pcd):
    
    
    print("Statistical oulier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=0.1)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    
    # Appliquer des couleurs
    inlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # Vert
    outlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # Rouge
    o3d.io.write_point_cloud("inlier_cloud.ply", inlier_cloud)
    o3d.io.write_point_cloud("outlier_cloud.ply", outlier_cloud)
    return inlier_cloud
        


def LABELISATION_POINTSCLOUD(o3d_point):
    segment_models={}
    segments={}
    max_plane_idx=10
    
    rest=o3d_point
    for i in range(max_plane_idx):
        colors = plt.get_cmap("tab20")(i)
        print(f"Nombre de points dans le nuage de points : {len(rest.points)}")
        if len(rest.points)<10:
            break
        segment_models[i], inliers = rest.segment_plane(
        distance_threshold=1,ransac_n=3,num_iterations=1000)
        segments[i]=rest.select_by_index(inliers)
        labels = np.array(segments[i].cluster_dbscan(eps=4, min_points=150))
        candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]
        best_candidate=int(np.unique(labels)[np.where(candidates== np.max(candidates))[0]])
        rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(list(np.where(labels!=best_candidate)[0]))
        segments[i]=segments[i].select_by_index(list(np.where(labels== best_candidate)[0]))
        segments[i].paint_uniform_color(list(colors[:3]))
        print("pass",i,"/",max_plane_idx,"done.")
        
    labels = np.array(rest.cluster_dbscan(eps=2, min_points=50))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    rest.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])
    return segments, segment_models[i]


def calcul_normal_point_cloud(pointcloud, normales=None):
    o3d_pcd=array_TO_o3d(pointcloud)
    o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    
    
    calculated_normals = np.asarray(o3d_pcd.normals)

    # Normaliser les normales calculées (pour éviter les erreurs de longueur)
    calculated_normals = calculated_normals / np.linalg.norm(calculated_normals, axis=1)[:, np.newaxis]
    
    # Normaliser les directions approximatives
    directions = normales / np.linalg.norm(normales, axis=1)[:, np.newaxis]
    
    # Ajuster les normales calculées pour qu'elles aient le même sens que les normales approximatives
    # Cela consiste à comparer chaque direction approximative avec la normale recalculée et inverser si nécessaire
    for i in range(len(directions)):
        if np.dot(directions[i], calculated_normals[i]) < -0.3:
            calculated_normals[i] = -calculated_normals[i]
        elif np.dot(directions[i], calculated_normals[i]) < 0.3:
            calculated_normals[i] = np.array([0,0,1])
    
    # Assigner les normales ajustées au nuage de points
    # o3d_pcd.normals = o3d.utility.Vector3dVector(calculated_normals)
    
    return calculated_normals
    
    

def view_point_cloud_from_array(pointcloud, normales=None, color=None):
    x=np.median(pointcloud[:,0])
    y=np.median(pointcloud[:,1])
    z=np.median(pointcloud[:,2])
    
    
    pointcloud[:,0]=pointcloud[:,0]-x
    pointcloud[:,1]=pointcloud[:,1]-y
    pointcloud[:,2]=pointcloud[:,2]-z
    
    o3d_pcd=array_TO_o3d(pointcloud, normales, color)

        
    
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(o3d_pcd)
    if normales is not None:
        visualizer.get_render_option().point_show_normal = True
    # Obtenir les options d'affichage et ajuster la taille des normales
    opt = visualizer.get_render_option()
    opt.point_size = 5  # Réduit la taille des points (si nécessaire)
    opt.line_width = 0.5  # La largeur des lignes pour les normales
    visualizer.run()


# def DBSCAN_OR_KMEANS(point_cloud, min_samples, n_neig=2)
    
def DBSCAN_pointcloud(point_cloud, min_samples=10, n_neig=2):
    X=point_cloud
    neigh = NearestNeighbors(n_neighbors=n_neig)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,n_neig-1]
    # plt.plot(distances);
    mean=np.mean(distances)
    pourc90=int(distances.shape[0]*0.90)

    y_pred = DBSCAN(eps = distances[pourc90], min_samples=min_samples).fit_predict(X)
    unique_clusters = np.unique(y_pred)
    np.delete(unique_clusters, 0)
    
    clusters = [X[y_pred == cluster_id] for cluster_id in unique_clusters]
    return clusters, y_pred
    

def KMEANS_pointcloud(array, interie_fact=0.1):
    X=array
    inertias = []
    diff=[]
    for i in range(1,9):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    n=0
    for i in range(len(inertias)):
        if i == len(inertias)-2:
            n=i
            break
        a=np.array([i+1, inertias[i+1]])
        b=np.array([i, inertias[i]])
        PS=a-b
        N=np.array([inertias[i+2]-inertias[i], i+2-i])
        norm_N=np.linalg.norm(N)
        norm_PS=np.linalg.norm(PS)
        norm_proj=np.abs((PS@N)/(norm_PS*norm_N)*norm_PS)
        
        if norm_proj>interie_fact*norm_N:
            n=i
            break
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    y_pred=kmeans.labels_
    unique_clusters = np.unique(y_pred)
    clusters = [X[y_pred == cluster_id] for cluster_id in unique_clusters]
    return clusters, y_pred


def HDBSCAN_pointcloud(point_cloud):
    X=point_cloud

    hdb = HDBSCAN(min_samples=1).fit(X)
    y_pred=hdb.labels_
    # plot(X, hdb.labels_, hdb.probabilities_)
    unique_clusters = np.unique(y_pred)

    clusters = [X[y_pred == cluster_id] for cluster_id in unique_clusters]
    return clusters, y_pred
    
    