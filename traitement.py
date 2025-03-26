import numpy as np

import plot_module as plot
import camera_module as cm
import pointcloud_module as pcd_m
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
import shapely
from shapely.geometry import Point, Polygon
import time
#Points sur la maison devant
M=np.array([
    2528510.114,
    1159666.963,
    511.78
    ])
#Points sur terrainde pétanque
M=np.array([
     2528500.692,
     1159665.769,
     510.55
     ])
#Points 3
M=np.array([
     2528510.323,
     1159666.941,
     511.75
     ])
# Fenetre bas
M=np.array([
     2528603.686,
     1159679.230,
     510.20
     ])
#Faite simone
# M=np.array([
#      2528524.957,
#      1159666.411,
#      518.316
#      ])
#fenetre simone
M=np.array([
     2528522.8052,
     1159668.8328,
     511.6924
     ])



nikon=cm.camera("NIKON D7500", "foldernotexite", 5568, 3712, -55.72194146069377, 23.384728774951231,4231.8597199545738,-0.14699248766634521,0.1616307088544226,-0.023431362387691224,0.0,-0.00074350800851607905,0.00052372914373731753,0.0,0.0)
fichier_path = 'C:/Users/Bruno/Documents/TM_script/Terrain/camera_ORIENTAITON.txt'
nikon.import_image_from_omega_phi_kappa_file(fichier_path)
pathlas="C:/Users/Bruno/Documents/TM_script/Terrain/point_homologue.las"
pathlas="C:/Users/Bruno/Documents/TM_script/Terrain/point_dense_reduce.las"
depthanything=cm.depthmap("_DSC6987_50.tif", "_DSC6987",pathlas, True, nikon )
dict_prof=depthanything.dict_prof
list_prof=depthanything.dict_prof_TO_liste(dict_prof)
array_prof=np.array(list_prof)

photoname="_DSC6987"
#%% Calcul pour contrôle fonction de photogrammétrie
m=nikon.M_to_uv("_DSC6987", M)
d=np.linalg.norm(M-nikon.images["_DSC6987"]["S"])
vect_MS=M-nikon.images["_DSC6987"]["S"]
M_calc=nikon.uv_to_M(photoname, m, d)

H, d_proj=nikon.calcul_proj_cam("_DSC6987",M)

# print(f"Dist S-M : {d}")
# pointcloud, rgb = pcd_m.readlas_to_numpy(pathlas)

M_calc_d_projet=nikon.uv_to_M_by_dist_prof(photoname, m, d_proj)

#%%CALCUL PAR SEPRATION DE GAUCHE A DROITE
# plot.plot_from_liste_prof(list_prof)

# depthanything.transformation_simple_depthmap_IA()

# plot.plot_from_liste_prof(list_prof)

# depthanything.initialisation_calc_par_groupe_colonne_pour_moindre_carre()



#%%CALCUL PAR CLUSTER DE ZONE 
#===================================================================================================================================
#===================================================================================================================================
#CREATION D ARRAY DES COORDONNES DES POINTS ET DE LA VALEUR DE PROFONDEUR DE LA DEPTH MAP
# x=np.column_stack((np.array(array_prof[:,5].tolist()),array_prof[:,4]))
# clusters, y_pred=pcd_m.DBSCAN_pointcloud(x, min_samples=4, n_neig=3)
# depthanything.clusters=clusters
# pcd_m.readlas_to_numpy(pathlas)
start_time = time.time()
print(start_time)
cluster_liste=depthanything.optimisation_des_clusters()
param_cluster=depthanything.param_transfo_cluster

end_time = time.time()
print(end_time)
elapsed_time = end_time - start_time
print(f"Durée d'exécution de calcul de Clusters : {elapsed_time:.2f} secondes")



# unique_labels = np.unique(y_pred)
# mask_noise=y_pred==-1  #CHOIX DU CLUSTER -1 correspond au point de bruit
# 
# CHOIX DE COULEUR PAR CLUSTER
# colors = plt.cm.get_cmap("tab10", len(unique_labels))
# point_colors = np.array([colors(label)[:3] for label in y_pred]) 
# array_prof_hdbscan=x[~mask_noise] #Valeur ne correspond pas au masque de bruit


# pcd_m.view_point_cloud_from_array(np.delete(array_prof_hdbscan, 3, axis=1), color=point_colors[~mask_noise])
# pcd_m.save_point_cloud_las(array_prof_hdbscan, "nuage_dbscan_cluster.las")


# array_prof_hdbscan=x[mask_noise] #Valeur ne correspond pas au masque de bruit
# pcd_m.view_point_cloud_from_array(np.delete(array_prof_hdbscan, 3, axis=1))
# pcd_m.save_point_cloud_las(array_prof_hdbscan, "nuage_dbscan_noise.las")
# plt.figure(2)
# plt.scatter(array_prof[:,0], array_prof[:,1], c = y_pred)


# array_prof_cluster6=array_prof[mask_noise]
# 
#2EME CLUSTER
#------------------------------------------------------------------------------------------------------------------------------------
# clusters, y_pred=pcd_m.DBSCAN_pointcloud(np.column_stack((array_prof_cluster6[:,4], array_prof_cluster6[:,2]*5)), min_samples=4, n_neig=2)
# unique_labels = np.unique(y_pred)
# colors = plt.cm.get_cmap("tab10", len(unique_labels))
# point_colors = np.array([colors(label)[:3] for label in y_pred])
# pcd_m.view_point_cloud_from_array(np.array(array_prof_cluster6[:,5].tolist()), color=point_colors)



# mask_noise=y_pred==0
# array_prof_cluster6_1=array_prof_cluster6[mask_noise]

# pcd_m.view_point_cloud_from_array(np.array(array_prof_cluster6_1[:,5].tolist()))
# inc, v, wi, B, B_calc,s0,X,B=depthanything.calcul_transformation_cluster(array_prof_cluster6,6)
# ajuste_depth=depthanything.depthmap_ajustee
# depthanything.save_image_depthmap(ajuste_depth, "test")

# pointcloud_depthia=depthanything.depthmap_ia_to_o3d_pcd()
# pcd_m.view_point_cloud_from_array(np.asarray(pointcloud_depthia.points))


#%% Calcul grille de points pour calcul
depthanything.creer_grille_point()

#%% CALCUL POUR CHAQUE POINT DE LA DEPTHMAP
start_time = time.time()
print(start_time)
depthanything.calcul_dist_ajustee()
# depth_ajuste=depthanything.depthmap_ajustee
end_time = time.time()
print(end_time)
elapsed_time = end_time - start_time
print(f"Durée d'exécution du calcul ajusté de la Depthmap : {elapsed_time:.2f} secondes")
# value=depthanything.calcul_dist_ajust_from_uv(np.array([231,95]), 300, 400)
# depthmap_ia=depthanything.depthmap_IA

#%% Sauvegarde du résultat
depthanything.save_image_depthmap(depthanything.depthmap_ajustee, "Res_final")
depthanything.save_pointcloud_from_depthmap(depthanything.depthmap_ajustee, "Res_final.las")

#%%Variable des debugs 
#===================================================================================================================================

debug_depthmap=depthanything.debug
debug_camera=depthanything.camera.debug