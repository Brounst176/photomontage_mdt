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
# M=np.array([
#      2528522.8052,
#      1159668.8328,
#      511.6924
#      ])
M=np.array([
     2528498.680,
     1159656.870,
     508.510
     ])



#



fichier_path = 'Input/calibration/camera_ORIENTAITON.txt'


pathlas="C:/Users/Bruno/Documents/TM_script/Terrain/point_homologue.las"
pathlas="Input/point_dense_reduce.las"
pathlidar="Input/lidar.las"
photoname="_DSC6987"
nikon=cm.camera("NIKON D7500", "foldernotexite", 5568, 3712, -55.72194146069377, 23.384728774951231,4231.8597199545738,-0.14699248766634521,0.1616307088544226,-0.023431362387691224,0.0,-0.00074350800851607905,0.00052372914373731753,0.0,0.0)
nikon.import_image_from_omega_phi_kappa_file(fichier_path)
# depthanything=cm.depthmap("_DSC6987_50_depthpro.tif","_DSC6987_50.tif", photoname, nikon , fact_reduce_depthajuste=0.5)
# depthanything=cm.depthmap("_DSC6987_50.tif","_DSC6987_50.tif", photoname, nikon , fact_reduce_depthajuste=0.5)


#%%Import d'un projet
# depthanything.import_projet_obj("Input/cabane/cabane_texture.obj")
# 
# depthanything.importer_image_origine("C:/Users/Bruno/Documents/TM_script/Depth_IA/_DSC6987.JPG")


# depthanything.liste_coord_image_pt_homologue(pathlas, True, True,adapte_basique_prof=True)

# depthanything.transformation_simple_depthmap_IA()
# dict_prof=depthanything.dict_prof
# list_prof=depthanything.dict_prof_TO_liste(dict_prof)
# array_prof=np.array(list_prof)
# plot.plot_from_liste_prof(list_prof, title="Depth Anything V2")
# depthanything.transformation_seconde_depthmap_IA()
# # depthanything.liste_coord_image_pt_homologue(pathlidar, False)



# dict_prof=depthanything.dict_prof
# list_prof=depthanything.dict_prof_TO_liste(dict_prof)
# array_prof=np.array(list_prof)
# plot.plot_from_liste_prof(list_prof, title="Depth Anything V2", equal=True)

#%% Calcul pour contrôle fonction de photogrammétrie
# m=nikon.M_to_uv("_DSC6987", M)
# d=np.linalg.norm(M-nikon.images["_DSC6987"]["S"])
# vect_MS=M-nikon.images["_DSC6987"]["S"]
# M_calc=nikon.uv_to_M(photoname, m, d)

# H, d_proj=nikon.calcul_proj_cam("_DSC6987",M)

# print(f"Dist S-M : {d}")
# pointcloud, rgb = pcd_m.readlas_to_numpy(pathlas)

# M_calc_d_projet=nikon.uv_to_M_by_dist_prof(photoname, m, d_proj)
# diff=M-M_calc_d_projet
# M_calc_d_projet_2=nikon.uv_to_M_by_dist_prof(photoname, [2016,2140], 10.868212)


# arr=depthanything.return_array_epurer_from(1050,1000)



#%%CALCUL PAR CLUSTER DE ZONE 
#===================================================================================================================================
#===================================================================================================================================

# start_time = time.time()
# print(start_time)
# cluster_liste=depthanything.optimisation_des_clusters()


# end_time = time.time()
# print(end_time)
# elapsed_time = end_time - start_time
# print(f"Durée d'exécution de calcul de Clusters : {elapsed_time:.2f} secondes")


#%% Calcul grille de points pour calcul
# depthanything.creer_grille_point()

#%% CALCUL AJUSTE AVEC LES POINTS de
# start_time = time.time()
# print(start_time)
# res_ok=depthanything.calcul_dist_ajustee(calcul_pt_homologue=True)
# end_time = time.time()
# print(end_time)
# elapsed_time = end_time - start_time
# print(f"Durée d'exécution du calcul ajusté de la Depthmap : {elapsed_time:.2f} secondes")

#%% CALCUL COMPLET
# start_time = time.time()
# print(start_time)
# res_ok=depthanything.calcul_dist_ajustee(calcul_pt_homologue=False)
# depth_ajuste=depthanything.depthmap_ajustee
# end_time = time.time()
# print(end_time)
# elapsed_time = end_time - start_time
# print(f"Durée d'exécution du calcul ajusté de la Depthmap : {elapsed_time:.2f} secondes")


#%% Sauvegarde du résultat
# depthanything.save_image_depthmap(depthanything.depthmap_ajustee, "Res_final")
# points_depthajustee=depthanything.save_pointcloud_from_depthmap(depthanything.depthmap_ajustee, "Res_final.las")
# pcd_m.view_point_cloud_from_array(np.array(points_depthajustee))


# %%CALCUL FINAL
# depthanything.calcul_prof_ajustee_from_homol(2100,1060)
# depthanything.calcul_prof_ajustee_from_homol(1956,644)

# start_time = time.time()
# print(start_time)
# depthanything.calcul_position_projet_sur_images()
# end_time = time.time()
# print(end_time)
# elapsed_time = end_time - start_time
# print(f"Durée d'exécution du calcul d'image : {elapsed_time:.2f} secondes")
#%%Variable des debugs 
#===================================================================================================================================

# debug_depthmap=depthanything.debug
# debug_camera=depthanything.camera.debug

#%%CALCUL AVEC PHOTO CABANE
nikon=cm.camera("NIKON D7500", "foldernotexite", 5568, 3712, -55.72194146069377, 23.384728774951231,4231.8597199545738,-0.14699248766634521,0.1616307088544226,-0.023431362387691224,0.0,-0.00074350800851607905,0.00052372914373731753,0.0,0.0)
nikon.import_image_from_omega_phi_kappa_file(fichier_path)
depthanything=cm.depthmap("_DSC6987_50.tif","_DSC6987_50.tif", photoname, nikon , fact_reduce_depthajuste=0.5)
depthanything.import_projet_obj("Input/cabane/cabane_texture.obj")

depthanything.importer_image_origine("C:/Users/Bruno/Documents/TM_script/Depth_IA/_DSC6987.JPG")


depthanything.liste_coord_image_pt_homologue(pathlas, True, True,adapte_basique_prof=True)

depthanything.transformation_simple_depthmap_IA()
depthanything.transformation_seconde_depthmap_IA()
# dict_prof=depthanything.dict_prof
# list_prof=depthanything.dict_prof_TO_liste(dict_prof)
# array_prof=np.array(list_prof)
# plot.plot_from_liste_prof(list_prof, title="Depth Anything V2", equal=True)
start_time = time.time()
print(start_time)
depthanything.calcul_position_projet_sur_images()
end_time = time.time()
print(end_time)
elapsed_time = end_time - start_time
print(f"Durée d'exécution du calcul d'image : {elapsed_time:.2f} secondes")

#%%CALCUL AVEC NATEL

# natel=cm.camera("NATEL", "foldernotexite")
# natel.import_image_from_omega_phi_kappa_file(fichier_path)
# natel.import_calib("Input/calibration/natel-bdc.xml")
# natel_deph=cm.depthmap("IMG_20250402_171632_50.tif","IMG_20250402_171632_50.tif", "IMG_20250402_171632", natel , fact_reduce_depthajuste=0.5, fact_reduce_IA=0.5)
# natel_deph.import_projet_obj("Input/immeuble.obj")

# natel_deph.importer_image_origine("C:/Users/Bruno/Documents/TM_script/Depth_IA/IMG_20250402_171632.jpg")


# natel_deph.liste_coord_image_pt_homologue(pathlas, True, True, adapte_basique_prof=True)

# natel_deph.transformation_simple_depthmap_IA()

# natel_deph.transformation_seconde_depthmap_IA()

# # dict_prof=natel_deph.dict_prof
# # list_prof=natel_deph.dict_prof_TO_liste(dict_prof)
# # array_prof=np.array(list_prof)
# # plot.plot_from_liste_prof(list_prof, title="Depth Anything V2")


# # # mask=((array_prof[:,4]>38) &(array_prof[:,4]<45))
# # # array_epurer=array_prof[mask]
# # # uv=array_epurer[:,[0,1]]
# # # plot.show_point_in_image("C:/Users/Bruno/Documents/TM_script/Depth_IA/IMG_20250402_171632.jpg", uv)

# start_time = time.time()
# print(start_time)
# natel_deph.calcul_position_projet_sur_images()
# end_time = time.time()
# print(end_time)
# elapsed_time = end_time - start_time
# print(f"Durée d'exécution du calcul d'image : {elapsed_time:.2f} secondes")
# #POUBELLE
## ==================================================================================================================================
#%%CALCUL PAR SEPRATION DE GAUCHE A DROITE
# plot.plot_from_liste_prof(list_prof)

# depthanything.transformation_simple_depthmap_IA()

# plot.plot_from_liste_prof(list_prof)

# depthanything.initialisation_calc_par_groupe_colonne_pour_moindre_carre()
