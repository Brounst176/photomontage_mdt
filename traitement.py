
import lightglue_module as lg
import numpy as np

import plot_module as plot
import camera_module as cm

import os
import time
from scipy.spatial.transform import Rotation as Rot
import math as m

# %%


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
M=np.array([2528528.609711709, 1159666.3519212431, 512.8080390118347])






class projet_photomontage:
    def __init__(self, pathprojet, imagename, position_approchee):
        self.camera=cm.camera(pathprojet)
        self.homol=lg.homol_IA(imagename, position_approchee, self.camera, pathprojet)
        self.pathprojet=pathprojet
        self.imagename=imagename
        self.depthmap=cm.depthmap("_DSC6987_50.tif","_DSC6987_50.tif", imagename, self.camera , fact_reduce_depthajuste=0.5)
        self.camera.import_calib("nikonD7500", os.path.join(pathprojet, "calibration/nikon_calib.xml"))
        self.camera.import_image_from_omega_phi_kappa_file(os.path.join(pathprojet,"calibration/camera_orientation_calibname.txt"))
        



fichier_path = 'Input/calibration/nikon_simone.txt'


pathlas="C:/Users/Bruno/Documents/TM_script/Terrain/point_homologue.las"
pathlas="Input/point_dense_reduce.las"
pathlidar="Input/lidar.las"
pathimage="Input/image"
pathprojet="Input"
photoname="_DSC6987.JPG"
nikon=cm.camera("foldernotexite")
nikon.import_calib("nikonD7500", "Input/calibration/nikon_simone.xml")
nikon.import_image_from_omega_phi_kappa_file(fichier_path)
images=nikon.images


images_principales="_DSC7005.JPG"


start_time = time.time()
print(start_time)
homol=lg.homol_IA(images_principales, np.array([2528510.74, 1159641.0]), nikon, pathprojet)
proche=homol.trouver_cameras_proches_numpy(n=10)
liste_homol=homol.first_feats_matches_calc(2500)
end_time = time.time()
print(end_time)
elapsed_time = end_time - start_time
print(f"Durée d'exécution de calcul de Clusters : {elapsed_time:.2f} secondes")

# %%
# CALCUL DE LA SUITE DES IMAGES HOMOLOGUES


# dict_homol_filtre, array_homol_tri=homol.feats_analyse()
# i=20
# while array_homol_tri[array_homol_tri[:,4]>3].shape[0]<5:
#     liste_homol=homol.homol_array
#     image_M=liste_homol[np.argmax(liste_homol[:, 1]),0]
    
#     image_priorite = homol.projetcamera.liste_image_direction_proximite(image_M,np.array(homol.image_traitee))
#     for i in range(5):
#         homol.feats_calcul(homol.feats_cible, image_priorite[i,0],2500)
        
#     dict_homol_filtre, array_homol_tri=homol.feats_analyse()


dict_homol_filtre, array_homol_tri=homol.calcul_iteratif_homol_matches()
# %%
# plot.show_point_in_image( os.path.join(pathprojet,"image/_DSC7015.JPG"), array_homol_tri[:, [1,2]], array_homol_tri[:, 4])

dict_valide,dict_supprimer= homol.calcul_approximatif_homol_3D(dict_homol_filtre)
homol.show_matches_paires(dict_valide, "_DSC7002.JPG", images_principales)
# homol.show_matches_paires(dict_valide, "_DSC6962.JPG", "_DSC7014.JPG")
# homol.show_matches_paires(dict_supprimer, "_DSC6961.JPG")
# %%
homol.proj_points_3D_dict_to_image()
dict_valide=homol.dict_homol
homol.RANSAC_DLT(k=200, n_samples=15)
# R=homol.R
# print("données mesurée")
# print(R)

# Supposons que vous avez R bruitée
# U, _, Vt = np.linalg.svd(R)
# R_ortho = U @ Vt

# Vérifier et corriger si det = -1 (reflection)
# if np.linalg.det(R_ortho) < 0:
#     U[:, -1] *= -1
#     R_ortho = U @ Vt
# r =  Rot.from_matrix(R)
# angles = r.as_euler("xyz",degrees=True)
# print(angles)

print("données réel")
R=nikon.images[images_principales]["R"]
print(R)
r =  Rot.from_matrix(R)
angles = r.as_euler("xyz",degrees=True)

# # # r_retour = Rot.from_euler('xyz', angles, degrees=True)
# # # # print(r_retour.as_matrix())
print(angles)
# homol.DLT_image_cible()


#%%
if abs(homol.cx)>500 or abs(homol.cy)>500:
    P=homol.RANSAC_DLT_rot()
    P=homol.RANSAC_DLT_foc()
# P=homol.RANSAC_DLT_rot()
#%%
homol.distorsion_nulle()

Qxx, x, A, dx, B, dl, wi, vi = homol.calcul_moindre_carre_non_lineaire(dict_valide)


indices_trie = np.argsort(np.abs(wi)[:,0])


# Les 5 plus grandes => les 5 derniers (à l'envers)
indices_top5 = indices_trie[-5:][::-1]
uv=[]
uv_out=[]
uv_faux=[]
nb_obs=wi.shape[0]//2

# if max(abs(wi[indices_top5]))>3:
#     if min(abs(wi[indices_top5]))>3:
#         delta=
        

for i in range(wi.shape[0]//2):
    
    if abs(wi[i, 0])>3.0 or abs(wi[i+nb_obs, 0])>3.0:
            uv_out.append([B[i,0],B[i+nb_obs,0]])
            uv_faux.append(str(B[i,0])+"_"+str(B[i+nb_obs,0]))
    else:
            uv.append([B[i,0],B[i+nb_obs,0]])
        
plot.show_only_point_in_image("Input/image/"+images_principales, np.array(uv), np.array(uv_out))
# print("Indices des 5 plus grandes valeurs :", indices_top5)
print("Valeurs correspondantes :", wi[indices_top5])

# Qxx, x, A, dx, B, dl, wi, vi = homol.calcul_moindre_carre_non_lineaire(dict_valide, uv_faux)
# nb_obs=wi.shape[0]//2
# for i in range(wi.shape[0]//2):
    
#     if abs(wi[i, 0])>3.0 or abs(wi[i+nb_obs, 0])>3.0:
#             uv_out.append([B[i,0],B[i+nb_obs,0]])
#             uv_faux.append(str(B[i,0])+"_"+str(B[i+nb_obs,0]))
#     else:
#             uv.append([B[i,0],B[i+nb_obs,0]])
        
# plot.show_only_point_in_image("Input/image/"+images_principales, np.array(uv), np.array(uv_out))

liste_inc=["S", "angles", "f", "cx", "cy", "k1", "k2","p1"]

Qxx, x, A, dx, B, dl, wi, vi  =  homol.calcul_moindre_carre_non_lineaire(dict_valide, uv_faux, liste_inc)



#%%

# Mx=[]

# uv=[]

# d=len(dict_valide)//50


# i=0
# for key, val in dict_valide.items():
#     if i%d==0:
#         Mx.append(val["coord3D"])
#         Mi=val["coord3D"]

#         print(f"[{Mi[0]}, {Mi[1]}, {Mi[2]}],")
#         # print(f"[{val['u_proj']}, {val['v_proj']}],")
        
        
#         uv.append([val["u_proj"],val["v_proj"]])
#     i+=1

    
# # 

# plot.show_point_in_image( os.path.join(pathprojet,"image/_DSC7015.JPG"), array_homol_tri[:, [1,2]], array_homol_tri[:, 4])
# photomontage=projet_photomontage("Input", "_DSC7015.JPG", np.array([2528515.74, 1159629.21]))


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
# m=nikon.M_to_uv("_DSC7002.JPG", M)
# d=np.linalg.norm(M-nikon.images["_DSC6987"]["S"])
# vect_MS=M-nikon.images["_DSC6987"]["S"]
# M_calc=nikon.uv_to_M(photoname, m, d)

# H, d_proj=nikon.calcul_proj_cam("_DSC6987",M)

# print(f"Dist S-M : {d}")
# # pointcloud, rgb = pcd_m.readlas_to_numpy(pathlas)

# M_calc_d_projet=nikon.uv_to_M_by_dist_prof(photoname, m, d_proj)
# diff=M-M_calc_d_projet
# M_calc_d_projet_2=nikon.uv_to_M_by_dist_prof(photoname, [2016,2140], 10.868212)
# 

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
# nikon=cm.camera("NIKON D7500", "foldernotexite", 5568, 3712, -55.72194146069377, 23.384728774951231,4231.8597199545738,-0.14699248766634521,0.1616307088544226,-0.023431362387691224,0.0,-0.00074350800851607905,0.00052372914373731753,0.0,0.0)
# nikon.import_image_from_omega_phi_kappa_file(fichier_path)
# depthanything=cm.depthmap("_DSC6987_50.tif","_DSC6987_50.tif", photoname, nikon , fact_reduce_depthajuste=0.5)
# depthanything.import_projet_obj("Input/cabane/cabane_texture.obj")

# depthanything.importer_image_origine("C:/Users/Bruno/Documents/TM_script/Depth_IA/_DSC6987.JPG")


# depthanything.liste_coord_image_pt_homologue(pathlas, True, True,adapte_basique_prof=True)
# # depthanything.liste_coord_image_pt_homologue(pathlidar, True, True,adapte_basique_prof=True)
# depthanything.transformation_simple_depthmap_IA()
# depthanything.transformation_seconde_depthmap_IA()

# dict_prof=depthanything.dict_prof
# list_prof=depthanything.dict_prof_TO_liste(dict_prof)
# array_prof=np.array(list_prof)
# plot.plot_from_liste_prof(list_prof, title="Depth Anything V2", equal=True)

# # mask=((array_prof[:,4]>14) &(array_prof[:,4]<25))
# # array_epurer=array_prof[mask]
# # uv=array_epurer[:,[0,1]]
# # plot.show_point_in_image(depthanything.image, uv)


# start_time = time.time()
# print(start_time)
# depthanything.calcul_position_projet_sur_images()
# end_time = time.time()
# print(end_time)
# elapsed_time = end_time - start_time
# print(f"Durée d'exécution du calcul d'image : {elapsed_time:.2f} secondes")
# debug_depthmap=depthanything.debug
# debug_camera=depthanything.camera.debug

#%%CALCUL AVEC NATEL

# natel=cm.camera("NATEL", "foldernotexite")
# natel.import_image_from_omega_phi_kappa_file(fichier_path)
# natel.import_calib("Input/calibration/natel-bdc.xml")
# natel_deph=cm.depthmap("IMG_20250402_171632.tif","IMG_20250402_171632.tif", "IMG_20250402_171632", natel , fact_reduce_depthajuste=1.0, fact_reduce_IA=1.0)
# natel_deph.import_projet_obj("Input/villa.obj")

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
# # natel_deph.calcul_position_projet_sur_images()
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
