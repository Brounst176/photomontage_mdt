# -*- coding: utf-8 -*-
import module_python.ori_photo_cacl_module as lg
import numpy as np
import module_python.photomontage_module as ph
import module_python.plot_module as plot
import module_python.camera_module as cm

import os
import time
from scipy.spatial.transform import Rotation as Rot
import math as m




#IMPORTER LA CREATION D'UNE CARTE DE PROFONDEUR
start_time = time.time()
print(start_time)

#%%Variable de base

#Projet à HEIG
pathimage="data_projet/heig/image"
pathprojet="data_projet/heig"
pathnuage="data_projet/heig/pointcloud.las"
images_principales="20250705101835.jpg"

position_approchee = np.array([2540581, 1181281])

#Projet à BREMBLENS
pathimage="data_projet/Bremblens/image"
pathprojet="data_projet/Bremblens"
pathnuage="data_projet/Bremblens/pointcloud.las"
images_principales="DJI_0608.JPG"

position_approchee = np.array([616, 136])




#%% Importation des données de base et préparation des class de base
#===================================================================================

modele_photogra=cm.camera("foldernotexite")
# modele_photogra.import_calib("nikon_d7500_17mm", pathprojet+"/nikon_d7500_17mm.xml") #POUR HEIG
modele_photogra.import_calib("DJIMINI", pathprojet+"/DJIMINI.xml")  #POUR BREMBLENS
modele_photogra.import_image_from_omega_phi_kappa_file(pathprojet+"/position_orientation.txt")

cacl_photo=lg.calcul_orientation_photo_homol(images_principales, position_approchee, modele_photogra, pathprojet)

#%% CALCUL DE LA POSITION DE L'IMAGE UTILISATEUR
#===================================================================================

#RECHERCHE DES POINTS HOMOLOGUES
cacl_photo.homol_array=None
cacl_photo.image_traitee=[]
liste_homol=cacl_photo.first_feats_matches_calc()

dict_homol_filtre, array_homol_tri=cacl_photo.calcul_iteratif_homol_matches()
#%%
#CALCUL DES COORDONNEES 3D DES POINTS HOMOLOGUES
dict_valide,dict_supprimer= cacl_photo.calcul_approximatif_homol_3D(dict_homol_filtre)

#%% 
# cacl_photo.proj_points_3D_dict_to_image()
#CALCUL POSITION ORIENTATION APPROCHEE PAR DLT
cacl_photo.RANSAC_DLT(k=200, n_samples=6, nb_inliers=8)



#%%
#PREMIERE DETERMINATION SIMPLE (f, S, orientation, k1) PAR MOINDRE CARRE NON LINEAIRE
cacl_photo.distorsion_nulle()
Qxx, x, A, B, wi, vi = cacl_photo.calcul_moindre_carre_non_lineaire(cacl_photo.dict_homol)

#RECHERCHE DES POINTS FAUX PAR WI
key_pt_homol_faux = cacl_photo.calcul_dictkey_point_faux_selon_wi(B, wi)

#DEUXIEME DETERMINATION ("S", "angles", "f", "cx", "cy", "k1", "k2"1) PAR MOINDRE CARRE NON LINEAIRE
cacl_photo.distorsion_nulle()
liste_inc=["S", "angles", "f", "cx", "cy", "k1", "k2"]

Qxx, x, A, B, wi, vi  =  cacl_photo.calcul_moindre_carre_non_lineaire(cacl_photo.dict_homol, key_pt_homol_faux, liste_inc)



#%%
#CREATION DU MODEL CAMERA DU RESULAT
image_cible=cm.camera("foldernot")
image_cible.import_from_class_calc_photo_homol(cacl_photo)


#%% CALCUL PHOTOMONTAGE
##===================================================================================

photomontage=ph.photomontage_depthmap(images_principales, pathprojet, image_cible, fact_reduce_IA=0.5, fact_reduce_photomontage=0.1)
# photomontage=ph.photomontage_depthmap(images_principales, pathprojet, modele_photogra, fact_reduce_IA=0.5, fact_reduce_photomontage=0.5)

photomontage.import_projet_obj(os.path.join(pathprojet, "projet.obj"))
photomontage.import_projet_emprise_obj(os.path.join(pathprojet, "emprise.obj"))
# 
photomontage.importer_image_origine(os.path.join(pathimage, images_principales))
photomontage.create_depthmap(os.path.join(pathimage, images_principales))


#%%
## importation du nuage de points et inversion de la dethmap
photomontage.depthmap_IA=photomontage.depthmap_IA_backup
photomontage.dict_prof={}
photomontage.liste_coord_image_pt_homologue(pathnuage, True, True,adapte_basique_prof=True)


# dict_prof=photomontage.dict_prof
# liste_prof=photomontage.dict_prof_TO_liste(photomontage.dict_prof)
# array_prof=np.array(liste_prof)


# plot.show_only_point_in_image(os.path.join(pathprojet, "image", images_principales), array_prof[:,[0,1]]*2)

#%%

photomontage.transformation_simple_depthmap_IA()
photomontage.transformation_seconde_depthmap_IA()


#%%
# photomontage.plot_from_prof()

#%%
image_PIL, image_projet=photomontage.calcul_position_projet_sur_images()

#%%
end_time = time.time()
print(end_time)
elapsed_time = end_time - start_time
print(f"Durée d'exécution du calcul ajusté de la Depthmap : {elapsed_time:.2f} secondes")
