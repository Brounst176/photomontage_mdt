import numpy as np

import plot_module as pltbdc

import camera_module as cm

nikon=cm.camera("NIKON D7500", "foldernotexite", 5568, 3712, -55.72194146069377, 23.384728774951231,4231.8597199545738,-0.14699248766634521,0.1616307088544226,-0.023431362387691224,0.0,-0.00074350800851607905,0.00052372914373731753,0.0,0.0)


depthmap_DSC6987=cm.depthmap("C:/Users/Bruno/Documents/TM_script/Terrain/_DSC6987_556-371.tif", "_DSC6987", nikon)


fichier_path = 'C:/Users/Bruno/Documents/TM_script/Terrain/camera_ORIENTAITON.txt'
nikon.import_image_from_omega_phi_kappa_file(fichier_path)


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
#Fenetre bas
# M=np.array([
#      2528603.686,
#      1159679.230,
#      510.20
#      ])
#Faite simone
M=np.array([
     2528524.957,
     1159666.411,
     518.316
     ])
#fenetre simone
M=np.array([
     2528522.8052,
     1159668.8328,
     511.6924
     ])

m=nikon.M_to_uv("_DSC6987", M)
d=np.linalg.norm(M-nikon.images["_DSC6987"]["S"])
vect_MS=M-nikon.images["_DSC6987"]["S"]
print(f"Dist S-M : {d}")


#Prends un nuage de points
list_uv_prof, img_prof=nikon.liste_coord_image_pt_homologue("C:/Users/Bruno/Documents/TM_script/Terrain/point_homologue.las", "_DSC6987",0.1)
depthmap=depthmap_DSC6987.depthmap_IA



A,B, Qll, inc, wi, v, X,B_calc,s0=nikon.tranformation_depthanything_gauss(depthmap, list_uv_prof, img_prof)
pltbdc.plot_mesure_calcule(X, B, B_calc)

list_uv_ajustee, liste_uv_supprimer=nikon.ajuste_liste_observation_sur_UV_depthanything(depthmap, list_uv_prof)

A,B, Qll, inc, wi, v, X,B_calc, s0=nikon.tranformation_depthanything_gauss(depthmap, list_uv_ajustee, img_prof)


X_epurer=[]
Y_epurer=[]
for i in range(Qll.shape[1]):
    if Qll[i,i]>5:
        X_epurer.append(X[i,0])
        Y_epurer.append(B[i,0])
        
        
pltbdc.plot_mesure_calcule(X, B, B_calc, X_epurer, Y_epurer)

# pltbdc.input_10_wi_to_image("_DSC7050_556-371.jpg",list_uv_prof,B,valeur, wi)
# nikon.suppression_point_isole("point_homologue.las")
# nikon.remove_statistical_outlier_pcd("point_homologue.las")
