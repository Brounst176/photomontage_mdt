
import module_python.fonction_optimisee_numba as jit_fct
import time
import os


import trimesh
from PIL import Image 
import copy 
import laspy
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Ellipse
import math
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import alphashape 
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point
from scipy.stats import chi2, norm
from numba import njit, jit #OPTIMISATION DES CALCULS
import cv2
import torch
from module_python.depth_anything_v2.dpt import DepthAnythingV2
import matplotlib
import module_python.plot_module as plot
import module_python.mesh_module as mm
import module_python.pointcloud_module as pcd

# DEPTHMAP CLASS
# ===================================================================================================

class photomontage_depthmap:
    def __init__(self, photoname, pathprojet, camera,  max_megapixel=1.3, fact_reduce_IA=0.5, fact_reduce_photomontage=0.1, depthmodel="depthanything"):
        """
        

        Parameters
        ----------
        pathdepthmap_IA : path
            chemin où se situe la depthmap IA.
        photoname : string
            nom de la photo.
        pathlas : path string
            chemin du nuage de points.
        calc_normal : bool
            Les normales du nuage de points doivent être recalculée.
        camera : class camera
            indiquer la class camera.
        max_megapixel : float, optional
            nombre maximum de pixel sur les images. The default is 1.5.
        fact_reduce_IA : float, optional
            Facteur de reduction de la carte de profondeur par rapport à l'imaage d'origine. The default is 0.1.
        fact_reduce_photomontage : float, optional
            Facteur de reduction de l'image d'origine pour importer le projet. The default is 0.1.
        """
        self.camera=camera
        self.depthmap_IA=None
        self.depthmap_IA_backup=None
        self.depth_modele=depthmodel
        self.photoname=photoname
        self.pathprojet=pathprojet
        self.image=None
        self.fact_reduce_IA=fact_reduce_IA
        self.fact_reduce_photomontage=fact_reduce_photomontage
        self.depthmap_ajustee=None
        self.depthmap_clusters=None
        self.dict_prof={}
        self.liste_groupe_resultat=[]
        self.clusters=[]
        self.param_transfo_cluster={}
        self.grille_calcul=[]
        self.debug=""
        self.boundaries_proj=None
        self.projet_obj=None
        self.projet_emprise=None
        self.dproj_min_proj=None
        self.dproj_max_proj=None

        
    def create_depthmap(self, pathimage):
        img=self.read_image(pathimage)
        img_resize=self.resize_img_from_scale(img, self.fact_reduce_IA)
        depth_array=self.image_to_depthmap_IA_array(img_resize)
        self.export_DEM_from_deptharray(depth_array)
        self.export_image_from_deptharray(depth_array, img, grayscale=True)
        
        self.depthmap_IA=depth_array
        self.depthmap_IA_backup=depth_array
        
        # self.depthmap_ajustee=self.initialisaiton_depthmap()
        # self.depthmap_clusters=self.initialisaiton_depthmap()
        
    def read_image(self, path_image):
        raw_img = cv2.imread(path_image)
        return raw_img


    def image_to_depthmap_IA_array(self,img_cv2):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
        
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        model = model.to(DEVICE).eval()
        

        depth = model.infer_image(img_cv2)
        
        return depth

    def export_DEM_from_deptharray(self,depth_array):
        #EXPORT IMAGE SOUS FORME DE MNT
        im = Image.fromarray(depth_array)
        im.save(self.pathprojet+"/Output/"+"image_depth.tif")
        
    def export_image_from_deptharray(self,depth_array, image_origine, pred_only=False, grayscale=False):
        depth_255 = (depth_array - depth_array.min()) / ( depth_array.max() - depth_array.min()) * 255.0
        depth_255 = depth_255.astype(np.uint8)
        
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        
        if grayscale:
            depth_rgb = np.repeat(depth_255[..., np.newaxis], 3, axis=-1)
        else:
            depth_rgb = (cmap(depth_255)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
        resized_img=self.resize_img_from_pixelwidth(depth_rgb, 1000)
        cv2.imwrite('Ouput/img.png', depth_rgb)
        if pred_only:
            cv2.imwrite(self.pathprojet+"/Output/"+"image_depth.png", resized_img)
            # cv2.imshow('Image DEM', resized_img)
            # cv2.imshow("Image", img_array)
            cv2.waitKey(0)  # Attend une touche
            cv2.destroyAllWindows()  # Ferme toutes les fenêtres

        else:
            image_origine_resize=self.resize_img_from_pixelwidth(image_origine, 1000)
            split_region = np.ones((image_origine_resize.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([image_origine_resize, split_region, resized_img])
            # cv2.imshow('combined result DEM', combined_result)
            # cv2.waitKey(0)  # Attend une touche
            # cv2.destroyAllWindows()  # Ferme toutes les fenêtres
            cv2.imwrite(os.path.join(self.pathprojet,"Output","image_depth.png"), combined_result)
        return depth_rgb

    def resize_img_from_pixelwidth(self,img, pixelwidth):
        scale_factor = pixelwidth/img.shape[1]
        new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return resized_img
        
    def resize_img_from_scale(self,img, scale_factor):
        new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return resized_img
    
    
    def import_projet_obj(self, pathobj):
        mesh = trimesh.load_mesh(pathobj)

        # Extraire les vertices
        vertices = mesh.vertices
        self.projet_obj=mesh
        liste_uv=[]
        dprojmin=9999
        dprojmax=0
        for point in vertices:
            liste_uv.append(self.camera.M_to_uv(self.photoname, point))
            H, dproj=self.camera.calcul_proj_cam(self.photoname, point)
            if dproj<dprojmin:
                dprojmin=dproj
            if dproj>dprojmax:
                dprojmax=dproj
        array_uv=np.array(liste_uv)
        min_u=np.min(array_uv[:,0])
        max_u=np.max(array_uv[:,0])
        min_v=np.min(array_uv[:,1])
        max_v=np.max(array_uv[:,1])
        if max_u>self.camera.w:
            max_u=self.camera.w
        if max_v>self.camera.h:
            max_v=self.camera.h
        if min_u<0:
            min_u=0
        if min_v<0:
            min_v=0
            
        
        self.boundaries_proj=np.array([
            [min_u, max_u],
            [min_v, max_v]
            ])
        self.dproj_min_proj=dprojmin
        self.dproj_max_proj=dprojmax
        print(self.boundaries_proj)
        return array_uv
    def import_projet_emprise_obj(self, pathobj):
        mesh = trimesh.load_mesh(pathobj)

        self.projet_emprise=mesh
        
    def importer_image_origine(self, pathimage):
        img = Image.open(pathimage)
        
        # Obtenir les dimensions de l'image originale
        width, height = img.size
        fact=self.fact_reduce_photomontage
        # Calculer les nouvelles dimensions
        new_width = int(width * fact)
        new_height = int(height * fact)
        resized_image = img.resize((new_width, new_height))

        self.image=resized_image
        
        
    def liste_coord_image_pt_homologue(self, pathlas, calc_normal=False, inv=False,adapte_basique_prof=False):

        fact_reduce=self.fact_reduce_IA
        R=self.camera.images[self.photoname]["R"]
        vect_normal_photo=self.camera.vect_dir_camera(self.photoname)
        verticale=np.array([0,0,1])
        
        
        
        point_cloud, normales=pcd.readlas_to_numpy(pathlas)
        
        
        o3d_pcd=pcd.array_TO_o3d(point_cloud, normales)
        # o3d_pcd=pcd.remove_statistical_outlier_pcd(o3d_pcd)
        point_cloud, normales=pcd.o3d_TO_array(o3d_pcd)
        mesh=self.camera.mesh_pyramide_camera_create(self.photoname, 50, self.boundaries_proj)
        
        points_inside, normal_inside=pcd.points_inside_mesh(point_cloud, normales, mesh)
        

        # pcd.view_point_cloud_from_array(points_inside, normal_inside)
        if calc_normal:
            normal_inside=pcd.calcul_normal_point_cloud(points_inside, normal_inside)
        # points_inside_reduce=pcd.reduction_nuage_nbre(points_inside)
            normal_inside=np.array([normal_inside[i]/(np.linalg.norm(normal_inside[i])*2) for i in range(normal_inside.shape[0])])
            o3d_pcd=pcd.array_TO_o3d(points_inside, normal_inside)
            
        else:
            o3d_pcd=pcd.array_TO_o3d(points_inside, normal_inside)
        # pcd.view_point_cloud_from_array(points_inside, normal_calc)
        
        
        voxel=3
        #VOXEL DU NUAGE DE POINTS POUR REDUCTION
        for i in range(10):
            points_inside_reduce=pcd.reduction_nuage_voxel(o3d_pcd,voxel)
            # print(points_inside_reduce.points)
            points=np.asarray(points_inside_reduce.points)
            if 2000>points.shape[0]>1000:
                break
            elif points.shape[0]<1000:
                voxel-=0.3
            else:
                voxel+=0.3
        #VOXEL d'aborb supprimer
        point_cloud, normales=pcd.o3d_TO_array(points_inside_reduce)
        
        
        
        
        print(f"Nbre de points après réduction: {point_cloud.shape[0]}")
        # pcd.save_point_cloud_las(point_cloud, "point_camera")
        
        point_cloud_epurer=[]
        point_supprimer=[]
        
        normale_epurer=[]
        normale_supprimer=[]
        
        
        #Nettoyage du nuage de points en fonction des normales des points
        if calc_normal:
            for i in range(len(point_cloud)):
                point=point_cloud[i]
                normale=normales[i]
                produit_scalaire = np.dot(normale, vect_normal_photo)
                produit_scalaire_verticale=np.dot(normale, verticale)
                # print(produit_scalaire)
                if produit_scalaire<0.1 or produit_scalaire_verticale>0.5:
                    
    
                    point_cloud_epurer.append(point)
                    normale_epurer.append(normale)
                else:
    
                    point_supprimer.append(point)
                    normale_supprimer.append(normale)
        
        
        
            #ANALYSE LES POINTS DE NORMALES FAUSSES (SI MOINS DE 10 POINTS DANS UN CLUSTER RAJOUTER AU point juste)
            clusters, y_pred= pcd.DBSCAN_pointcloud(np.array(point_supprimer))
            mask_noise = y_pred == -1
    
            point_cloud_epurer=point_cloud_epurer+np.array(point_supprimer)[mask_noise].tolist()
            normale_epurer=normale_epurer+np.array(normale_supprimer)[mask_noise].tolist()
            point_supprimer = np.array(point_supprimer)[~mask_noise].tolist()
        
        liste_uv_profondeur=[]
        dict_uv_profondeur=self.dict_prof
        if len(point_cloud_epurer)==0:
            point_cloud_epurer=point_cloud.tolist()
            normale_epurer=normal_inside.tolist()
            
            
        depth_array=np.array(self.depthmap_IA)
        depthback=np.array(self.depthmap_IA_backup)
        
        
        #CREATION DE LA DICT DES UV ET DES POINTS
        img_uv_prof=np.zeros((int(self.camera.h*fact_reduce), int(self.camera.w*fact_reduce)))
        for i in range(len(point_cloud_epurer)):
            point=point_cloud_epurer[i]
            normale=normale_epurer[i]

            d=self.camera.dist_MS(point, self.photoname)
            P_proj, d_proj=self.camera.calcul_proj_cam(self.photoname, point)
            uv=self.camera.M_to_uv(self.photoname, point)
            u=math.floor(uv[0]*fact_reduce)
            v=math.floor(uv[1]*fact_reduce)

            if u<int(self.camera.w*fact_reduce) and v<int(self.camera.h*fact_reduce):
                depthIA=depth_array[v,u]
                depthIAback=depthback[v,u]
                if depthIAback!=0:
                    if u in dict_uv_profondeur.keys():
                        if v in dict_uv_profondeur[u].keys():
                            if d_proj<dict_uv_profondeur[u][v]["d"]:
                                dict_uv_profondeur[u][v]["d"]=d_proj
                        else:
                            dict_uv_profondeur[u].update({v:{"d":float(d_proj), "sigma":0.3, "depthIA":float(depthIA), "point":point, "normal":normale}})
                    else:
                        dict_uv_profondeur[u]={v:{"d":float(d_proj), "sigma":0.3, "depthIA":float(depthIA), "point":point, "normal":normale}}
                   
                    
                    prof_uv=np.array([u, v, d, 0.1,depthIA])
                    liste_uv_profondeur.append(prof_uv)
                    img_uv_prof[v, u]=d
            



    
        # pcd.save_point_cloud_las(np.array(point_cloud_epurer), "point_cloud_import_to_camera.las")
        # pcd.save_point_cloud_las(np.array(point_supprimer), "point_cloud_supprimer_to_camera.las")
        
        # pcd.view_point_cloud_from_array(np.array(point_cloud_epurer))
        # pcd.view_point_cloud_from_array(np.array(point_supprimer))
        
        
        # Trier le dictionnaire par clé dans l'ordre croissant
        dict_uv_profondeur = dict(sorted(dict_uv_profondeur.items()))
        if adapte_basique_prof:
            liste_prof=self.dict_prof_TO_liste(dict_uv_profondeur)
            array_prof=np.array(liste_prof)
            plot.plot_from_liste_prof(liste_prof, "depthanything1")
            
            
            #Echelle depthajuste
            max_depth=np.max(array_prof[:,4])
            min_depth=np.min(array_prof[:,4])
            max_dproj=np.max(array_prof[:,2])
            min_dproj=np.min(array_prof[:,2])
            a=(max_dproj-min_dproj)/(max_depth-min_depth)
            b=max_dproj-max_depth*a

            
            self.depthmap_IA=self.depthmap_IA*a+b
            
            array_prof[:,4]= array_prof[:,4]*a+b
    
            max_depth=np.max(array_prof[:,4])
            
    
            
            if inv:
                self.depthmap_IA=self.depthmap_IA*-1+max_depth
                array_prof[:,4]= array_prof[:,4]*-1+max_depth
                self.depthmap_IA = np.where((self.depthmap_IA == b*-1+max_depth), self.depthmap_IA-self.depthmap_IA+9999, self.depthmap_IA)
            
    
            
            cluster, y_pred=pcd.DBSCAN_pointcloud(array_prof[:, [4,2]],2,15)
            unique=np.unique(y_pred)
            plt.scatter(array_prof[:, 4], array_prof[:, 2], c=y_pred, cmap='coolwarm')
            plt.show()
            plt.close()
            unique=np.delete(unique, 0)
            liste_cluster=[]
            
            if len(unique)>0:
            
                for i in range(len(unique)):
                    mask=y_pred==unique[i]
                    array_pred=array_prof[mask]
                    mean_depth=np.mean(array_pred[:,4])
                    mean_dproj=np.mean(array_pred[:,2])
                    min_dproj=np.min(array_pred[:,2])
                    liste_cluster.append([mean_depth,mean_dproj, unique[i], min_dproj])
                pred_groupe=np.array(liste_cluster)
        
        
                mask=(
                    (pred_groupe[:,1]<=2*pred_groupe[:,0])
                    & 
                    (pred_groupe[:,1]>=0.3*pred_groupe[:,0])
                    )
        
                grpe_epurer=pred_groupe[mask]
                grpe_epurer = grpe_epurer[grpe_epurer[:, 3].argsort()]
                # print(grpe_epurer)
                mask=(grpe_epurer[:,0]>0)
        
                for i in range(grpe_epurer.shape[0]):
                    id_cluster=grpe_epurer[i,2]
                    mask_id=y_pred==id_cluster
                    array_cluster= array_prof[mask_id]
                    indices = np.where((grpe_epurer[:,1] >= np.min(array_cluster[:,2])) & (grpe_epurer[:,1] <= np.max(array_cluster[:,2])) & (grpe_epurer[:,0] < grpe_epurer[i,0]))
        
            
                    for i, indice in enumerate(indices[0].tolist()):
                        if grpe_epurer[indice,0]< np.min(array_cluster[:,0])-max_depth/4:
                            mask[indice]=False
        
                        
                        
                grpe_epurer=grpe_epurer[mask]
                    
                    
                    
                        
                    
                    
                        
                
                
                y_pred_arr = np.array(y_pred)
                mask_array_prof=np.isin(y_pred_arr, grpe_epurer[:, 2])
                array_prof=array_prof[mask_array_prof]

            
            plt.scatter(array_prof[:, 4], array_prof[:, 2], c="green")
            plt.title("Données épurées")
            plt.show()
            plt.close()
            
    
                    
                # print("Distances :", distances)
                # print("Indices :", indices)
                # print("Voisins :", pred_groupe[indices[0]])
            
                        #FAIRE iCI LE TRI DU NUAGE DE POINTS PROJETE
                        #=================================================================================================
                        
                
            plt.show()
            plt.close()
            # mask=return_mask_detection_donnee_aberrante(array_prof[:, [4,2]])
            # liste_epurer=array_prof[mask].tolist()
            liste_epurer=array_prof.tolist()
            dict_uv_profondeur=self.liste_TO_dict_prof(liste_epurer)
        
        
        
        self.dict_prof=dict_uv_profondeur
        # return dict_uv_profondeur
        
        
    def calcul_precision_dethmap(self, array_prof):
        diff=array_prof[:,4]-array_prof[:,2]
        return np.std(np.abs(diff))*2.5
        
    def calcul_position_projet_sur_images(self):
        
        
        
        depthmap_IA=np.array(self.depthmap_IA, dtype=np.float64)
        larg_image=depthmap_IA.shape[0]
        
        dprojmin=self.dproj_min_proj

        depthmap_IA_backup=self.depthmap_IA_backup
        fact_reduce_photomontage=self.fact_reduce_photomontage
        boundaries=self.boundaries_images_reduites()
        # photoname=self.photoname
        fact=self.fact_reduce_photomontage/self.fact_reduce_IA
        print(fact)
        image=self.image
        S=self.camera.images[self.photoname]["S"]
        R=self.camera.images[self.photoname]["R"]
        camera=self.camera
        camera.set_calib_from_image(self.photoname)
        print(camera.k1)
        dict_prof=self.dict_prof
        liste_prof=self.dict_prof_TO_liste(dict_prof)
        array_prof=np.array(liste_prof)[:,[0,1,2,3,4]]
        array_prof=np.array(array_prof, dtype=np.float64)
        
        prec_depth=self.calcul_precision_dethmap(array_prof)

        # new_color = (255, 0, 0)  # Rouge
        projet=self.projet_obj
        #paramètre de couleur

        light_dir = np.array([1, 1, 1])
        light_dir= self.camera.vect_dir_camera(self.photoname)# Direction de la lumière
        light_dir[2]=light_dir[2]+50
        light_dir[0]=light_dir[0]+50
        light_dir[1]=light_dir[1]+10
        view_dir=self.camera.vect_dir_camera(self.photoname)  # Direction de l'observateur (par exemple, vers l'avant)

        nbre_de_calcul=(-boundaries[1,0]+boundaries[1,1])*(-boundaries[0,0]+boundaries[0,1])
        print(f"Nbre de pixel: {nbre_de_calcul}")
        w_image_projet= int(camera.w*fact_reduce_photomontage)
        h_image_projet=int(camera.h*fact_reduce_photomontage)
        image_rgba_projet = np.zeros((h_image_projet,w_image_projet, 4), dtype=np.uint8)
        image_dproj_projet=np.zeros((h_image_projet,w_image_projet, 1), dtype=np.float32)
        
        
        MAX_PIXELS = 100

        # Taille de l'emprise totale
        u_min, u_max = boundaries[0]
        v_min, v_max = boundaries[1]
        
        width = u_max - u_min
        height = v_max - v_min
        total_pixels = width * height
        
        # Taille idéale de bloc (approximativement carré)
        approx_block_size = int(np.sqrt(MAX_PIXELS))
        block_w = approx_block_size
        block_h = approx_block_size
        sub_boundaries = []
        for v_start in range(v_min, v_max, block_h):
            for u_start in range(u_min, u_max, block_w):
                u_end = min(u_start + block_w, u_max)
                v_end = min(v_start + block_h, v_max)
        
                nb_pixels = (u_end - u_start) * (v_end - v_start)
                if nb_pixels == 0:
                    continue
                sub_boundaries.append(np.array([[u_start, u_end], [v_start, v_end]], dtype=np.int64))
        pixel_total=0
        for sub_boundary in tqdm(sub_boundaries, desc="Calcul d'intersections"):

                pixel_coords, directions_list, closest_rays= jit_fct.init_calcul_intersection(sub_boundary, fact_reduce_photomontage,np.array(S, dtype=np.float64),np.array(R, dtype=np.float64), camera.k1,camera.k2,camera.k3,camera.k4,camera.p1,camera.p2,camera.b1,camera.b2,camera.w,camera.h,camera.cx,camera.cy,camera.f)
                # print(len(directions_list))
        

        
                # Traitement batch
                image_rgba_projet, image_dproj_projet = self.calcul_image_rgb_from_mesh(
                    image_rgba_projet,
                    image_dproj_projet,
                    pixel_coords,
                    directions_list,
                    closest_rays,
                    S,
                    R,
                    light_dir,
                    view_dir,
                    projet
                )
                # pixel_total+=nb_pixels
                # print(f"Il reste {nbre_de_calcul-pixel_total} à calculer les intersections")
        
        image_arraymodifier=self.calcul_visibilite_projet(np.array(image, dtype=np.float64), np.array(depthmap_IA, dtype=np.float64), np.array(depthmap_IA_backup, dtype=np.float64),  boundaries, image_rgba_projet, image_dproj_projet, prec_depth, nbre_de_calcul, fact, fact_reduce_photomontage,S,R, camera.k1,camera.k2,camera.k3,camera.k4,camera.p1,camera.p2,camera.b1,camera.b2,camera.w,camera.h,camera.cx,camera.cy,camera.f, light_dir, view_dir,array_prof, larg_image)
        
        image_projet = Image.fromarray(image_rgba_projet.astype(np.uint8))
        image_projet.show()
        image_PIL = Image.fromarray(image_arraymodifier.astype(np.uint8))
        image_PIL.show()

        
    def calcul_visibilite_projet(self, image, depthmap_IA, depthmap_IA_backup, boundaries, image_rgba_projet, image_dproj_projet, prec_depth, nbre_de_calcul, fact, fact_reduce_photomontage,S,R, k1,k2,k3,k4,p1,p2,b1,b2,w,h,cx,cy,f, light_dir, view_dir,array_prof, larg_image):

        pourc_prec=0
        i=0
        for v in range(boundaries[1,0],boundaries[1,1]):
            for u in  range(boundaries[0,0],boundaries[0,1]):
                i+=1
                # depth_IA_value,depth_IA_value_origine, uv,pourc_prec,u_IA,v_IA= jit_fct.initialisation_projet_sur_image(nbre_de_calcul, i, pourc_prec,u,v,fact_reduce_photomontage, fact,depthmap_IA,depthmap_IA_backup)
                pourc=int(i/nbre_de_calcul*100)
                delta=int(i/nbre_de_calcul*100)%10
                if delta==0 and pourc_prec!=pourc:
                    print(f"Pourcentage traitée: {pourc}")
                    pourc_prec=pourc
                u_IA=int(u/fact)
                v_IA=int(v/fact)
                # uv_array=np.array([
                #     [u, v],
                #     ])
                depth_IA_value=depthmap_IA[v_IA, u_IA]
                
                depth_IA_value_origine=depthmap_IA_backup[v_IA, u_IA]
                # print(depth_IA_value)

                d_proj=image_dproj_projet[v,u]
                color=image_rgba_projet[v,u]
                
                color=[color[0],color[1],color[2]]




                
               
                if d_proj!=0:
                    uv=np.array([u/fact_reduce_photomontage,v/fact_reduce_photomontage], dtype=np.float64)
                    M,vect = jit_fct.uv_to_M_by_dist_prof(S,R, uv, depth_IA_value, k1,k2,k3,k4,p1,p2,b1,b2, w, h,cx,cy,f)
                    
                    
                    if depth_IA_value_origine==0 :
                        image[v,u]=color
                    else:
                        profondeur_valeur=None
                        depth_calc=None
                        

                        
                        
                        
                        if depth_IA_value>prec_depth*2.5+d_proj:
                            image[v,u]=color
                        elif self.projet_emprise.contains(M[np.newaxis, :])[0]:
                            # print("Points dans l'emprise")1
                            image[v,u]=color
                        elif depth_IA_value-prec_depth*2.5<d_proj:
                            
                            array_prof_15dm = jit_fct.return_array_epurer_from(u_IA, v_IA, depth_IA_value, array_prof, larg_image)
                            if array_prof_15dm is not None:
                                y_pred, unique_label = jit_fct.dbscan_non_optimise(array_prof_15dm)
                                profondeur_valeur=jit_fct.return_array_calcul_moindre_carre(y_pred, unique_label, depth_IA_value, array_prof_15dm)
                            
                            if profondeur_valeur is not None:
                                deptha=profondeur_valeur[:, 4][:, np.newaxis]
                                deptha=depth_IA_value-deptha
                                
                                array_prof=np.append(profondeur_valeur,deptha, axis=1)
                                array_prof = array_prof[array_prof[:, 5].argsort()]
                                
                                mask=array_prof[:, 5]>0
                                mask_array=array_prof[mask]
                                value_prov=mask_array[0,4]
                                # value_prov=jit_fct.calcul_value_prov(np.array(profondeur_valeur, dtype=np.float64), depth_IA_value)
                                if value_prov>d_proj:
                                    image[v,u]=color
                                    # image.putpixel((u, v), color)
                                else:
                                    
                                    inc, vi, wi, B_calc,s0, Quot, Kxx=jit_fct.gauss_markov(np.array(profondeur_valeur, dtype=np.float64), robuste=True, iter_robuste=4)

                                    
                                    if Quot<1.8:
                                        depth_calc=depth_IA_value*inc[0,0]+inc[1,0]
                                        if depth_calc>d_proj:
                                            image[v,u]=color
                                    else:
                                        if depth_IA_value>d_proj:
                                            
                                            image[v,u]=color
                                            a=1
                            else:
                                if depth_IA_value>d_proj:

                                    image[v,u]=color
                                    a=1
                                a=1
        return image
        
    def calcul_image_rgb_from_mesh(self, image_rgba_projet, image_dproj_projet, pixel_coords, directions_list, closest_rays, S,R, light_dir, view_dir, mesh):
        
        
        surface_color = (255, 0, 0)  # Couleur de la surface (rouge)
        # surface_colorb = (0, 0, 2555)
        ambient_intensity = 0.2  # Intensité de la lumière ambiante
        light_intensity = 5.0  # Intensité de la lumière diffusée
        specular_intensity = 0.5  # Intensité de la lumière spéculaire
        shininess = 5
        


        origins = np.repeat(np.array([S]), len(directions_list), axis=0)
        directions = np.array(directions_list, dtype=np.float64)
        
        
        # print("Début du calcul d'intersection")
        locs, ray_ids, tri_ids = mesh.ray.intersects_location(origins, directions)
        
        # print("Début du calcul de prepartion_donnee")
        # ray_ids = np.array(ray_ids, dtype=np.int64)
        # origins = np.array(origins, dtype=np.float64)
        
        
        if ray_ids.size != 0:
            closest_dist, closest_hit_idx=jit_fct.prep_donnee_intersection( np.array(locs, dtype=np.float64), np.array(ray_ids, dtype=np.int64),  np.array(tri_ids, dtype=np.int64), np.array(origins, dtype=np.float64), np.array(directions_list, dtype=np.float64))
        
        
            # for r_id, (dist2, i) in closest.items():
            for r_id in np.where(closest_hit_idx != -1)[0]:
                # print("Début du calcul de prepartion_donnee")
                i = closest_hit_idx[r_id]
                dist2=closest_dist[r_id]
                tri_id = tri_ids[i]
                vertex_indices = mesh.faces[tri_id]
                normal = mesh.face_normals[tri_id]
                
                try:
                    uvs = mesh.visual.uv[vertex_indices]
                    uv = np.array([[uvs[0, 0], uvs[0, 1]]])  # toujours simplifié
            
                    color_mesh = mesh.visual.material.to_color(uv)[0]
                    color_mesh=tuple(color_mesh)
                    # print(color)
                except:
                    color_mesh=(25,25,25)
                    
                # normal = mesh.face_normals[closest_triangle_idx]
                color=mm.phong_lighting(normal, light_dir, view_dir, color_mesh, ambient_intensity, light_intensity, specular_intensity, shininess)
                y, x = pixel_coords[r_id]
                
                vect = closest_rays[(y, x)]['vect']
                
                M=S+vect*np.sqrt(dist2)
                P_proj, d_proj=jit_fct.calcul_proj_cam(S, R, M)
                # print(d_proj)
                # print(dist2)
                # time.sleep(5)
    
                image_rgba_projet[y, x] = color
                image_dproj_projet[y, x]=d_proj
            
        return image_rgba_projet, image_dproj_projet
    def fusion_depthpro_depthanything(self, depthanything, depthpro=None):
        for i in range(depthpro.shape[0]):
            for j in range(depthpro.shape[1]):
                if depthanything[i,j]==0:
                    depthpro[i,j]=0
                    
        return depthpro
    
    def initialisaiton_depthmap(self):
        shape0=int(self.depthmap_IA.shape[0]*(self.fact_reduce_photomontage/self.fact_reduce_IA))
        shape1=int(self.depthmap_IA.shape[1]*(self.fact_reduce_photomontage/self.fact_reduce_IA))
        depth_init=np.ones((shape0, shape1))*-1
        
        
        return depth_init
    
    
    def dict_prof_TO_liste(self, dict_prof):
        liste_prof=[]
        for u, value in dict_prof.items():
            for v, value2 in value.items():
                prof_uv=np.array([u, v, value2["d"], value2["sigma"], value2["depthIA"], value2["point"], value2["normal"]], dtype=object)
                liste_prof.append(prof_uv)
                
        return liste_prof
    
    def liste_TO_dict_prof(self, liste_prof):
        dict_prof={}
        for obs in liste_prof:
            u=obs[0]
            v=obs[1]
            if u in dict_prof.keys():
                if v in dict_prof[u].keys():
                    if obs[2]<dict_prof[u][v]["d"]:
                        dict_prof[u][v]["d"]=float(obs[2])
                else:
                    dict_prof[u].update({v:{"d":float(obs[2]), "sigma":obs[3], "depthIA":float(obs[4]), "point":obs[5], "normal":obs[6]}})
            else:
                dict_prof[u]={v:{"d":float(obs[2]), "sigma":obs[3], "depthIA":float(obs[4]), "point":obs[5], "normal":obs[6]}}
        return dict(sorted(dict_prof.items()))
    
    def depthmap_to_liste(self, depthmap):
        liste_depthmap=[]
        for i in range(depthmap.shape[0]):
            for j in range(depthmap.shape[1]):
                liste_depthmap.append([i, j, depthmap[i,j]])
                
        return liste_depthmap
    def transformation_simple_depthmap_IA(self):
        """
        Cette fonction transforme la depthmap IA vers un depthmap approximative et proche de la valeur terrain

        Returns
        -------
        None.

        """
        dict_prof=self.dict_prof
        array_prof=np.array(self.dict_prof_TO_liste(dict_prof))
        array_prof = array_prof[array_prof[:, 2].argsort()]
        
        
        
        
        array_min=array_prof[0:10,:]
        x=array_prof[:,[4,2]]
        print(x)
        mask=return_mask_detection_donnee_aberrante(x)
        
        clusters, y_pred=pcd.DBSCAN_pointcloud(x, min_samples=5, n_neig=3)
        
        mask=y_pred==-1
        array_prof_corr=array_prof[~mask]
        
        array_prof_corr=np.vstack((array_min, array_prof_corr))
        

        nb_l=array_prof_corr.shape[0]
        B=array_prof_corr[:,2][:, np.newaxis]
        X=array_prof_corr[:,4]
        
        A=np.ones((nb_l,2))
        A[:,0]=X

        Qll=np.eye(nb_l, nb_l)

        sigma_B=np.array(range(nb_l))[:, np.newaxis].astype(float)
        sigma_B+=0.001
        Qll=sigma_B*Qll
        
        inc, vi, wi, B_calc,s0, Quot, Kxx=gauss_markov(np.array(Qll, dtype=np.float64), np.array(A, dtype=np.float64), np.array(B, dtype=np.float64))
        max_depth=np.max(X)
        X_range=np.array(range(int(max_depth)))[:, np.newaxis]
        a=np.ones((int(max_depth),1))
        
        A_range=np.column_stack((X_range, a))

        res=A_range@inc

        
        plt.plot(x[:,0], x[:,1], '.', alpha=0.3)
        plt.plot(X_range, res)
        
        plt.xlabel('Valeurs Depth IA', fontsize=12)
        plt.ylabel('Valeurs terrain (dproj)', fontsize=12)
        # ax.plot(inc, B)
        # plt.legend()
        plt.title("Valeur des profondeurs IA et monoplotting")
        plt.show()
        plt.close()
        
        
        self.depthmap_IA=self.depthmap_IA*inc[0,0]+inc[1,0]
        
        for i in dict_prof:
            for j in dict_prof[i]:
                self.dict_prof[i][j]["depthIA"]=float(inc[0,0]*self.dict_prof[i][j]["depthIA"]+inc[1,0])        
        
        
    def transformation_seconde_depthmap_IA(self):
        """
        Cette fonction transforme la depthmap IA vers un depthmap approximative et proche de la valeur terrain

        Returns
        -------
        None.

        """
        dict_prof=self.dict_prof
        array_prof=np.array(self.dict_prof_TO_liste(dict_prof))
        
        
        array_prof = array_prof[array_prof[:, 2].argsort()]
        
        array_min=array_prof[0:10,:]
        x=np.column_stack((array_prof[:,4],array_prof[:,2]))
        
        array_epurer=data_epurer_densite_point(array_prof, 4,2,25)
        array_epurer = array_epurer[array_epurer[:, 2].argsort()]
        # self.debug=array_epurer
        self.calcul_iteration_transformation_seconde(array_epurer)
        
        plt.show()
        plt.close()
        

        
    def fonction_lineaire_moindre_carre(self, array_2d):
        nb_l=array_2d.shape[0]
        B=array_2d[:,1][:, np.newaxis]
        X=array_2d[:,0]
        
        A=np.ones((nb_l,2))
        A[:,0]=X

        Qll=np.eye(nb_l, nb_l)

        # sigma_B=np.array(range(nb_l))[:, np.newaxis].astype(float)
        # sigma_B+=0.001
        # Qll=sigma_B*Qll
        
        inc, vi, wi, B_calc,s0, Quot, Kxx=gauss_markov(np.array(Qll, dtype=np.float64), np.array(A, dtype=np.float64), np.array(B, dtype=np.float64))
        a=inc[0,0]
        b=inc[1,0]
        return a, b, Quot
        
        
    def calcul_iteration_transformation_seconde(self, array_epurer, x_prec=None, liste_array=[]):
        
        liste_transfo=[]
        mask_ajout=[]

        array_cal=np.copy(array_epurer)

        mean_diff,sum_diff, std_diff,x1=self.determiner_diff_mean_extreminte_lineaire(array_cal, x_prec)
        print(f"mean_diff={mean_diff}")
        
        separer_nuage=False
        if array_cal.shape[0]>5:
            x=np.column_stack((array_cal[:,4],array_cal[:,2]))
            clusters, y_pred=pcd.KMEANS_pointcloud(x,n_cluster=2)
            
    
            # centroids = kmeans.cluster_centers_
            mask0=y_pred==1
            mask1=y_pred==0
            array_prof_0=array_cal[mask0]
            array_prof_1=array_cal[mask1]
            if np.min(array_prof_0[:,2])>np.min(array_prof_1[:,2]):
                array_epurer=array_prof_1
                array_suivant=array_prof_0
            else:
                array_epurer=array_prof_0
                array_suivant=array_prof_1
            if (mean_diff>0.2 and std_diff<0.4*mean_diff) or (mean_diff>2 and std_diff<1.5*mean_diff) or np.min(array_suivant[:,4])-np.max(array_epurer[:,4])>(np.max(array_cal[:,4])-np.min(array_cal[:,4]))*0.2:
                separer_nuage=True

            
        
        if separer_nuage :
            
            

            liste_array.insert(0, array_suivant)
            
            # self.debug=mask
            self.calcul_iteration_transformation_seconde(array_epurer, x_prec=x_prec, liste_array=liste_array)

            

            
        else:
            nb_l=array_cal.shape[0]
            B=array_cal[:,2][:, np.newaxis]-x1[1]
            X=array_cal[:,4]-x1[0]
            max_depth=np.max(array_cal[:,4])
            
            if x_prec is None:
                min_depth=0
            else:
                min_depth=x_prec[0]
            A=np.ones((nb_l,1))
            A[:,0]=X


            Qll=np.eye(nb_l, nb_l)

            inc1, vi, wi, B_calc,s0, Quot, Kxx=gauss_markov(np.array(Qll, dtype=np.float64), np.array(A, dtype=np.float64), np.array(B, dtype=np.float64))

            if inc1[0,0]<0:
                if len(liste_array)>0:
                    if np.min(liste_array[0][:,2])>np.mean(array_cal[:,2]) :
                        a=(np.min(liste_array[0][:,2])-np.mean(array_cal[:,2]))/(np.min(liste_array[0][:,4])-np.mean(array_cal[:,4]))*0.5
                    else:
                        a=(np.mean(liste_array[0][:,2])-np.mean(array_cal[:,2]))/(np.mean(liste_array[0][:,4])-np.mean(array_cal[:,4]))*0.5
                        
                else:
                    a=1.0
            else:
                a=inc1[0,0]
            b=x1[1]-a*x1[0]
            if len(liste_array)>0:
                if np.min(liste_array[0][:,4])<max_depth:
                    max_depth=np.min(liste_array[0][:,4])*0.9+np.mean(liste_array[0][:,4])*0.1
                
            if max_depth*a+b>np.max(array_cal[:,2]):
                max_depth=(np.max(array_cal[:,2])-b)/a
            if len(liste_array)>0:
                if np.min(liste_array[0][:,4])<np.max(array_cal[:,4]):
                    max_depth=np.mean(liste_array[0][:,4])*0.25+np.min(liste_array[0][:,4])*0.75
            
            
            if len(liste_array)==0:
                max_depth=500
                max_depth=35

            plt.plot(X+x1[0], B+x1[1], ".", alpha=0.3)
            plt.plot([min_depth, max_depth],[min_depth*a+b, max_depth*a+b], c="red")
            

            
            
            mask=(self.depthmap_IA > min_depth) & (self.depthmap_IA < max_depth)
            
            self.transformation_lineaire_depthmap_ia_and_dict_prof(a,b, min_depth, max_depth)
            

            
            x_prec=np.array([max_depth, max_depth*a+b], dtype=float)
            if len(liste_array)>0:
                if np.min(liste_array[0][:,4])-np.max(array_cal[:,4])>(np.max(liste_array[0][:,4])-np.min(array_cal[:,4]))*0.2:
                    max_dproj3=np.min(liste_array[0][:,2])
                    max_depth3=np.min(liste_array[0][:,4])
                    min_depth2=x_prec[0]
                    a2=(max_dproj3-x_prec[1])/(max_depth3-x_prec[0])
                    if a2>a:
                        a2=(a+a2)/2

                    max_depth2=max_depth3-(max_depth3-x_prec[0])/2
                    b2=x_prec[1]-a2*x_prec[0]
                    self.transformation_lineaire_depthmap_ia_and_dict_prof(a2,b2, min_depth2, max_depth2)
                    
                    
                    plt.plot([min_depth2, max_depth2],[min_depth2*a2+b2,max_depth2*a2+b2], c="red")
                    
                    x_prec=np.array([max_depth2, max_depth2*a2+b2], dtype=float)
                    
                    a3=(max_dproj3-x_prec[1])/(max_depth3-x_prec[0])
                    b3=x_prec[1]-a3*x_prec[0]
                    
                    self.transformation_lineaire_depthmap_ia_and_dict_prof(a3,b3, max_depth2, max_depth3)

                    plt.plot([max_depth2, max_depth3],[max_depth2*a3+b3,max_depth3*a3+b3], c="red")
                    x_prec=np.array([max_depth3, max_depth3*a3+b3], dtype=float)
                    
                    
                    
                    
                    
                    
                    
                    

                
            
            
                    
            if len(liste_array)>0:

                print("Nouveau calcul")
                self.calcul_iteration_transformation_seconde(liste_array[0], x_prec=x_prec, liste_array=liste_array[1:])

    def transformation_lineaire_depthmap_ia_and_dict_prof(self, a,b, min_depth, max_depth):

        self.depthmap_IA = np.where((self.depthmap_IA >= min_depth) & (self.depthmap_IA < max_depth), (self.depthmap_IA)*a+b, self.depthmap_IA)
        for i in self.dict_prof:
            for j in self.dict_prof[i]:
                if self.dict_prof[i][j]["depthIA"]>=min_depth and self.dict_prof[i][j]["depthIA"]<max_depth :
                    self.dict_prof[i][j]["depthIA"]=float(a*self.dict_prof[i][j]["depthIA"]+b)
        
    def determiner_diff_mean_extreminte_lineaire(self, array_epurer, x_prec=None):
        array_first_10_columns = array_epurer[:5]
        array_last_10_columns = array_epurer[-2:]
        x1=np.array([np.mean(array_first_10_columns[:,4]),np.mean(array_first_10_columns[:,2])])
        x2=np.array([np.mean(array_last_10_columns[:,4]),np.mean(array_last_10_columns[:,2])])
        if x_prec is  None:
            x_prec=x1
        if array_epurer.shape[0]<50:
            return 0.0, 0.0,0.0,x_prec
        
        else:
            
            
            a,b=calcul_fct_lineaire(x1, x2)
            # plt.plot([x1[0], x2[0]],[x1[1], x2[1]], c="blue")
            res=a*array_epurer[:,4]+b
            diff=res-array_epurer[:,4]

            return np.abs(np.mean(diff)), np.sum(diff), np.std(diff), x_prec
        
    def creer_grille_point(self):
        width=self.depthmap_ajustee.shape[1]
        height=self.depthmap_ajustee.shape[0]
        intervalle_u=creer_liste_intervalles(10, width)
        intervalle_v=creer_liste_intervalles(10, height)
        liste_points_grille=[]
        clusters_dict = self.param_transfo_cluster
        fact=self.fact_reduce_photomontage/self.fact_reduce_IA
        for u in intervalle_u:
            for v in intervalle_v:
                depthajuste_value=self.depthmap_ajustee[v[1], u[1]]
                depthIA=self.depthmap_IA[int(v[1]/fact), int(u[1]/fact)]
                if depthajuste_value>0 and depthajuste_value!=9999:
                    point=np.array([u[1], v[1],depthajuste_value,0.1, depthIA])
                    liste_points_grille.append(point)
                    id_cluster=str(int(self.depthmap_clusters[v[1], u[1]]))
                    if id_cluster in clusters_dict:
                        clusters_dict.pop(id_cluster)
        
        
        for id_cluster in clusters_dict:
            centre= clusters_dict[id_cluster]["enveloppe"].centroid
            coords = list(centre.coords)[0]
            depthIA=self.depthmap_IA[int(v[1]/fact), int(u[1]/fact)]
            depthajuste_value=self.depthmap_ajustee[int(coords[1]*fact), int(coords[0]*fact)]
            if depthajuste_value>0 and depthajuste_value!=9999:
                point=np.array([coords[0]*fact, coords[1]*fact,depthajuste_value,0.1, depthIA])
                liste_points_grille.append(point)
                
                
                
            
        array_grillee=np.array(liste_points_grille)
        plt.plot(array_grillee[:,0],-array_grillee[:,1], ".")
        plt.show()
        plt.close()
        
        self.grille_calcul=liste_points_grille
        return liste_points_grille

    
    
    def creation_des_clusters(self, array_prof, index=0, view=False, first=False):
        list_clusters=[]

        if first==True:
            x=np.column_stack((np.array(array_prof[:,5].tolist()),array_prof[:,4]))
            # min_samples=int(array_prof.shape[0]**(1/8))
            clusters, y_pred=pcd.DBSCAN_pointcloud(x, min_samples=2, n_neig=3)
            unique_labels = np.unique(y_pred)
        else:
            x=np.column_stack((np.array(array_prof[:,5].tolist()),array_prof[:,4]))
            # min_samples=int(array_prof.shape[0]**(1/8))
            clusters, y_pred=pcd.DBSCAN_pointcloud(x, min_samples=2, n_neig=3)
            unique_labels = np.unique(y_pred)
            

        # x_debruite, point_colors = pcd.array_et_colors_set_from_clusters(x, y_pred)
        # pcd.view_point_cloud_from_array(np.delete(x_debruite, 3, axis=1), color=point_colors)
    
        
        for i in range(len(unique_labels)):
            if unique_labels[i]>-1:
                mask_noise=y_pred==unique_labels[i]
                array_cluster=array_prof[mask_noise]
                if array_cluster.shape[0]>5:
                    
                    #RAJOUTER LE RANSAC EST SI OUTLIER RANSAC PLUS GRAND QUE X FAIRE UN DBSCAN DU GROUPE SUR LES DISTANCE PROJ ET LES VALEURS DEPHTMAPS
                    inliers, outlier=self.ransac_simple(array_cluster.tolist())
                    # plot.plot_from_liste_prof(outlier.tolist())
                    
                    # print(f"Nbre de valeur en outlier {len(outlier)}")
                    if len(outlier)>50 and index<6:
                        cluster=self.creation_des_clusters(array_cluster, index+1)
                        list_clusters=list_clusters+cluster
                    else:
                        # array_cluster=np.array(inliers)
                        min_d=np.min(array_cluster[:,2])
                        max_d=np.max(array_cluster[:,2])
                        cluster=[[ array_cluster, min_d, max_d]]
                        list_clusters=list_clusters+cluster
                    
        return list_clusters
    
    
    def optimisation_des_clusters(self):
        list_prof=self.dict_prof_TO_liste(self.dict_prof)
        array_prof=np.array(list_prof)
        list_clusters=self.creation_des_clusters(array_prof, view=True, first=True)
                #Ajouter pour pouvoir ensuite trier les clister du plus proche au plus loin
        list_clusters =sorted(list_clusters, key=lambda x: x[1])
        self.clusters=list_clusters
        print(f"DEBUT DU CALCUL DES CLUSTERS")
        print(f"--------------------------------------------------")
        points_cluster_uv=[]
        for i, cluster in enumerate(list_clusters):
            points_uv=self.calcul_transformation_cluster( cluster[0], i)
            points_cluster_uv=points_cluster_uv+points_uv
        print(f"Nbre de points {len(points_cluster_uv)}")
        print(f"FIN DU CALCUL DES CLUSTERS")
        print(f"--------------------------------------------------")


        # pcd.view_point_cloud_from_array(np.array(points_cluster_uv))
        # pcd.save_point_cloud_las(np.array(points_cluster_uv), "Res_cluster.las")
        # self.save_image_depthmap(self.depthmap_ajustee, "resultat_cluster")
        
        return list_clusters
        

        
    def calcul_transformation_cluster(self, cluster_array, cluster_index):
        bounding_box_depthmap=[np.min(cluster_array[:,4]), np.max(cluster_array[:,4])]
        nb_l=cluster_array.shape[0]
        

        print(f"Debut du calcul de cluster {cluster_index}")
        nb_i=3
        A=np.zeros((nb_l,nb_i))
        B=np.zeros((nb_l,1))
        Kll=np.eye(nb_l)
        sigma0=1
        X=np.zeros((nb_l,1))
        
        #CALCUL DE L'ENVELOPPE DE CLUSTER SUR L'IMAGE
        #========================================================================================
        uv_array=np.column_stack((cluster_array[:,0],cluster_array[:,1]))    
        uv_tuple=[tuple(point) for point in uv_array]
        
        if nb_l<50:
            alpha=0 
        else:
            alpha = 0.05  # Paramètre alpha à ajuster selon la densité de vos points
        alpha_shape = alphashape.alphashape(uv_tuple, alpha)

        print(alpha_shape.geom_type)
        
        if alpha_shape.geom_type == 'Polygon':
            x, y = alpha_shape.exterior.xy
            # plt.fill(x, y, alpha=0.2, color='green', label='AlphaShape')
        # elif alpha_shape.geom_type == 'MultiPolygon':
        #     print("Enveloppe multyPolygon")
        elif alpha_shape.geom_type == 'LineString':
            return []
        # plt.plot(uv_array[:, 0], uv_array[:, 1], 'bo', label='Points originaux')
        # plt.show()
        # plt.close()
        
        
        
        for i in range(nb_l):
            obs=cluster_array[i].tolist()
            u=int(obs[0])
            v=int(obs[1])
            B[i,0]=obs[2]
            sigma=obs[3]
            # d_proche=obs[3]
            x=obs[4]

            X[i,0]=x

            Kll[i,i]=sigma**2

            
            A[i,0]=u
            # A[i,1]=v
            A[i,1]=x
            A[i,2]=1

        Qll=1/sigma0**2*Kll
        inc, vi, wi, B_calc,s0, Quot, Kxx=gauss_markov(np.array(Qll, dtype=np.float64), np.array(A, dtype=np.float64), np.array(B, dtype=np.float64), robuste=False, print_res=True)

        # inc  = self.optimisation_quadratique(Qll,A, B)
        
        
        points_uv=[]
        if Quot<2.5:
        # if inc is not None:
            self.param_transfo_cluster[str(cluster_index)]={"inc":inc, "quot": Quot, "enveloppe" : alpha_shape, "bounding_box": bounding_box_depthmap}
            fact=self.fact_reduce_photomontage/self.fact_reduce_IA
            bounding_box=np.array([[np.min(cluster_array[:,0]), np.max(cluster_array[:,1])],[np.max(cluster_array[:,0]), np.min(cluster_array[:,1])]])
            bounding_box=bounding_box*fact
            bounding_box=np.floor(bounding_box).astype(int)

            for u in range(bounding_box[0][0], bounding_box[1][0]+1):
                # print(i)
                for v in range(bounding_box[1][1], bounding_box[0][1]+1):
                    
                    ij=np.array([u/fact, v/fact])
                    uv=np.array([u/self.fact_reduce_photomontage, v/self.fact_reduce_photomontage])
                    
                    u_IA=int(np.floor(u/fact))
                    v_IA=int(np.floor(v/fact))
                    
                    if self.depthmap_ajustee[v, u]<=0:
                        point = Point(ij)
                        if alpha_shape.contains(point):
                            depth_IA_value=self.depthmap_IA[v_IA,u_IA]
                            if depth_IA_value==0:
                                dist=9999
                            elif bounding_box_depthmap[0]<depth_IA_value<bounding_box_depthmap[1]:
                                dist=0
                                # dist=inc[0,0]*u+inc[1,0]*v+inc[2,0]*depth_IA_value+inc[3,0]
                                dist=inc[0,0]*u_IA+inc[1,0]*depth_IA_value+inc[2,0]
                                # points_uv.append(self.camera.uv_to_M_by_dist_prof(self.photoname, uv, dist))
                                
                                self.depthmap_clusters[v,u]=cluster_index
                                self.depthmap_ajustee[v,u]=dist
        
        
        # plot.plot_mesure_calcule(X, B, B_calc, "avec RANSAC", outlier[])
        # pcd.view_point_cloud_from_array(np.array(points_uv))
        # pcd.save_point_cloud_las(np.array(points_uv), "Res_cluster")
        return points_uv

    def return_array_epurer_from(self, u_IA, v_IA):
        
        
        uv_decalage=self.depthmap_IA.shape[1]//54


        liste_u=list(range(u_IA-uv_decalage,u_IA+uv_decalage+1))
        groupe_mesure = {key: self.dict_prof[key] for key in liste_u if key in self.dict_prof}


        liste_prof=self.dict_prof_TO_liste(groupe_mesure)
        array_prof=np.array(liste_prof)
        # plot.plot_from_liste_prof(liste_prof)
        value_depth=self.depthmap_IA[v_IA,u_IA]
        # print(f"Valeur depthIA: {value_depth} et ")

        if array_prof.shape[0]< 5:
            return None
        
        diff_depth=np.abs(array_prof[:,4][:, np.newaxis]-value_depth)

        array_prof=np.append(array_prof,diff_depth, axis=1)

        array_prof=tri_array_par_rapport_a_une_colonne(array_prof, 7)


        mask_dist=array_prof[:,7]<1.5
        array_prof_15dm=array_prof[mask_dist]

        if array_prof_15dm.shape[0]< 5:
            return None
        array_test=array_prof_15dm[:,[2, 4]]
        
        clusters, y_pred=pcd.DBSCAN_pointcloud(array_test, min_samples=2, n_neig=4)
        unique_label=np.unique(y_pred)
        
        # x, color=pcd.array_et_colors_set_from_clusters(array_prof_15dm, y_pred)
        # plt.scatter(x[:,2], x[:,4], c=color)
        # plt.show()
        # plt.close()
        liste_label_include=[]
        for i in unique_label:

            if i!=-1:
                
                mask=y_pred==i
                array_mask=array_prof_15dm[mask]
                # inliers, outlier=self.ransac_simple(array_mask.tolist(), min_samples=2)
                if np.min(np.min(array_mask[:,2]))<1.5*value_depth:
                    if np.min(array_mask[:,4])<value_depth and np.max(array_mask[:,4])>value_depth:
                        liste_label_include.append(i)

        if len(liste_label_include)==1:
            mask=y_pred==liste_label_include[0]
            
            if array_prof_15dm[mask].shape[0]>4:
                array_res=array_prof_15dm[mask]
                # inliers, outlier=self.ransac_simple(array_res.tolist(), min_samples=2)
                if array_res.shape[0]>4:
                    return array_res[:, [0,1,2,3,4]]
                else:
                    return None

        return None
        
        
    def calcul_dist_ajustee(self, calcul_pt_homologue=True):
        sigma0=1
        grille=self.creer_grille_point()

        grille_array=np.array(grille)
        
        fact=self.fact_reduce_photomontage/self.fact_reduce_IA
        
        u_liste=grille_array[:,0][:, np.newaxis]
        
        u_liste=u_liste.astype(int)
        v_liste=grille_array[:,1][:, np.newaxis]
        v_liste=v_liste.astype(int)
        uv_grille=np.hstack((u_liste, v_liste))


        boundaries=self.boundaries_images_reduites()
        
        res_ok=[]
        for v in range(boundaries[1,0],boundaries[1,1]):
            for u in  range(boundaries[0,0],boundaries[0,1]):
                u_IA=int(u/fact)
                v_IA=int(v/fact)
                uv_array=np.array([
                    [u, v],
                    ])

                depth_IA_value=self.depthmap_IA[v_IA, u_IA]
                
                depth_IA_value_origine=self.depthmap_IA_backup[v_IA, u_IA]
                
                depthmap_ajustee_value= self.depthmap_ajustee[v, u]
                if depthmap_ajustee_value<=0:
                    depth_calc=None
                    if depth_IA_value_origine==0:
                        depth_calc=9999
                    else:
                        
                        if calcul_pt_homologue:
                            profondeur_valeur = self.return_array_epurer_from(u_IA, v_IA)

                            if profondeur_valeur is not None:
                                nb_l=profondeur_valeur.shape[0]
                                B=profondeur_valeur[:,2][:, np.newaxis].astype(float)
                                X=profondeur_valeur[:,4][:, np.newaxis].astype(float)
                                
                                A=np.ones((nb_l,2))
                                A[:,0]=X[:,0]
                                Qll=np.eye(nb_l)*0.1
                                inc, vi, wi, B_calc,s0, Quot, Kxx=gauss_markov(np.array(Qll, dtype=np.float64), np.array(A, dtype=np.float64), np.array(B, dtype=np.float64), robuste=True, iter_robuste=4)
                                
                                X_range=np.array(range(int(np.min(X)),int(np.max(X))+2))[:, np.newaxis]
                                a=np.ones((int(np.max(X))+2-int(np.min(X)),1))
                                
                                A_range=np.column_stack((X_range, a))
                                res=A_range@inc

                            
                                
                                if Quot<1.1:
                                    A_cacl=np.array([
                                        [depth_IA_value,1]
                                        ])
                                    depth_calc=A_cacl@inc
                                    # res_ok.append([u, v, depth_IA_value, depth_calc, inc,np.max(vi), np.max(wi), X_range, res, X, B])
                                    
                                    
                                            
                                        
                                    
                                    
                                    # plt.plot(X_range, res)
                                    # plt.plot(X[:,0], B[:,0], '.', c="green", alpha=0.3)
                                    # plt.plot([depth_IA_value], [depth_calc[0,0]])
                                    # print(depth_IA_value)
                                    # print(depth_calc)
                                    # plt.title(str(u)+"/"+str(v))
                                    # plt.show()
                                    # plt.close()
                                    # print(f"Le calcul de gauss-markov n'est rend pas un résultat juste avec un quotien de {Quot} et un vi maximum de {np.max(vi)} et un wi max de {np.max(wi)}")
                                # else:
                                    # print(f"Le calcul de gauss-markov n'est rend pas un résultat juste avec un quotien de {Quot} et un vi maximum de {np.max(vi)} et un wi max de {np.max(wi)}")
                                    
                                    # plt.plot(X[:,0], B[:,0], '.', c="red", alpha=0.3)
                                    # plt.plot(X_range, res)
                                    # plt.show()
                                    # plt.close()
                                    
                        else:
                            diff_uv=uv_grille-uv_array
                            dist_grille_uv=diff_uv@diff_uv.T
                            dist_grille_uv = np.diagonal(dist_grille_uv)
                            dist_grille_uv = dist_grille_uv**0.5
                            
                            indice_grille=dist_grille_uv<self.depthmap_ajustee.shape[1]//3
                            
                            profondeur_valeur=grille_array[indice_grille]
    
                                
                            nb_l=profondeur_valeur.shape[0]
                            B=profondeur_valeur[:,2][:, np.newaxis].astype(float)
                            X=profondeur_valeur[:,4][:, np.newaxis].astype(float)
                            
                            u_liste_epurer=profondeur_valeur[:,0][:, np.newaxis]
    
                            A=np.ones((nb_l,3))
                            A[:,0]=u_liste_epurer[:,0]
                            A[:,1]=X[:,0]
                            Kll=np.eye(nb_l)
    
                            
                            dist_grille_uv=np.abs(u_liste_epurer-u)
                            y_uv=(1.6**dist_grille_uv**0.52)/200+0.1
                            
    
                            diff_depthvalue=np.abs(X-depth_IA_value)
                            
                            # y_depth=(1.65**diff_depthvalue**0.63)/200+0.1
                            y_depth=diff_depthvalue**5/50+0.1
                            # y_depth=1
                            
                            Qll=Kll*(y_depth+y_uv+0.00001)
                            Qll=np.asarray(Qll, dtype=np.float64) #Corriger pour avoir du float
    
                            inc, vi, wi, B_calc,s0, Quot, Kxx=gauss_markov(np.array(Qll, dtype=np.float64), np.array(A, dtype=np.float64), np.array(B, dtype=np.float64))
                        
                            depth_calc=inc[0,0]*u+inc[1,0]*depth_IA_value+inc[2,0]
                            
                    if depth_calc is not None:
                        nb_i=A.shape[1]
                        
                        
                        self.depthmap_ajustee[v, u]=float(depth_calc)

                # self.calcul_dist_ajust_from_uv(uv_array, 300,400)
                if v%50==0 and u%50==0:
                    print(f"Les éléments ont été calculés jusqu' à l'uv {u} {v}")
    
        print("Les depthmap ajustées est entièrement calculée")
        return np.array(res_ok,  dtype=object)
    def calcul_prof_ajustee_from_homol(self, u_IA, v_IA):
        profondeur_valeur = self.return_array_epurer_from(u_IA, v_IA)
        depth_IA_value=self.depthmap_IA[v_IA, u_IA]
        print(profondeur_valeur)
        if profondeur_valeur is not None:
            nb_l=profondeur_valeur.shape[0]
            B=profondeur_valeur[:,2][:, np.newaxis].astype(float)
            X=profondeur_valeur[:,4][:, np.newaxis].astype(float)
            
            A=np.ones((nb_l,2))
            A[:,0]=X[:,0]
            Qll=np.eye(nb_l)*0.1
            inc, vi, wi, B_calc,s0, Quot, Kxx=gauss_markov(np.array(Qll, dtype=np.float64), np.array(A, dtype=np.float64), np.array(B, dtype=np.float64), robuste=True, iter_robuste=4)
            
            X_range=np.array(range(int(np.min(X)),int(np.max(X))+2))[:, np.newaxis]
            a=np.ones((int(np.max(X))+2-int(np.min(X)),1))
            
            A_range=np.column_stack((X_range, a))
            res=A_range@inc

            
            # if Quot<8.1:
            A_cacl=np.array([
                [depth_IA_value,1]
                ])
            depth_calc=A_cacl@inc
            plt.plot(X_range, res, label="Fonction linéaire", c="orange")
            plt.plot(X[:,0], B[:,0], '.', c="green", alpha=0.3, label="Points homologues")
            plt.plot([depth_IA_value], [depth_calc[0,0]], "o", c="red", label="Résultat")
            plt.xlabel('Valeurs Depth transformées (approx.)', fontsize=12)
            plt.ylabel('Valeurs terrain (dproj)', fontsize=12)
            plt.title("Calcul optimisé par moindre carré d'un pixel")
            plt.legend()
            
            # print(depth_IA_value)
            print(Quot)
            # plt.title(str(u)+"/"+str(v))
            plt.show()
            plt.close()
            
            
            return depth_calc
            # else:
            #     return None
            #     # res_ok.append([u, v, depth_IA_value, depth_calc, inc,np.max(vi), np.max(wi), X_range, res, X, B])
                
                
                        
                    
                
                
                
                # print(f"Le calcul de gauss-markov n'est rend pas un résultat juste avec un quotien de {Quot} et un vi maximum de {np.max(vi)} et un wi max de {np.max(wi)}")
            # else:
                # print(f"Le calcul de gauss-markov n'est rend pas un résultat juste avec un quotien de {Quot} et un vi maximum de {np.max(vi)} et un wi max de {np.max(wi)}")
                
                # plt.plot(X[:,0], B[:,0], '.', c="red", alpha=0.3)
                # plt.plot(X_range, res)
                # plt.show()
                # plt.close()
                
    
    def filtre_depthIA_pt_homologue(self):
        liste_prof=self.dict_prof_TO_liste(self.dict_prof)
        array_prof=np.array(liste_prof)
        max_mono=max(array_prof[:,2])
        min_mono=min(array_prof[:,2])
        max_ia=max(array_prof[:,4])
        min_ia=min(array_prof[:,4])
        
        a=(max_mono-min_mono)/(max_ia-min_ia)
        b=max_mono-a*max_ia
        
        points_pour_filtre=[] 
        
        
        for i in range(len(liste_prof)):
            points_pour_filtre.append([liste_prof[i][0],liste_prof[i][1], liste_prof[i][2]-a*liste_prof[i][4]+b])
            
        return np.array(points_pour_filtre)
        
    
    def suppression_mesures_fausse_liste_prof(self, liste_prof):
 

        liste_prof_array=np.array(liste_prof)

        liste_prof_array=tri_array_par_rapport_a_une_colonne(liste_prof_array, 4)
        max_value_depthmap=np.max(liste_prof_array[:, 4])


        intervalle_nbre=len(liste_prof)//5
        
        
        liste_prof_ajustee=[]
        liste_prof_supprimer=[]
        means=[]
        std_means=[]
        intervalles=creer_liste_intervalles(intervalle_nbre, liste_prof_array.shape[0])
        
        for intervalle in intervalles:
            i=intervalle[0]
            j=intervalle[1]
            
            sous_ensemble = liste_prof_array[i:j]
        # for i in range(nb_intervalle):
            # if intervalle*(i+1)>(max_value_depthmap-intervalle):
            #     masque = (liste_prof_array[:, 4] >= intervalle*i) & (liste_prof_array[:, 4] <= max_value_depthmap)
            # else:
            #     masque = (liste_prof_array[:, 4] >= intervalle*i) & (liste_prof_array[:, 4] < intervalle*(i+1))
            # sous_ensemble = liste_prof_array[masque]
            if len(sous_ensemble)>1:
                mean_local=np.mean(sous_ensemble[:,2])
                mean_std=np.std(sous_ensemble[:,2])
                mediane_inter=np.median(sous_ensemble[:,2])
            else:
                mean_local=0
                mean_std=9999
                min_depthmap=np.min(sous_ensemble[:,2])
                max_depthmap=np.min(sous_ensemble[:,2])
                mediane_inter=(max_depthmap+min_depthmap)/2
            mean=[]
            # print(f"Nbre de valeru dans l'intervalle : {len(sous_ensemble)}")

            if len(means)>0:
                for j, value in enumerate(sous_ensemble):
                    if value[2]<means[-1]+std_means[-1] and value[4]>0 and mean_local+2.5*mean_std>value[2]>mean_local-2.5*mean_std:
                        liste_prof_ajustee.append(value)
                        mean.append(value[2])
                    else:
                        liste_prof_supprimer.append(value)

            else:
                for j, value in enumerate(sous_ensemble):
                    
                    if value[4]>0 and mean_local+2.5*mean_std>value[2]>mean_local-2.5*mean_std:
                        liste_prof_ajustee.append(value)
                        mean.append(value[2])
                    else:
                        liste_prof_supprimer.append(value)
                        
            if len(mean)>1:
                # print(f"nbre de mesure = {len(mean)}")
                means.append(np.mean(mean))
                std_means.append(np.std(mean))
                
        return liste_prof_ajustee, liste_prof_supprimer
    
    
    
    def ransac_simple(self, liste_prof, min_samples=3):
        
        # min_samples=int(len(liste_prof)**0.1)
        ransac = RANSACRegressor(
            LinearRegression(),
            max_trials=100,
            min_samples=min_samples,
            residual_threshold=2,
            random_state=0,
        )
        
        array_prof=np.array(liste_prof, dtype=object)
        X=array_prof[:,4].reshape(-1, 1)
        y=array_prof[:,2].flatten()
        
        
        quadratic = PolynomialFeatures(degree=2)
        X_quad = quadratic.fit_transform(X)
        ransac = ransac.fit(X_quad, y)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        inlier_indices = np.where(inlier_mask)[0]
        outlier_indices = np.where(outlier_mask)[0]
        inliers = array_prof[inlier_indices]
        outlier = array_prof[outlier_indices]
        
        X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
        y_quad_fit = ransac.predict(quadratic.fit_transform(X_fit))
        
        return inliers, outlier
        


    def update_depthmap_ajustee(self, param_poly, de, jusque, min_max_mesure):
        # print(min_max_mesure)
        
        nb_poly=param_poly.shape[0]
        
        for i in range(self.depthmap_IA.shape[0]):
            for j in range(de, jusque+1):
                depth_IA_value=self.depthmap_IA[i,j]
                
                    
                dist=0
                if depth_IA_value==0:
                    dist=9999
                elif min_max_mesure[0]>depth_IA_value or depth_IA_value>min_max_mesure[1]:    
                    dist=0
                else:
                    for n in range(nb_poly):
                        nb=nb_poly-1-n
                        dist+=param_poly[n,0]*depth_IA_value**nb
                self.depthmap_ajustee[i,j]=dist
                
                
    def save_pointcloud_from_depthmap(self, depthmap, pointcloud_name):
        points=[]
        for i in range(self.depthmap_ajustee.shape[0]):
            for j in range(self.depthmap_ajustee.shape[1]):
                
                uv=np.array([j/self.fact_reduce_photomontage,i/self.fact_reduce_photomontage])
                d_proj=self.depthmap_ajustee[i,j]
                
                if d_proj>0 and d_proj<1000:
                    points.append(self.camera.uv_to_M_by_dist_prof(self.photoname, uv, d_proj))

        pcd.save_point_cloud_las(np.array(points), pointcloud_name+".las")
        return points
        
    def save_image_depthmap(self, depthmap, image_name):
        image=Image.fromarray(depthmap)
        image.save(image_name+".tif")
        
    def depthmap_ia_to_o3d_pcd(self):
        points=[]
        for i in range(self.depthmap_IA.shape[0]):
            for j in range(self.depthmap_IA.shape[1]):
                points.append(np.array([j,-i,self.depthmap_IA[i,j]]))
        points=np.array(points)
        pcd_o3d = pcd.array_TO_o3d(points)
        
        return pcd_o3d
    
    def initialisation_calc_par_groupe_colonne_pour_moindre_carre(self,  nbre_mesure=25):
        self.liste_groupe_resultat=[]
        
        
        for intervalle in creer_liste_intervalles(nbre_mesure,self.depthmap_IA.shape[1]):
            i=intervalle[0]
            j=intervalle[1]
            print(f"Initilisaiton du calcul pour l'intervalle de {i} à {j}")
            groupe_mesure = dict(list(self.dict_prof.items())[i:j])
            liste_prof=self.dict_prof_TO_liste(groupe_mesure)
            print(f"Nbre de mesure de base: {len(liste_prof)}")
            liste_prof_ajuste, liste_prof_supprimée=self.suppression_mesures_fausse_liste_prof(liste_prof)
            print(f"Nbre de mesure après le 1er tri: {len(liste_prof_ajuste)}")
            
            liste_ajustee_RANSAC, liste_supprimee_RANSAC=self.RANSAC_epuration(liste_prof_ajuste)
            print(f"Nbre de mesure après le RANSAC: {len(liste_ajustee_RANSAC)}")
            liste_prof_supprimée=liste_prof_supprimée+liste_supprimee_RANSAC
            A, B,Qll, inc, wi, vi, X, B_calc,s0=self.tranformation_depthanything_gauss(liste_ajustee_RANSAC)
            self.liste_groupe_resultat.append({
                "Res_poly": inc,
                "wi":wi,
                "vi":vi,
                "depthIA":X,
                "Monoplotting" : B,
                "Res_obs":B_calc,
                "s0":s0
                })
            min_max_mesure=[np.min(np.array(liste_ajustee_RANSAC)[:,4]),np.max(np.array(liste_ajustee_RANSAC)[:,4])]
            self.update_depthmap_ajustee(inc,i,j, min_max_mesure)
            
            

            X_epurer=[]
            Y_epurer=[]
            for i in range(len(liste_prof_supprimée)):

                X_epurer.append(liste_prof_supprimée[i][4])
                Y_epurer.append(liste_prof_supprimée[i][2])
                

            for i in range(Qll.shape[1]):
                if Qll[i,i]>5:
                    X_epurer.append(X[i,0])
                    Y_epurer.append(B[i,0])
                    
            # plot.plot_mesure_calcule(X, B, B_calc,"Avec RANSAC et analyse des points", X_epurer, Y_epurer)
        
        self.save_pointcloud_from_depthmap(self.depthmap_ajustee, "resultat")
        self.save_image_depthmap(self.depthmap_ajustee, "resultat")
            
    def tranformation_depthanything_gauss(self, liste_prof, rob=4, nbre_inc=5):
        # print(liste_prof)
        print(f"calcul de gauss-markov")
        nbre_valeur=100
        nb_i=nbre_inc
        nb_l=len(liste_prof)
        A=np.zeros((nb_l,nb_i))
        B=np.zeros((nb_l,1))
        Qll=np.eye(nb_l)
        X=np.zeros((nb_l,1))

        
        for i in range(nb_l):
            obs=liste_prof[i]
            u=int(obs[0])
            v=int(obs[1])
            B[i,0]=obs[2]
            sigma=obs[3]
            # d_proche=obs[3]
            x=obs[4]

            X[i,0]=x

            Qll[i,i]=sigma
            # elif  d_proche>(mean_d+std_d):
            #     print(f"Mesure {i} dépondérée car distance éloignée")
        #         Qll[i,i]=99999
            
            
            
            for j in range(nb_i):
                A[i, j]=x**(nb_i-1-j)
                
                
        inc, vi, wi, B_calc,s0, Quot, Kxx=gauss_markov(np.array(Qll, dtype=np.float64), np.array(A, dtype=np.float64), np.array(B, dtype=np.float64), robuste=False)

            
        print(f"calcul de gauss-markov terminé")
        
        return A, B,Qll, inc, wi, vi, X, B_calc,s0, Quot
    
    
    def RANSAC_epuration(self, liste_prof, nbre_intervalle=4):
        
        
        #RANSAC REDUCTION 
        liste_prof_array=np.array(liste_prof)
        nbre_point=liste_prof_array.shape[0]
        liste_prof_array=self.tri_array_par_rapport_a_une_colonne(liste_prof_array, 4)
        nbre_mesure=nbre_point//nbre_intervalle

        liste_ajustee_RANSAC=[]
        liste_supprimee_RANSAC=[]
        for i in range(0, liste_prof_array.shape[0], nbre_mesure):
            
            sous_ensemble = liste_prof_array[i:i + nbre_mesure]
            X=sous_ensemble[:,4].reshape(-1, 1)
            y=sous_ensemble[:,2].flatten()
            if len(sous_ensemble)>10:
            # Robustly fit linear model with RANSAC algorithm
                ransac = RANSACRegressor(
                    LinearRegression(),
                    max_trials=100,
                    min_samples=10,
                    residual_threshold=2,
                    random_state=0,
                )
                quadratic = PolynomialFeatures(degree=2)
                X_quad = quadratic.fit_transform(X)
                ransac = ransac.fit(X_quad, y)
                # ransac=ransac.fit(X, y)
                inlier_mask = ransac.inlier_mask_
                outlier_mask = np.logical_not(inlier_mask)
                inlier_indices = np.where(inlier_mask)[0]
                outlier_indices = np.where(outlier_mask)[0]
                inliers = sous_ensemble[inlier_indices]
                outlier = sous_ensemble[outlier_indices]
                liste_ajustee_RANSAC=liste_ajustee_RANSAC+[row for row in inliers]
                liste_supprimee_RANSAC=liste_supprimee_RANSAC+[row for row in outlier]
                # liste_supprimee_RANSAC.append(inliers)
                # Get fitted RANSAC curve
                X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
                y_quad_fit = ransac.predict(quadratic.fit_transform(X_fit))
                
                
                # inlier_mask = ransac.inlier_mask_
                # outlier_mask = np.logical_not(inlier_mask)
                
                # Get R2 value
                # quadratic_r2 = r2_score(y, ransac.predict(X_quad))
                
                # Plot inliers
                # inlier_mask = ransac.inlier_mask_
                # plt.scatter(X[inlier_mask], y[inlier_mask], c="blue", marker="o", label="Inliers")
                
                # Plot outliers
                # outlier_mask = np.logical_not(inlier_mask)
                # plt.scatter(
                #     X[outlier_mask], y[outlier_mask], c="lightgreen", marker="s", label="Outliers"
                # )
                
                # # Plot fitted RANSAC curve
                # plt.plot(
                #     X_fit,
                #     y_quad_fit,
                #     label="quadratic (d=2), $R^2=%.2f$" % quadratic_r2,
                #     color="red",
                #     lw=2,
                #     linestyle="-",
                # )
                
                # plt.xlabel("X")
                # plt.ylabel("Sea ice extent")
                # plt.legend(loc="upper left")
                # plt.tight_layout()
                # plt.show()
                # plt.close()
            else:
                liste_ajustee_RANSAC=liste_ajustee_RANSAC+[row for row in sous_ensemble]
        
        return liste_ajustee_RANSAC, liste_supprimee_RANSAC
    
    
    
    #DIVERS FONCTION SIMPLE
    # =================================================================================================
    def boundaries_images_reduites(self):
        if self.boundaries_proj is None:
            boundaries=np.array([
                [0,self.depthmap_ajustee.shape[1]],
                [0,self.depthmap_ajustee.shape[0]]
                ])
            
            #BAS DE LA GRANGE
            # boundaries=np.array([
            #     [1246,1390],
            #     [920,960]
            #     ])*fact
            # devant_abre
            # boundaries=np.array([
            #     [1017,1050],
            #     [1114,1150]
            #     ])*fact
        else:
            boundaries=self.boundaries_proj*self.fact_reduce_photomontage
        boundaries=boundaries.astype(int)
        
        return boundaries
    
    def plot_from_prof(self):
        list_prof=self.dict_prof_TO_liste(self.dict_prof)
        # array_prof=np.array(list_prof)
        plot.plot_from_liste_prof(list_prof, title="Depth Anything V2", equal=True)
            

def tri_array_par_rapport_a_une_colonne(array, colonne):
    liste_prof_array_tri = array[array[:, colonne].argsort()]
    return liste_prof_array_tri

def creer_liste_intervalles(intervalle, nbre_mesures):
    intervalles=[]
    for i in range(0, nbre_mesures-intervalle, intervalle):
        if i+2*intervalle>nbre_mesures:
            j=nbre_mesures-1

        else:
            j=i+intervalle-1
        intervalles.append([i,j])
        
    return intervalles
            

        
def return_mask_detection_donnee_aberrante(data_2d):
    data = np.array(data_2d, dtype=float)
    data_epurer=data_epurer_densite_point(data,0)
    # 1. Calcul de la moyenne des coordonnées
    mean = np.mean(data_epurer, axis=0)
   # 2. Calcul de la matrice de covariance
    cov_matrix = np.cov(data_epurer, rowvar=False)
    
    # 3. Calcul des valeurs propres et vecteurs propres de la matrice de covariance
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 4. Calcul des distances par rapport à l'ellipse
    threshold = chi2.ppf(0.99, df=2)
    outliers=[]
    for i, point in enumerate(data):
        if not is_inside_ellipse(point, mean, cov_matrix, threshold):
            outliers.append(i)
    mask_inside_ellipse = np.ones(len(data), dtype=bool)
    mask_inside_ellipse[outliers] = False
    data_inside_ellipse = data[mask_inside_ellipse]
    # Visualisation des données
    plt.plot(data[outliers, 0], data[outliers, 1], '.', alpha=0.3, color='red', label='Points aberrants')
    plt.plot(data_inside_ellipse[:, 0], data_inside_ellipse[:, 1], '.', alpha=0.3, color='blue', label='Points à l\'intérieur de l\'ellipse')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Détection des points aberrants')
    plt.legend()
    # plt.axis('equal')
    
    # Affichage de l'ellipse de confiance
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = np.sqrt(eigenvalues[0]*threshold) * np.cos(theta)
    ellipse_y = np.sqrt(eigenvalues[1]*threshold) * np.sin(theta)
    
    # Rotation et translation de l'ellipse
    ellipse = np.array([ellipse_x, ellipse_y]).T @ eigenvectors.T + mean
    plt.plot(ellipse[:, 0], ellipse[:, 1], label='Ellipse de confiance', color='green')
    
    plt.show()
    
    plt.close()
    
    return mask_inside_ellipse
        
def is_inside_ellipse(point, mean, cov_matrix, threshold):
    diff = point - mean
    distance_squared = diff.T @ np.linalg.inv(cov_matrix) @ diff
    return distance_squared <= threshold

def data_epurer_densite_point(data, index_depth=0, index_mono=1, nb_val=15):
    maxi=int(np.max(data[:, index_depth]))
    intervalle=(maxi)//10
    print(maxi)
    print(intervalle)
    if intervalle==0:
        intervalle=1
    array_modified=np.copy(data)
    intervalles=creer_liste_intervalles(intervalle, maxi+10)
    for inter in intervalles:
        
        indices = np.where((array_modified[:, index_depth] >= inter[0]) & (array_modified[:, index_depth] <= inter[1]+1))[0]
        
        if len(indices) > nb_val:
            nbre=len(indices)-nb_val
            ind_choice=np.random.choice(indices, size=nbre,  replace=False)
            array_modified = np.delete(array_modified, ind_choice, axis=0)
        elif len(indices)<3:
            array_modified = np.delete(array_modified, indices, axis=0)
    plt.plot(data[:,index_depth], data[:,index_mono], ".", alpha=0.3)
    plt.plot(array_modified[:,index_depth], array_modified[:,index_mono], ".", alpha=0.3)

    plt.show()
    plt.close()
    
    return array_modified
            
            
        
        
        
        
def array_to_2d(self,array, index1, index2):
    array_2d=array[:,[index1, index2]]
    
    return array_2d

def calcul_ellipse_2inc(inc, Kxx):
    """
    Calcul un ellipse en fonction de 2 pararmètre inconnnus et de la matrice de variance-covariance

    Parameters
    ----------
    inc : np.array shape(1,2)
        Coordonnée de points de l'ellipse.
    Kxx : np.array
        Matrice de variance covariance.

    Returns
    -------
    ellipse : Ellipse matplotlib
        Ellipse.
    longueur : float
        Rayon max de l'ellipse.
    largeur : float
        Rayon min de l'ellipse.
    angle : float
        Angle de l'ellipse en degré.

    """
    # Matrice de variance-covariance
    cov_matrix = Kxx
    x=inc[0,0]
    y=inc[1,0]
    # Calcul des valeurs propres et vecteurs propres
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    
    # Calcul des axes de l'ellipse (rayons)
    axis_length = np.sqrt(eigvals)
    
    longueur=axis_length[0]
    largeur=axis_length[1]
    
    
    # Calcul de l'orientation de l'ellipse (en degrés)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    
    # Création de l'ellipse
    ellipse = Ellipse(xy=(x, y), width=2*axis_length[0], height=2*axis_length[1], angle=angle, edgecolor='r', fc='none')
    
    return ellipse, longueur, largeur, angle

def calcul_fct_lineaire(x1,x2):
    a=(x2[1]-x1[1])/(x2[0]-x1[0])
    
    b=x1[1]-a*x1[0]
    
    return a, b
@jit(nopython=True)
def gauss_markov2(profondeur_valeur,sigma0=1, robuste=False, iter_robuste=10, delta=2.5, print_res=False):
    nb_l=profondeur_valeur.shape[0]
    B = np.array(profondeur_valeur[:,2][:, np.newaxis], dtype=np.float64)
    X=np.array(profondeur_valeur[:,4][:, np.newaxis], dtype=np.float64)

    A=np.ones((nb_l,2))
    A[:,0]=X[:,0]
    Qll=np.eye(nb_l)*0.1

    
    nb_i=A.shape[1]
    nb_l=A.shape[0]
    
    P=np.linalg.inv(Qll)
    Qxx=np.linalg.inv(A.T@P@A)
    inc=Qxx@A.T@P@B
    vi=A@inc-B+0.00000001
    vi = np.asarray(vi, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    vitpvi=(vi.T@P@vi)
    s0=np.sqrt((vitpvi)/(nb_l-nb_i))
    Quot=s0/sigma0
    
    if robuste and Quot>1.2 and nb_l>15: 
        prev_inc=inc.copy()
        for iteration in range(iter_robuste):
            # perte = fonction_de_perte_huber(vi, delta)
            perte=np.where(np.abs(vi) <= delta,
                            0.5 * vi**2,
                            delta * (np.abs(vi) - 0.5 * delta))
            W = np.diag(perte.flatten())
            Qxx_rob= np.linalg.inv(A.T @ W @ A)
            inc = Qxx_rob @ A.T @ W @ B
            v = B - A @ inc
            if np.linalg.norm(inc - prev_inc) < 0.00001:
                # print(f"Convergence atteinte après {iteration + 1} itérations.")
                break
            prev_inc=inc.copy()
    
    vi=A@inc-B
    Qvv=Qll-A@Qxx@A.T
    wi=np.zeros((nb_l, 1))
    for i in range(nb_l):
        wi[i,0]=vi[i,0]/(sigma0*np.sqrt(Qvv[i,i]))
    Quot=s0/sigma0
    B_calc=A@inc
    Kxx=s0**2*Qxx
    if print_res:
        print(f"Le calcul a convergé avec un quotion de {Quot} sur {nb_l} mesures et un résidu maximum de {np.max(np.abs(vi))} et un wi max de {np.max(np.abs(wi))}")

    return inc, vi, wi, B_calc,s0, Quot, Kxx

@jit(nopython=True)
def gauss_markov(Qll, A, B,sigma0=1, robuste=False, iter_robuste=10, delta=2.5, print_res=False):
    
    nb_i=A.shape[1]
    nb_l=A.shape[0]
    
    P=np.linalg.inv(Qll)
    Qxx=np.linalg.inv(A.T@P@A)
    inc=Qxx@A.T@P@B
    vi=A@inc-B+0.00000001
    vi = np.asarray(vi, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    vitpvi=(vi.T@P@vi)
    s0=np.sqrt((vitpvi)/(nb_l-nb_i))
    Quot=s0/sigma0
    
    if robuste and Quot>1.2 and nb_l>15: 
        prev_inc=inc.copy()
        for iteration in range(iter_robuste):
            # perte = fonction_de_perte_huber(vi, delta)
            perte=np.where(np.abs(vi) <= delta,
                            0.5 * vi**2,
                            delta * (np.abs(vi) - 0.5 * delta))
            W = np.diag(perte.flatten())
            Qxx_rob= np.linalg.inv(A.T @ W @ A)
            inc = Qxx_rob @ A.T @ W @ B
            v = B - A @ inc
            if np.linalg.norm(inc - prev_inc) < 0.00001:
                # print(f"Convergence atteinte après {iteration + 1} itérations.")
                break
            prev_inc=inc.copy()
    
    vi=A@inc-B
    Qvv=Qll-A@Qxx@A.T
    wi=np.zeros((nb_l, 1))
    for i in range(nb_l):
        wi[i,0]=vi[i,0]/(sigma0*np.sqrt(Qvv[i,i]))
    Quot=s0/sigma0
    B_calc=A@inc
    Kxx=s0**2*Qxx
    if print_res:
        print(f"Le calcul a convergé avec un quotion de {Quot} sur {nb_l} mesures et un résidu maximum de {np.max(np.abs(vi))} et un wi max de {np.max(np.abs(wi))}")

    return inc, vi, wi, B_calc,s0, Quot, Kxx
@jit(nopython=True)
def fonction_de_perte_huber(residuals, delta):
    """Calcule la fonction de perte de Huber pour les résidus."""
    return np.where(np.abs(residuals) <= delta,
                    0.5 * residuals**2,
                    delta * (np.abs(residuals) - 0.5 * delta))
