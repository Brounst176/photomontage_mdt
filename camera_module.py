# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:20:04 2025

@author: Bruno
"""
import pointcloud_module as pcd
from scipy.spatial.transform import Rotation as Rot
import trimesh
import laspy
import matplotlib.pyplot as plt
from matplotlib.path import Path
import math
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
import open3d as o3d;
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
import cvxpy as cp
import alphashape 
import numpy as np
import plot_module as plot
from shapely.geometry import Point

class camera:
    def __init__(self, name="", images_pathfolder="", w=0, h=0, cx=0, cy=0, f=0, k1=0, k2=0, k3=0, k4=0, p1=0, p2=0, b1=0, b2=0):
        """
        Initialiser les paramètres de la caméra avec des valeurs par défaut (0).
        
        :param cx: Coordonnée X du centre de l'image (focus en pixels)
        :param cy: Coordonnée Y du centre de l'image (focus en pixels)
        :param f: Distance focale (en pixels)
        :param k1, k2, k3, k4: Coefficients de distorsion radiale
        :param p1, p2: Coefficients de distorsion tangentielles
        :param b1, b2: Paramètres supplémentaires (ex : paramètres de projection stéréographique)
        :param name: Nom de la caméra
        :param w: largeur pixel de l'image
        :param h: hauteur pixel de l'image
        """
        self.cx = cx
        self.cy = cy
        self.f = f
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.p1 = p1
        self.p2 = p2
        self.b1 = b1
        self.b2 = b2
        self.name =name
        self.images_pathfolder = images_pathfolder
        self.w = w
        self.h = h
        self.images = {}
        self.debug=""

    def __repr__(self):
        """
        Représentation textuelle de l'objet Camera.
        """
        return (f"Camera(cx={self.cx}, cy={self.cy}, f={self.f}, k1={self.k1}, k2={self.k2}, "
                f"k3={self.k3}, k4={self.k4}, p1={self.p1}, p2={self.p2}, b1={self.b1}, b2={self.b2})")


    def distorsion_frame_brown_agisoft(self, m_cam):
        """
        Applique la distorsion de Brown à un point m_cam (X, Y, Z) dans le système caméra en utilisant les coefficients init.
        Input dans le système caméra
        :param x: Coordonnée en X
        :param y: Coordonnée en Y
        :param z: Coordonnée en Z 
        :return: Coordonnées (u, v) de l'image
        """
        X=-m_cam[0]
        Y=m_cam[1]
        Z=self.f
        x=X/Z

        y=Y/Z
        uv=self.distorsion_frame_brown_agisoft_from_xy(x,y)

        return uv
    def distorsion_frame_brown_agisoft_from_xy(self, x, y):

        r=(x**2+y**2)**0.5
        x_prime = x*(1+self.k1*r**2+self.k2*r**4+self.k3*r**6+self.k4*r**8)+(self.p1*(r**2+2*x**2)+2*self.p2*x*y)
        y_prime = y*(1+self.k1*r**2+self.k2*r**4+self.k3*r**6+self.k4*r**8)+(self.p1*(r**2+2*y**2)+2*self.p2*x*y)
        u=self.w*0.5+self.cx+x_prime*self.f+x_prime*self.b1+y_prime*self.b2
        v=self.h*0.5+self.cy+y_prime*self.f
        uv=np.array([u,v])
        return uv
    
    def set_camera_calib(self, name="", w=0, h=0, cx=0, cy=0, f=0, k1=0, k2=0, k3=0, k4=0, p1=0, p2=0, b1=0, b2=0 ):
        self.cx = cx
        self.cy = cy
        self.f = f
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.p1 = p1
        self.p2 = p2
        self.b1 = b1
        self.b2 = b2
        self.name =name
        self.w = w
        self.h = h
        
    def ajout_photo(self, name, R, S):
        """
        Ajoute une photo
        
        :param name: nom de la photo
        :param R: Matrice de rotation (r11, r12, ...)
        :param S: Matrice des coordonnées de la caméra (X, Y, Z) dans le système globale
        """
        self.images[name]={"R":R, "S":S}
    def vect_dir_camera(self,photoname):
        R=self.images[photoname]["R"]
        N=np.dot(R.T,np.array([0,0,-1]))
        return N
        
    def point_terrain_to_point_camera(self, photoname, M):
        S=self.images[photoname]["S"]
        R=self.images[photoname]["R"]

        V=M-S

        F=np.array([0,0,-self.f])

        rms=np.dot(R,V)

        m_cam=F-F[2]*rms/rms[2]
        return m_cam
        
    def M_to_uv(self, photoname, M):
        m_cam=self.point_terrain_to_point_camera(photoname, M)
        uv=self.distorsion_frame_brown_agisoft(m_cam)
        return uv
    
    def uv_to_M(self, photoname, uv, distance):
        m_prime=self.uv_to_m_prime(uv)
        S=self.images[photoname]["S"]
        R=self.images[photoname]["R"]
        
        
        x=m_prime[0]
        y=m_prime[1]
        for i in range(100):
            uv_prov=self.distorsion_frame_brown_agisoft_from_xy(x, y)
            diff_uv=uv-uv_prov

            if abs(diff_uv[0])>0.001 or abs(diff_uv[1])>0.001:
                x+=diff_uv[0]/10000
                y+=diff_uv[1]/10000
            else:
                break
        X=-self.f*x
        Y=self.f*y
        F=np.array([0,0,-self.f])
        m=np.array([X,Y,0])

        M_prime=np.dot(R.T,(m-F))

        fact=distance/np.linalg.norm(M_prime)

        M=S-fact*M_prime
        return M
    
    def calcul_proj_cam(self, photoname, M):
        """
        Cette fonction calcule le distance de projection d'un point et le coordonnée projetée'

        Parameters
        ----------
        photoname : string
            Photoname de la camera.
        M : np.array
            Coordonnée d'un point dans l'espace.

        Returns
        -------
        d_proj : (int) distance projetée.
        H : (np.array) coodronnée du point projeté

        """
        S=self.images[photoname]["S"]
        R=self.images[photoname]["R"]
        P=M
        PS=P-S
        
        N=self.vect_dir_camera(photoname)
        norm_N=np.linalg.norm(N)
        norm_PS=np.linalg.norm(PS)
        H=P-(PS@N)/(norm_PS*norm_N)*norm_PS/norm_N*N
        norm_PH=np.abs((PS@N)/(norm_PS*norm_N)*norm_PS)
        return H, norm_PH
        
        
    def uv_to_M_by_dist_prof(self, photoname, uv, dist_proj):
        m_prime=self.uv_to_m_prime(uv)
        S=self.images[photoname]["S"]
        R=self.images[photoname]["R"]
        
        
        x=m_prime[0]
        y=m_prime[1]
        for i in range(100):
            uv_prov=self.distorsion_frame_brown_agisoft_from_xy(x, y)
            diff_uv=uv-uv_prov

            if abs(diff_uv[0])>0.001 or abs(diff_uv[1])>0.001:
                x+=diff_uv[0]/10000
                y+=diff_uv[1]/10000
            else:
                break
        X=-self.f*x
        Y=self.f*y
        F=np.array([0,0,-self.f])
        m=np.array([X,Y,0])

        M_prime=np.dot(R.T,(m-F))
        # d_M_prime=np.linalg.norm(M_prime)
        # d_h=M_prime[2]
        # N=self.vect_dir_camera(photoname)
        
        fact=dist_proj/self.f
        M=S-fact*M_prime
        return M
        
    def uv_to_m_prime(self, uv):
        mx=(uv[0]-self.w/2-self.cx)/(self.f+self.b1+self.b2)
        my=(uv[1]-self.h/2-self.cy)/self.f
        return np.array([mx,my, 0])
        
    def rot_zyx(self, o: float, p: float, k: float, degrees:bool = False):
        """
        Matrice de rotation à partir de la séquence angulaire opk, suivant la convention zyx (celle utilisée par Agisoft)
        :param o:
        :param p:
        :param k:
        :param degrees
        :return:
        """
        r = Rot.from_euler('zyx', [k, p, o], degrees=degrees)
        # return rot_z(k)@rot_y(p)@rot_x(o)
        return r.as_matrix().T
    
    def import_image_from_omega_phi_kappa_file(self, filepath):
        with open(filepath) as file:
            for line in file:
                I=line.strip().split("\t")
                if len(I)>1:
                     R=np.array([
                         [float(I[7]),float(I[8]),float(I[9])],
                         [float(I[10]),float(I[11]),float(I[12])],
                         [float(I[13]),float(I[14]),float(I[15])],
                         ])
                     S=np.array([
                         float(I[1]),
                         float(I[2]),
                         float(I[3])
                         ])
                     name=I[0]
                     self.ajout_photo(name, R, S)

    def mesh_pyramide_camera_create(self,photoname, distance):
        x1=self.uv_to_M(photoname,np.array([0,0]),distance)
        x2=self.uv_to_M(photoname,np.array([0,self.h]),distance)
        x3=self.uv_to_M(photoname,np.array([self.w,self.h]),distance)
        x4=self.uv_to_M(photoname,np.array([self.w,0]),distance)
        
        base_points = np.array([
            x1,  # Point 1 de la base
            x2,  # Point 2 de la base
            x3,  # Point 3 de la base
            x4   # Point 4 de la base
        ])

        sommet = self.images[photoname]["S"]  # Sommet de la pyramide

        # Créer une liste des faces du maillage
        faces = np.array([
            [0, 1, 2],  
            [0, 2, 3], 
            [0, 3, 4],  
            [0, 1, 4],  
            [1, 2, 4],  
            [2, 3, 4]  
        ])
        
        # Créer le maillage de la pyramide
        vertices = np.vstack([base_points, sommet])
        maillage = trimesh.Trimesh(vertices=vertices, faces=faces)
        maillage.export(photoname+'.stl')
        return maillage

    
    def pointcloudcamera_from_pathlas(self, pathlas, photoname):
        point_cloud, normales=pcd.readlas_to_numpy(pathlas)
        o3d_pcd=pcd.array_TO_o3d(point_cloud, normales)
        o3d_pcd=pcd.remove_statistical_outlier_pcd(o3d_pcd)
        point_cloud, normales=pcd.o3d_TO_array(o3d_pcd)
        mesh=self.mesh_pyramide_camera_create(photoname, 500)
        
        points_inside, normal_inside=pcd.points_inside_mesh(point_cloud, normales, mesh)
        
        return points_inside, normal_inside
        
    def liste_coord_image_pt_homologue(self, pathlas, depthmap_IA, photoname, fact_reduce=0.1, calc_normal=False):
        """
        Créer un image avec les valeurs de profondeur depuis un nuage de points las
    
        pathlas: chemin du nuage de points las
        photoname: nom de la photo à tester
        fact_reduce: facteur de réduction de l'image
        
        Retourne une image array et une liste des valeurs de profondeurs
        """
        
        R=self.images[photoname]["R"]
        vect_normal_photo=self.vect_dir_camera(photoname)
        verticale=np.array([0,0,1])
        
        
        
        point_cloud, normales=pcd.readlas_to_numpy(pathlas)
        
        
        o3d_pcd=pcd.array_TO_o3d(point_cloud, normales)
        # o3d_pcd=pcd.remove_statistical_outlier_pcd(o3d_pcd)
        point_cloud, normales=pcd.o3d_TO_array(o3d_pcd)
        mesh=self.mesh_pyramide_camera_create(photoname, 500)
        
        points_inside, normal_inside=pcd.points_inside_mesh(point_cloud, normales, mesh)
        depth_array=np.array(depthmap_IA)
        
        # pcd.view_point_cloud_from_array(points_inside, normal_inside)
        if calc_normal:
            normal_inside=pcd.calcul_normal_point_cloud(points_inside, normal_inside)
        # points_inside_reduce=pcd.reduction_nuage_nbre(points_inside)
        normal_inside=np.array([normal_inside[i]/(np.linalg.norm(normal_inside[i])*2) for i in range(normal_inside.shape[0])])
        o3d_pcd=pcd.array_TO_o3d(points_inside, normal_inside)
        # pcd.view_point_cloud_from_array(points_inside, normal_calc)
        
        
        voxel=3
        #VOXEL DU NUAGE DE POINTS POUR REDUCTION
        for i in range(10):
            points_inside_reduce=pcd.reduction_nuage_voxel(o3d_pcd,voxel)
            # print(points_inside_reduce.points)
            points=np.asarray(points_inside_reduce.points)
            if 4000>points.shape[0]>3000:
                break
            elif points.shape[0]<3000:
                voxel-=0.3
            else:
                voxel+=0.3
            
        point_cloud, normales=pcd.o3d_TO_array(points_inside_reduce)
        
        
        
        
        print(f"Nbre de points après réduction: {point_cloud.shape[0]}")
        # pcd.save_point_cloud_las(point_cloud, "point_camera")
        
        point_cloud_epurer=[]
        point_supprimer=[]
        
        normale_epurer=[]
        normale_supprimer=[]
        
        
        #Nettoyage du nuage de points en fonction des normales des points
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
        dict_uv_profondeur={}
        
        #CREATION DE LA DICT DES UV ET DES POINTS
        img_uv_prof=np.zeros((int(self.h*fact_reduce), int(self.w*fact_reduce)))
        for i in range(len(point_cloud_epurer)):
            point=point_cloud_epurer[i]
            normale=normale_epurer[i]

            d=self.dist_MS(point, photoname)
            P_proj, d_proj=self.calcul_proj_cam(photoname, point)
            uv=self.M_to_uv(photoname, point)
            u=math.floor(uv[0]*fact_reduce)
            v=math.floor(uv[1]*fact_reduce)
            
            if u<int(self.w*fact_reduce) and v<int(self.h*fact_reduce):
                depthIA=depth_array[v,u]
                if depthIA!=0:
                    if u in dict_uv_profondeur.keys():
                        if v in dict_uv_profondeur[u].keys():
                            if d<dict_uv_profondeur[u][v]["d"]:
                                dict_uv_profondeur[u][v]["d"]=d
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
                
        return dict_uv_profondeur
            
    def dist_MS(self, M, photoname):
        return np.linalg.norm(self.images[photoname]["S"]-M)
    
    # def read_profondeur_depthanything(self, pathDepth):
    #     depthmap=Image.open(pathDepth)
    #     return depthmap

    
    def create_depth_FROM_fonction_poly(self, depthmap_IA, param_poly, photoname):
        depth_array=np.array(depthmap_IA)
        depth_traitee=np.zeros(depth_array.shape)
        nb_poly=param_poly.shape[0]
        points=[]
        
        for i in range(depth_array.shape[0]):
            for j in range(depth_array.shape[1]):
                depth_IA_value=depth_array[i,j]
                
                    
                dist=0
                if depth_IA_value==0:
                    dist=np.nan
                else:
                    for n in range(nb_poly):
                        nb=nb_poly-1-n
                        dist+=param_poly[n,0]*depth_IA_value**nb
                    uv=np.array([j*10,i*10])
                    points.append(self.uv_to_M_by_dist_prof(photoname, uv, dist))
                depth_traitee[i,j]=dist
            
                        
                
        image=Image.fromarray(depth_traitee)
        image.save("test.tif")
        pcd.save_point_cloud_las(np.array(points), "qq.las")
        return depth_traitee, np.array(points)
                
        # for i in range(shape)






# DEPTHMAP CLASS
# ===================================================================================================

class depthmap:
    def __init__(self, pathdepthmap_IA, photoname, pathlas, calc_normal, camera,  max_megapixel=1.5,  IA_modele="depthanythingV2", fact_reduce_IA=0.5, fact_reduce_depthajuste=0.1):
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
        IA_modele : string, optional
            Nom de la depthmap IA utilisée. The default is "depthanythingV2".
        fact_reduce : float, optional
            Facteur de reduction de l'image. The default is 0.1.
        """
        self.depthmap_IA=self.create_depthmap(pathdepthmap_IA)
        self.max_depthmap=np.max(self.depthmap_IA)
        self.IA_modele=IA_modele
        self.photoname=photoname
        self.camera=camera
        self.fact_reduce_IA=fact_reduce_IA
        self.fact_reduce_depthajuste=fact_reduce_depthajuste
        self.depthmap_ajustee=self.initialisaiton_depthmap()
        self.depthmap_clusters=self.initialisaiton_depthmap()
        self.dict_prof=camera.liste_coord_image_pt_homologue(pathlas, self.depthmap_IA, photoname, fact_reduce_IA, calc_normal)
        self.liste_groupe_resultat=[]
        self.clusters=[]
        self.param_transfo_cluster={}
        self.grille_calcul=[]
        self.debug=""
    def create_depthmap(self, pathDepth):
        depthmap=Image.open(pathDepth)
        return np.array(depthmap)
    
    def fusion_depthpro_depthanything(self, depthanything, depthpro=None):
        for i in range(depthpro.shape[0]):
            for j in range(depthpro.shape[1]):
                if depthanything[i,j]==0:
                    depthpro[i,j]=0
                    
        return depthpro
    
    def initialisaiton_depthmap(self):
        shape0=int(self.depthmap_IA.shape[0]*(self.fact_reduce_depthajuste/self.fact_reduce_IA))
        shape1=int(self.depthmap_IA.shape[1]*(self.fact_reduce_depthajuste/self.fact_reduce_IA))
        depth_init=np.ones((shape0, shape1))*-1
        return depth_init
    
    
    def dict_prof_TO_liste(self, dict_prof):
        liste_prof=[]
        for u, value in dict_prof.items():
            for v, value2 in value.items():
                prof_uv=np.array([u, v, value2["d"], value2["sigma"], value2["depthIA"], value2["point"], value2["normal"]], dtype=object)
                liste_prof.append(prof_uv)
                
        return liste_prof
        
    def transformation_simple_depthmap_IA(self):
        max_dethaIA=int(np.max(self.depthmap_IA))
        inter=max_dethaIA//5
        
        list_prof=self.dict_prof_TO_liste(self.dict_prof)
        array_prof=np.array(list_prof)
        # Trier le tableau par la 5e colonne (colonne indexée à 4)
        sorted_array = array_prof[array_prof[:, 4].argsort()[::-1]]
        sorted_list=sorted_array.tolist()
        Quot=0
        liste=[]
        index=1
        # liste_transfo=[]
        # inter = creer_liste_intervalles(50, sorted_array.shape[0])
        # while len(inter)>0:
        #     param, i_supp=self.calcul_iteratif_sur_quot(inter, sorted_list)
        #     liste_transfo.append(param)
        #     for i in range(i_supp):
        #         del inter[0]
        A, B,Qll, inc, wi, vi, X, B_calc ,s0 , Quot=self.tranformation_depthanything_gauss(sorted_list, nbre_inc=3)
        
        plot.plot_mesure_calcule(X, B, B_calc, "2D")
    
    def creer_grille_point(self):
        width=self.depthmap_ajustee.shape[1]
        height=self.depthmap_ajustee.shape[0]
        intervalle_u=creer_liste_intervalles(10, width)
        intervalle_v=creer_liste_intervalles(10, height)
        liste_points_grille=[]
        clusters_dict = self.param_transfo_cluster
        fact=self.fact_reduce_depthajuste/self.fact_reduce_IA
        for u in intervalle_u:
            for v in intervalle_v:
                depthajuste_value=self.depthmap_ajustee[v[1], u[1]]
                depthIA=self.depthmap_IA[int(v[1]/fact), int(u[1]/fact)]
                if depthajuste_value>0 and depthajuste_value!=9999:
                    point=np.array([u[1], v[1],depthajuste_value, depthIA])
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
                point=np.array([coords[0]*fact, coords[1]*fact,depthajuste_value, depthIA])
                liste_points_grille.append(point)
                
                
                
            
        array_grillee=np.array(liste_points_grille)
        plt.plot(array_grillee[:,0],-array_grillee[:,1], ".")
        plt.show()
        plt.close()
        
        self.grille_calcul=liste_points_grille
                    
            
    def calcul_iteratif_sur_quot(self, liste_intervalle, sorted_list):
        liste=[]
        i=0
        for inter in liste_intervalle:
            liste_inter=sorted_list[inter[0]: inter[1]]
            liste_prov=liste+liste_inter
            A, B,Qll, inc, wi, vi, X, B_calc ,s0 , Quot=self.tranformation_depthanything_gauss(liste_prov, nbre_inc=3)
            if Quot<1:
                i+=1
                inc_prec=inc.copy()
                liste=liste_prov
                max_liste=np.max(np.array(liste)[:,4])
                min_liste=np.min(np.array(liste)[:,4])
                Quot_prec=Quot
            else:
                if i==0:
                    i=1
                    liste=liste_prov
                    inc_prec=inc.copy()
                    max_liste=np.max(np.array(liste)[:,4])
                    min_liste=np.min(np.array(liste)[:,4])
                    Quot_prec=Quot
                return [inc_prec, Quot_prec, max_liste, min_liste], i
            
    
    
    def creation_des_clusters(self, array_prof, view=False, first=False):
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
            


        # mask_noise=y_pred==-1 
        # colors = plt.cm.get_cmap("tab10", len(unique_labels))
        # point_colors = np.array([colors(label)[:3] for label in y_pred]) 
        # x_debruite=x[~mask_noise]
        # pcd.view_point_cloud_from_array(np.delete(x_debruite, 3, axis=1), color=point_colors[~mask_noise])
    
        
        for i in range(len(unique_labels)):
            if unique_labels[i]>-1:
                mask_noise=y_pred==unique_labels[i]
                array_cluster=array_prof[mask_noise]
                if array_cluster.shape[0]>5:
                    
                    #RAJOUTER LE RANSAC EST SI OUTLIER RANSAC PLUS GRAND QUE X FAIRE UN DBSCAN DU GROUPE SUR LES DISTANCE PROJ ET LES VALEURS DEPHTMAPS
                    inliers, outlier=self.ransac_simple(array_cluster.tolist())
                    # plot.plot_from_liste_prof(outlier.tolist())
                    
                    # print(f"Nbre de valeur en outlier {len(outlier)}")
                    if len(outlier)>50:
                        cluster=self.creation_des_clusters(array_cluster)
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
        self.debug=np.array(points_cluster_uv)

        pcd.view_point_cloud_from_array(np.array(points_cluster_uv))
        # pcd.save_point_cloud_las(np.array(points_cluster_uv), "cluster.las")
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
        inc, vi, wi, B_calc,s0, Quot=self.gauss_markov(Qll, A, B, robuste=False, print_res=True)

        inc  = self.optimisation_quadratique(Qll,A, B)
        
        
        points_uv=[]
        # if Quot<2.5:
        if inc is not None:
            self.param_transfo_cluster[str(cluster_index)]={"inc":inc, "quot": Quot, "enveloppe" : alpha_shape, "bounding_box": bounding_box_depthmap}
            fact=self.fact_reduce_depthajuste/self.fact_reduce_IA
            bounding_box=np.array([[np.min(cluster_array[:,0]), np.max(cluster_array[:,1])],[np.max(cluster_array[:,0]), np.min(cluster_array[:,1])]])
            bounding_box=bounding_box*fact
            bounding_box=np.floor(bounding_box).astype(int)

            for u in range(bounding_box[0][0], bounding_box[1][0]+1):
                # print(i)
                for v in range(bounding_box[1][1], bounding_box[0][1]+1):
                    
                    ij=np.array([u/fact, v/fact])
                    uv=np.array([u/self.fact_reduce_depthajuste, v/self.fact_reduce_depthajuste])
                    
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
                                points_uv.append(self.camera.uv_to_M_by_dist_prof(self.photoname, uv, dist))
                                
                                self.depthmap_clusters[v,u]=cluster_index
                                self.depthmap_ajustee[v,u]=dist
        
        
        # plot.plot_mesure_calcule(X, B, B_calc, "avec RANSAC", outlier[])
        # pcd.view_point_cloud_from_array(np.array(points_uv))
        # pcd.save_point_cloud_las(np.array(points_uv), "cluster6.las")
        return points_uv

    
    
    def gauss_markov(self,Qll, A, B,sigma0=1, robuste=False, delta=2.5, print_res=False):
        
        nb_i=A.shape[1]
        nb_l=A.shape[0]
        
        P=np.linalg.inv(Qll)
        Qxx=np.linalg.inv(A.T@P@A)
        inc=Qxx@A.T@P@B
        vi=A@inc-B
        # s0=1
        s0=np.sqrt((vi.T@P@vi)/(nb_l-nb_i))
        
        
        if robuste:
            prev_inc=inc.copy()
            for iteration in range(10):
                perte = self.fonction_de_perte_huber(vi, delta)
                W = np.diag(perte.flatten())
                Qxx_rob= np.linalg.inv(A.T @ W @ A)
                inc = Qxx_rob @ A.T @ W @ B
                v = B - A @ inc
                if np.linalg.norm(inc - prev_inc) < 0.1:
                    print(f"Convergence atteinte après {iteration + 1} itérations.")
                    break
                prev_inc=inc.copy()
        
        vi=A@inc-B
        Qvv=Qll-A@Qxx@A.T
        wi=np.zeros((nb_l, 1))
        for i in range(nb_l):
            wi[i,0]=vi[i,0]/(sigma0*np.sqrt(Qvv[i,i]))
        Quot=s0/sigma0
        B_calc=A@inc
        if print_res:
            print(f"Le calcul a convergé avec un quotion de {Quot} sur {nb_l} mesures et un résidu maximum de {np.max(np.abs(vi))} et un wi max de {np.max(np.abs(wi))}")
       
        
        
        return inc, vi, wi, B_calc,s0, Quot

    def optimisation_quadratique(self, Qll, A, B):
        
        
        nb_i=A.shape[1]
        nb_l=A.shape[0]
        x = cp.Variable((nb_i+nb_l,1))
        V=np.zeros((nb_l, nb_i+nb_l))
        X=np.zeros((nb_i, nb_i+nb_l))
        for i in range(nb_l):
            V[i, nb_i+i]=1.0
        for i in range(nb_i):
            X[i,i]=1.0
            
        F=2*V.T @V
        f=np.zeros((nb_i+nb_l,1))
        H = A@X-V
        h = B
        G=np.zeros((1, nb_i+nb_l))
        G[0,1]=1
        g=np.zeros((nb_i+nb_l,1))
        g[1,0]=-0.00000000000001

        
        objective = cp.Minimize(1/2*cp.quad_form(x,F) + f.T @ x)

        constraints = [H@x == h, G@x<=g]
        #resolution du système
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
            print("\nLa valeur optimale est : ", prob.value)
            print("Le statut de la solution est:", prob.status)
            print("La solution x est:")
            # print(x.value)
            return x.value
        except:
            print("Le statut de la solution est:", prob.status)
            return None
            
        
        

    def fonction_de_perte_huber(self,residuals, delta):
        """Calcule la fonction de perte de Huber pour les résidus."""
        return np.where(np.abs(residuals) <= delta,
                        0.5 * residuals**2,
                        delta * (np.abs(residuals) - 0.5 * delta))
    
    
    def calcul_dist_ajustee(self):
        sigma0=1
        grille=self.grille_calcul
        # grilless=self.dict_prof_TO_liste(self.dict_prof)
        grille_array=np.array(grille)
        nb_l=len(grille)
        fact=self.fact_reduce_depthajuste/self.fact_reduce_IA
        B=grille_array[:,2][:, np.newaxis]
        X=grille_array[:,3][:, np.newaxis]
        u_liste=grille_array[:,0][:, np.newaxis]
        
        u_liste=u_liste.astype(int)
        v_liste=grille_array[:,1][:, np.newaxis]
        v_liste=v_liste.astype(int)
        uv_grille=np.hstack((u_liste, v_liste))
        print(uv_grille)
        A=np.ones((nb_l,3))
        
        A[:,0]=u_liste[:,0]
        A[:,1]=X[:,0]
        self.debug= X
        
        for v in range(self.depthmap_ajustee.shape[0]):
            for u in range(self.depthmap_ajustee.shape[1]):
                u_IA=int(u/fact)
                v_IA=int(v/fact)
                uv_array=np.array([
                    [u, v],
                    ])
                depth_IA_value=self.depthmap_IA[v_IA, u_IA]
                depthmap_ajustee_value= self.depthmap_ajustee[v, u]
                if depthmap_ajustee_value<=0:
                    
                    if depth_IA_value==0:
                        value_calc=9999
                    else:
                        Kll=np.eye(nb_l)
                        
                        #calcul précision selon la distance à l'UV
                        diff_uv=uv_grille-uv_array
                        dist_grille_uv=diff_uv@diff_uv.T
                        dist_grille_uv = np.diagonal(dist_grille_uv)
                        dist_grille_uv = dist_grille_uv[:, np.newaxis]**0.5
                        dist_grille_uv=np.abs(u_liste-u)
                        y_uv=(1.6**dist_grille_uv**0.52)/200+0.1
                        
                        # y_uv=0
                
                        
                        
                        diff_depthvalue=np.abs(X-depth_IA_value)
                        
                        # y_depth=(1.65**diff_depthvalue**0.63)/200+0.1
                        y_depth=(1.5**diff_depthvalue**0.8)/200+0.1
                        # y_depth=1
                        
                        Qll=Kll*(y_depth+y_uv+0.00001)
                        

                        inc, vi, wi, B_calc,s0, Quot=self.gauss_markov(Qll, A, B)
                        
                        depth_calc=inc[0,0]*u+inc[1,0]*depth_IA_value+inc[2,0]
                        self.depthmap_ajustee[v, u]=depth_calc
                        
                        
                        
                # self.calcul_dist_ajust_from_uv(uv_array, 300,400)
                if v%50==0 and u%50==0:
                    print(f"Les éléments ont été calculés jusqu' à l'uv {u} {v}")
        self.debug= Qll
        print("Les depthmap ajustées est entièrement calculée")
    
    def calcul_dist_ajust_from_uv(self, uv_array, IA_normalisation, Dist_normalisation):
        fact=self.fact_reduce_depthajuste/self.fact_reduce_IA
        
        depth_IA_value=self.depthmap_IA[v_IA, u_IA]
        # print(depth_IA_value)
        depthmap_ajustee_value= self.depthmap_ajustee[uv_array[1], uv_array[0]]
        # print(depthmap_ajustee_value)
        if depthmap_ajustee_value<=0:
            
            if depth_IA_value==0:
                value_calc=9999
            else:
                list_dist_cluster=[]
                
                    
                poids=1/(dist_cluster/Dist_normalisation+abs(IA_cluster-depth_IA_value)/IA_normalisation)
                # print(poids)
                # print(value_cluster)
                somme_value+=value_cluster*poids
                somme_poids+=poids
                        
                value_calc=somme_value/somme_poids
                # print(value_calc)
            
            self.depthmap_ajustee[uv_array[1], uv_array[0]]=value_calc
            
            
            
    
    
    
    
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
    
    
    
    def ransac_simple(self, liste_prof):
        
        min_samples=int(len(liste_prof)**0.1)
        ransac = RANSACRegressor(
            LinearRegression(),
            max_trials=100,
            min_samples=3,
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
                
                uv=np.array([j/self.fact_reduce_depthajuste,i/self.fact_reduce_depthajuste])
                d_proj=self.depthmap_ajustee[i,j]
                
                if d_proj>0 and d_proj<1000:
                    points.append(self.camera.uv_to_M_by_dist_prof(self.photoname, uv, d_proj))

        pcd.save_point_cloud_las(np.array(points), pointcloud_name+".las")
        
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
        print(liste_prof)
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
                
                
        inc, vi, wi, B_calc,s0, Quot=self.gauss_markov(Qll, A, B, robuste=False)

            
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

def tri_array_par_rapport_a_une_colonne(self, array, colonne):
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
            
        
                
        
        
        