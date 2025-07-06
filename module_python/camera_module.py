# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:20:04 2025

@author: Bruno
"""

from scipy.spatial.transform import Rotation as Rot
import trimesh

from PIL import Image 
import copy 


import numpy as np

import xml.etree.ElementTree as ET

import module_python.pointcloud_module as pcd


class camera:
    def __init__(self, images_pathfolder=""):
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
        self.calibrations = {}
        self.cx = 0
        self.cy = 0
        self.f = 0
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.k4 = 0
        self.p1 = 0
        self.p2 = 0
        self.b1 = 0
        self.b2 = 0
        self.images_pathfolder = images_pathfolder
        self.w = 0
        self.h = 0
        self.images = {}
        self.debug=""

    def __repr__(self):
        """
        Représentation textuelle de l'objet Camera.
        """
        return (f"Camera(cx={self.cx}, cy={self.cy}, f={self.f}, k1={self.k1}, k2={self.k2}, "
                f"k3={self.k3}, k4={self.k4}, p1={self.p1}, p2={self.p2}, b1={self.b1}, b2={self.b2})")
    
    
    def import_from_class_calc_photo_homol(self, homol):
        self.cx = homol.cx
        self.cy = homol.cy
        self.f  = homol.f 
        self.k1 = homol.k1
        self.k2 = homol.k2
        self.k3 = homol.k3
        self.k4 = homol.k4
        self.p1 = homol.p1
        self.p2 = homol.p2
        self.b1 = homol.b1
        self.b2 = homol.b2
        self.w  = homol.w 
        self.h  = homol.h 
        
        self.calibrations["image_use"]={"w" : self.w, "h":self.h,"cx":self.cx, "cy":self.cy, "f":self.f, "k1":self.k1, "k2":self.k2, "k3":self.k3, "k4":self.k4, "p1":self.p1, "p2":self.p2, "b1":self.b1, "b2":self.b2}
        
        
        self.ajout_photo(homol.imagepath, homol.R, homol.S.ravel(), "image_use")
        
        
    def import_calib(self, calibration_name, path_agisoft_calib):
        # Charger le fichier XML
        self.cx = 0
        self.cy = 0
        self.f = 0
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.k4 = 0
        self.p1 = 0
        self.p2 = 0
        self.b1 = 0
        self.b2 = 0
        self.w = 0
        self.h = 0
        
        tree = ET.parse(path_agisoft_calib)
        root = tree.getroot()
        
        
        # Parcourir tous les éléments dans le fichier XML
        for elem in root:
            if elem.tag=="width":
                self.w=int(elem.text)
            elif elem.tag=="height":
                self.h=int(elem.text)
            elif elem.tag=="f":
                self.f=float(elem.text)
            elif elem.tag=="cx":
                self.cx=float(elem.text)
            elif elem.tag=="cy":
                self.cy=float(elem.text)
            elif elem.tag=="k1":
                self.k1=float(elem.text)
            elif elem.tag=="k2":
                self.k2=float(elem.text)
            elif elem.tag=="k3":
                self.k3=float(elem.text)
            elif elem.tag=="k4":
                self.k4=float(elem.text)
            elif elem.tag=="p1":
                self.p1=float(elem.text)
            elif elem.tag=="p2":
                self.p2=float(elem.text)
            elif elem.tag=="b1":
                self.b1=float(elem.text)
            elif elem.tag=="b2":
                self.b2=float(elem.text)
                
        self.calibrations[calibration_name]={"w" : self.w, "h":self.h,"cx":self.cx, "cy":self.cy, "f":self.f, "k1":self.k1, "k2":self.k2, "k3":self.k3, "k4":self.k4, "p1":self.p1, "p2":self.p2, "b1":self.b1, "b2":self.b2}
        print(self.__repr__)

            
            
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
                     try:
                         I[16]
                     except:
                         calibname="None"
                     else:
                         calibname= I[16]
                     self.ajout_photo(name, R, S, calibname)
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
        y_prime = y*(1+self.k1*r**2+self.k2*r**4+self.k3*r**6+self.k4*r**8)+(self.p2*(r**2+2*y**2)+2*self.p1*x*y)
        u=self.w*0.5+self.cx+x_prime*self.f+x_prime*self.b1+y_prime*self.b2
        v=self.h*0.5+self.cy+y_prime*self.f
        uv=np.array([u,v])
        return uv
    
    def set_camera_calib(self, w=0, h=0, cx=0, cy=0, f=0, k1=0, k2=0, k3=0, k4=0, p1=0, p2=0, b1=0, b2=0 ):
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
        self.w = w
        self.h = h
    def set_calib_from_image(self, photoname=None):
        try:
            calib=self.calibrations[self.images[photoname]["camera"]]
        except:
            print("calibration "+self.images[photoname]["camera"] +" inexistante")
            self.cx = 0
            self.cy = 0
            self.f = 0
            self.k1 = 0
            self.k2 = 0
            self.k3 = 0
            self.k4 = 0
            self.p1 = 0
            self.p2 = 0
            self.b1 = 0
            self.b2 = 0
            self.w = 0
            self.h = 0
        else:
            self.cx = calib["cx"]
            self.cy = calib["cy"]
            self.f = calib["f"]
            self.k1 = calib["k1"]
            self.k2 = calib["k2"]
            self.k3 = calib["k3"]
            self.k4 = calib["k4"]
            self.p1 = calib["p1"]
            self.p2 = calib["p2"]
            self.b1 = calib["b1"]
            self.b2 = calib["b2"]
            self.w = calib["w"]
            self.h = calib["h"]
        
    def ajout_photo(self, name, R, S, camera):
        """
        Ajoute une photo
        
        :param name: nom de la photo
        :param R: Matrice de rotation (r11, r12, ...)
        :param S: Matrice des coordonnées de la caméra (X, Y, Z) dans le système globale
        """
        self.images[name]={"R":R, "S":S, "camera":camera}
        
    def liste_image_direction_proximite(self, photoname, image_non=[]):
        
        R_m=self.images[photoname]["R"]
        S_M=self.images[photoname]["S"]
        vect_M=self.vect_dir_camera(photoname)
        ordre_image_correspondance=[]
        
        dict_images=copy.deepcopy(self.images)
        del dict_images[photoname]
        for key in dict_images:
            if key not in image_non:
                vect_i=self.vect_dir_camera(key)
                S_I=dict_images[key]["S"]
                scalaire1 = np.dot(vect_M, vect_i)
                d_MI=np.linalg.norm(S_I-S_M)
                vect_M_I = (S_I-S_M) / d_MI
                scalaire2= np.dot(vect_M_I, vect_M)
                scalaire3= np.dot(-vect_M_I, vect_i)
                
                
                poids_dist=0.00001
                if d_MI<60:
                    poids_dist=1-d_MI/70
                poidsscalaire=scalaire1
                
                poids=(poids_dist*poidsscalaire)

            
            
                
            
                ordre_image_correspondance.append([key, poids, scalaire1, scalaire2, scalaire3, d_MI])
            
        ordre_image_correspondance=np.array(ordre_image_correspondance, dtype=object)
        indice_tri= ordre_image_correspondance[:,1].argsort()[::-1] 
        return ordre_image_correspondance[indice_tri]
        
    def vect_dir_camera(self,photoname):
        R=self.images[photoname]["R"]
        N=np.dot(R.T,np.array([0,0,-1]))
        return N
        
    def point_terrain_to_point_camera(self, photoname, M):
        S=self.images[photoname]["S"]
        R=self.images[photoname]["R"]
        self.set_calib_from_image(photoname)
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
        self.set_calib_from_image(photoname)
        
        
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
        self.set_calib_from_image(photoname)
        
        
        P=M
        PS=P-S
        
        N=self.vect_dir_camera(photoname)
        norm_N=np.linalg.norm(N)
        norm_PS=np.linalg.norm(PS)
        H=P-(PS@N)/(norm_PS*norm_N)*norm_PS/norm_N*N
        norm_MH=np.abs((PS@N)/(norm_PS*norm_N)*norm_PS)
        return H, norm_MH
    
    # def dist_to_dproj(self, N, dist, )
        
    def uv_to_M_by_dist_prof(self, photoname, uv, dist_proj):
        m_prime=self.uv_to_m_prime(uv)
        S=self.images[photoname]["S"]
        R=self.images[photoname]["R"]
        self.set_calib_from_image(photoname)
        
        x=m_prime[0]
        y=m_prime[1]
        for i in range(100):
            uv_prov=self.distorsion_frame_brown_agisoft_from_xy(x, y)
            diff_uv=uv-uv_prov
            print(i)
            if abs(diff_uv[0])>0.3 or abs(diff_uv[1])>0.3:
                x+=diff_uv[0]/4000
                y+=diff_uv[1]/4000
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
        r = Rot.from_euler('xyz', [o, p, k], degrees=True)
        # return rot_z(k)@rot_y(p)@rot_x(o)
        return r.as_matrix()
    
    def rotmatrix_to_euler(self, R):
        r =  Rot.from_matrix(R)
        angles = r.as_euler("xyz",degrees=True)
        return angles

    def mesh_pyramide_camera_create(self,photoname, distance, boundaries=None):
        self.set_calib_from_image(photoname)
        if boundaries is None:
            x1=self.uv_to_M(photoname,np.array([0,0]),distance)
            x2=self.uv_to_M(photoname,np.array([0,self.h]),distance)
            x3=self.uv_to_M(photoname,np.array([self.w,self.h]),distance)
            x4=self.uv_to_M(photoname,np.array([self.w,0]),distance)
        else:
            x1=self.uv_to_M(photoname,np.array([boundaries[0,0],0]),distance)
            x2=self.uv_to_M(photoname,np.array([boundaries[0,0],self.h]),distance)
            x3=self.uv_to_M(photoname,np.array([boundaries[0,1],self.h]),distance)
            x4=self.uv_to_M(photoname,np.array([boundaries[0,1],0]),distance)
            
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
        # maillage.export(photoname+'.stl')
        return maillage

    
    def pointcloudcamera_from_pathlas(self, pathlas, photoname):
        point_cloud, normales=pcd.readlas_to_numpy(pathlas)
        o3d_pcd=pcd.array_TO_o3d(point_cloud, normales)
        o3d_pcd=pcd.remove_statistical_outlier_pcd(o3d_pcd)
        point_cloud, normales=pcd.o3d_TO_array(o3d_pcd)
        mesh=self.mesh_pyramide_camera_create(photoname, 500)
        
        points_inside, normal_inside=pcd.points_inside_mesh(point_cloud, normales, mesh)
        
        return points_inside, normal_inside
        
    
            
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





