# -*- coding: utf-8 -*-
"""
Created on Fri May  2 07:24:59 2025

@author: Bruno
"""
import module_photogrammetrie_vector as mpv
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

import numpy as np
import os
import cv2
import pointcloud_module as pcd
import plot_module as plot
from scipy.spatial.transform import Rotation as Rot
import fonction_optimisee_numba as jit_fct
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial.transform import Rotation as Rot
import math as m
global device
global extractor
global matcher
global pathprojet



class homol_IA:
    def __init__(self, imagepath, position_approchee, projetcamera, pathprojet):
        
        self.position_approchee=position_approchee
        self.local_decalage=np.array([m.floor(position_approchee[0] / 10000) * 10000-5000, m.floor(position_approchee[1] / 10000) * 10000-5000])
        self.imagepath=imagepath
        self.image_cv2=None
        self.image_cible=None
        self.feats_cible=None
        self.fact_reduce_maitresse=1.0
        self.feats={}
        self.projetcamera=projetcamera
        self.pathprojet=pathprojet
        self.homol_array=None
        self.dict_homol=None
        self.image_traitee=[]
        self.R=None
        self.o=None
        self.p=None
        self.k=None
        self.S=None
        self.cx=0.0
        self.cy=0.0
        self.f=None
        self.w=None
        self.h=None
        self.k1=0.0
        self.k2=0.0
        self.k3=0.0
        self.k4=0.0
        self.p1=0.0
        self.p2=0.0
        self.b1=0.0
        self.b2=0.0
        
        self.init_camera_read()
        
    def init_camera_read(self):
        self.image_cv2=cv2.imread(os.path.join(self.pathprojet, "image", self.imagepath))

        
        self.w=self.image_cv2.shape[1]
        self.h=self.image_cv2.shape[0]
    
    def distorsion_nulle(self):
        self.k1=0.0
        self.k2=0.0
        self.k3=0.0
        self.k4=0.0
        self.p1=0.0
        self.p2=0.0
        self.b1=0.0
        self.b2=0.0
        self.cx=0.0
        self.cy=0.0
        
    def trouver_cameras_proches_numpy(self, n=5):
        
        dico_cameras=self.projetcamera.images
        
        noms = list(dico_cameras.keys())
        positions = np.array([dico_cameras[nom]["S"][[0,1]] for nom in noms])  # (N, 3)
    
        position_ref = np.array(self.position_approchee)
        distances = np.linalg.norm(positions - position_ref, axis=1)  # (N,)
    
        indices_tris = np.argsort(distances)[:n]
        return [(noms[i], distances[i]) for i in indices_tris]
        
    def import_modele_IA(self):
        global device
        global extractor
        global matcher
        try:
            device
        except:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
            extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)  # load the extractor
            matcher = LightGlue(features='superpoint').eval().to(device)  # load the matcher
        
    def first_feats_matches_calc(self, resize=None):
        global device
        global extractor
        global matcher
        
        
        #ELEMENT DE L'IMAGE MAITRESSE
        w_m=self.projetcamera.calibrations[self.projetcamera.images[self.imagepath]["camera"]]["w"]
        h_m=self.projetcamera.calibrations[self.projetcamera.images[self.imagepath]["camera"]]["h"]
        
        max_im_m=max(w_m, h_m)

        if resize is not None:
            self.fact_reduce_maitresse=resize/max_im_m
        
        self.import_modele_IA()
        
        self.image_cible = load_image(os.path.join(self.pathprojet,'image',self.imagepath), resize=resize, fn="max").cpu()
        self.feats_cible = extractor.extract(self.image_cible)
        
        plus_proches = self.trouver_cameras_proches_numpy(n=5)
        
        liste_homol=[]
        for imagename, dist in plus_proches:
            print(f"calcul de l'image {imagename}")
            self.feats_calcul(self.feats_cible, imagename, resize)

        self.homol_array=self.homol_array[self.homol_array[:,1].argsort()[::-1]]
        return self.homol_array
    
    
    def feats_calcul(self, feats0, imagename, resize=None):
        global device
        global extractor
        global matcher
        self.import_modele_IA()
        if imagename!=self.imagepath:
        
            #ELEMENT DE L'IMAGE MAITRESSE
            w=self.projetcamera.calibrations[self.projetcamera.images[self.imagepath]["camera"]]["w"]
            h=self.projetcamera.calibrations[self.projetcamera.images[self.imagepath]["camera"]]["h"]
            self.image_traitee.append(imagename)
            max_im=max(w, h)
            fact_reduce_im=1.0
            if resize is not None:
                fact_reduce_im=resize/max_im
            
            
            
            image = load_image(os.path.join(self.pathprojet,'image',imagename), resize=resize, fn="max").cpu()
            feats = extractor.extract(image)
            matches01 = matcher({'image0': self.feats_cible, 'image1': feats})
            feats_cible_matches, feats, matches01 = [
                rbd(x) for x in [self.feats_cible, feats, matches01]
            ]
            kpts0, kpts1, matches = feats_cible_matches["keypoints"], feats["keypoints"], matches01["matches"]
            
            m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
            
            scores = matches01["scores"].detach().cpu().numpy()
            pts1=m_kpts0.cpu().numpy()
            pts2=m_kpts1.cpu().numpy()

            #epuration des points homologues
            F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.USAC_MAGSAC, ransacReprojThreshold=2, confidence=0.999999, maxIters=10000)
            
            # Inliers (points cohérents)
            pts1_inliers = pts1[mask.ravel() > 0]
            pts2_inliers = pts2[mask.ravel() > 0]

            # Outliers (points rejetés)
            pts1_outliers = pts1[mask.ravel() == 0]
            pts2_outliers = pts2[mask.ravel() == 0]
            
            
            if pts1_inliers.shape[0]>10 and self.seuil_score_nb_homol(pts1_inliers.shape[0])<np.median(scores):
                print(f"Correspondance OK : L'image {imagename}  compte {pts1_inliers.shape[0]} points homologues et un score median de {np.median(scores)}")
                axes = viz2d.plot_images([self.image_cible,image])
                viz2d.plot_matches(pts1_inliers, pts2_inliers, color="lime", lw=0.2)
                if self.homol_array is None:
                    self.homol_array =np.array([[imagename, pts1_inliers.shape[0], np.round(pts1_inliers/self.fact_reduce_maitresse, decimals=2), np.round(pts2_inliers/fact_reduce_im, decimals=2), scores[mask.ravel() > 0]]], dtype=object)
                else:
                    self.homol_array =np.vstack((self.homol_array,np.array([[imagename, pts1_inliers.shape[0], np.round(pts1_inliers/self.fact_reduce_maitresse, decimals=2), np.round(pts2_inliers/fact_reduce_im, decimals=2), scores[mask.ravel() > 0]]], dtype=object)))
            else:
                print(f"L'image {imagename} compte {pts1_inliers.shape[0]} points homologues et un score median de {np.median(scores)}")
        
    def calcul_iteratif_homol_matches(self):
        dict_homol_filtre, array_homol_tri=self.feats_analyse()
        i=20
        while array_homol_tri[array_homol_tri[:,4]>3].shape[0]<5:
            liste_homol=self.homol_array
            image_M=liste_homol[np.argmax(liste_homol[:, 1]),0]
            
            image_priorite = self.projetcamera.liste_image_direction_proximite(image_M,np.array(self.image_traitee))
            for i in range(5):
                self.feats_calcul(self.feats_cible, image_priorite[i,0],2500)
                
            dict_homol_filtre, array_homol_tri=self.feats_analyse()
            
        return dict_homol_filtre, array_homol_tri
    
    def feats_analyse(self):
        
        dict_homol_maitresse={}
        
        for i in range(self.homol_array.shape[0]):
            for j in range(self.homol_array[i,2].shape[0]):
                u_maitresse=self.homol_array[i,2][j,0]
                v_maitresse=self.homol_array[i,2][j,1]
                imagename=self.homol_array[i,0]
                u_homol=self.homol_array[i,3][j,0]
                v_homol=self.homol_array[i,3][j,1]
                S=self.projetcamera.images[imagename]["S"]
                key=str(u_maitresse)+"_"+str(v_maitresse)
                if key not in dict_homol_maitresse.keys():
                    dict_homol_maitresse[str(u_maitresse)+"_"+str(v_maitresse)]={"u" : u_maitresse, "v":  v_maitresse, "homol" :  []}
                
                
                vecteur=self.projetcamera.uv_to_M(imagename, [u_homol, v_homol], 1.0)-S
                dict_homol_maitresse[key]["homol"].append([imagename, u_homol, v_homol, vecteur, S])
                
        dict_homol_filtre = {k: v for k, v in dict_homol_maitresse.items() if len(v['homol']) > 0}
        for v in dict_homol_filtre.values():
            v["nb_homol"] = len(v["homol"])
        
        array_homol_tri = np.array([
            [key, v["u"], v["v"], v["homol"], v["nb_homol"]]
            for key, v in dict_homol_filtre.items()
        ], dtype=object)
        
        return dict_homol_filtre, array_homol_tri
        
        
    def calcul_approximatif_homol_3D(self, dict_homol):
        points3d={}
        A_liste=[]
        l_liste=[]
        res=[]
        dict_supprimer={}
        dict_valide={}
        liste_point3D=[]
        for key,v in dict_homol.items():
            if v["nb_homol"]>1:
                
                nb_i=3+v["nb_homol"]
                nb_l=v["nb_homol"]*3
                A=np.zeros((v["nb_homol"]*3, 3+v["nb_homol"]))
                l=np.zeros((v["nb_homol"]*3, 1))
                Qll=np.eye(v["nb_homol"]*3)*0.1**2
                P=np.linalg.inv(Qll)
                sigma0=1.0
                
                for i in range(v["nb_homol"]):
                    A[i*3,0]=1
                    A[i*3+1,1]=1
                    A[i*3+2,2]=1
                    A[i*3,i+3]=-v["homol"][i][3][0]
                    A[i*3+1,i+3]=-v["homol"][i][3][1]
                    A[i*3+2,i+3]=-v["homol"][i][3][2]
                    
                    l[i*3,0]=v["homol"][i][4][0]
                    l[i*3+1,0]=v["homol"][i][4][1]
                    l[i*3+2,0]=v["homol"][i][4][2]
                
                A_liste.append(A)
                l_liste.append(l)
                
                Qxx=np.linalg.inv(A.T@P@A)
                x=np.linalg.inv(A.T@A)@A.T@l

                res.append(x)
                
                
                vi=A@x-l+0.00000001
                vi = np.asarray(vi, dtype=np.float64)
                P = np.asarray(P, dtype=np.float64)
                vitpvi=(vi.T@P@vi)
                s0=np.sqrt((vitpvi)/(nb_l-nb_i))
                Quot=s0/sigma0
                
                
                if Quot>1.2 and nb_l>2: 
                    prev_inc=x.copy()
                    delta=0.2
                    for iteration in range(2):
                        # perte = fonction_de_perte_huber(vi, delta)
                        perte=np.where(np.abs(vi) <= delta,
                                        0.5 * vi**2,
                                        delta * (np.abs(vi) - 0.5 * delta))
                        W = np.diag(perte.flatten())
                        Qxx_rob= np.linalg.inv(A.T @ W @ A)
                        x = Qxx_rob @ A.T @ W @ l
                        vii = l - A @ x
                        if np.linalg.norm(x - prev_inc) < 0.001:
                            # print(f"Convergence atteinte après {iteration + 1} itérations.")
                            break
                        prev_inc=x.copy()
                
                
                #CALCUL PAR REPROJECTION 
                CoordM=np.array([x[0,0], x[1,0], x[2,0]])
                FS_list=[]
                uv_proj_list=[]
                for i in range(v["nb_homol"]):
                    imagename=v["homol"][i][0]
                    uv_mes=np.array([v["homol"][i][1],v["homol"][i][2]])
                    uv_proj=self.projetcamera.M_to_uv(imagename, CoordM)
                    v["homol"][i].append(uv_proj)
                    FS_uv= np.linalg.norm(uv_mes - uv_proj)
                    
                    FS_list.append(FS_uv)
                    uv_proj_list.append(uv_proj)
                    
                    
                    
                
                
                dict_homol[key]["coord3D"]=CoordM
                dict_homol[key]["FS"]=FS_list
                dict_homol[key]["uv_proj"]=uv_proj_list
                dict_homol[key]["Qxx"]=Qxx
                dict_homol[key]["Quot"]=Quot
                
                
                if (Quot>1.2 or np.mean(FS_list)>5 or (Qxx[0,0]>10 or Qxx[1,1]>10 and Qxx[2,2]>10 ) or x[4,0]<0) :
                    dict_supprimer[key]=dict_homol[key]
                else:
                    liste_point3D.append(CoordM)
                    dict_valide[key]=dict_homol[key]
        pcd.save_point_cloud_las(np.array(liste_point3D), "Ouput/homol.las")
        self.dict_homol=dict_valide
        return  dict_valide,dict_supprimer
    
    def proj_points_3D_dict_to_image(self):
        dict_homol=self.dict_homol
        img=Image.open(os.path.join(self.pathprojet, "image",self.imagepath))
        max_size = 1000
        w, h = img.size
        scale = min(max_size / w, max_size / h)
        new_size = (int(w * scale), int(h * scale))
        # image_origine.show()
        img = img.resize(new_size, Image.LANCZOS)
        
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.3)
        
        img_array=np.asarray(img)
        fig, ax = plt.subplots()
        ax.imshow(img_array)
        ax.axis('off') 
        u_liste=[]
        v_liste=[]
        u_proj_liste=[]
        v_proj_liste=[]
        for k, va in dict_homol.items():
            uv_proj=self.projetcamera.M_to_uv(self.imagepath, va["coord3D"])
            dict_homol[k]["u_proj"]=uv_proj[0]
            dict_homol[k]["v_proj"]=uv_proj[1]
            dict_homol[k]["fs_proj"]=((uv_proj[0]-va["u"])**2 + (uv_proj[1]-va["v"])**2)**0.5
            u_proj_liste.append(uv_proj[0]*scale)
            v_proj_liste.append(uv_proj[1]*scale)
            
            u_liste.append(va["u"]*scale)
            v_liste.append(va["v"]*scale)
            
            
            u_line=[uv_proj[0]*scale, va["u"]*scale]
            v_line=[uv_proj[1]*scale, va["v"]*scale]
            ax.plot(u_line, v_line, c="orange")
            
        sc=ax.scatter(u_liste, v_liste, c="lime", linewidths=0, alpha=0.8, label='Points homologues')
        sc=ax.scatter(u_proj_liste, v_proj_liste, c="red", linewidths=0, alpha=0.8,  label='Points projetés depuis 3D')
        # plt.title("Comparaison des points 3D projetés et des points homologues")
        plt.legend(loc="lower left")
            
            
            
            
        self.dict_homol=dict_homol
    
    
    def RANSAC_DLT(self, k=200, n_samples=20, threshold=5.0, nb_inliers=15, dict_homol=None):
        if dict_homol is None:
            dict_homol=self.dict_homol
        bestFit=None
        bestErr=9999
        
        bestKey_inliers=[]
        vi_best=None
        bestDet=0
        for i in range(k):
            cles_aleatoires = random.sample(list(dict_homol.keys()), n_samples)
            
            cles_inliers=[]
            Li, vi = self.DLT_image_cible(cles_aleatoires, dict_homol)

            nb_l=len(dict_homol)
            i=0
            fs_list=[]
            for key, val in dict_homol.items():
                fs_px=(vi[i*2,0]**2+vi[i*2+1,0]**2)**0.5
                # print(f"FS_brute: {fs_px}")
                
               
                
                if fs_px<threshold:
                    cles_inliers.append(key)
                    fs_list.append(fs_px)
                i+=1
            # print(f"k{i}:{len(cles_inliers)}")
            if len(cles_inliers)>nb_inliers:
                # Li, vi = self.DLT_image_cible(cles_inliers)
                i=0
                fs_list=[]
                
                self.DLT_res_to_param(Li)
                # if np.linalg.det(self.R)>0.8:
                # print(np.linalg.det(self.R))
                
                for key, val in self.dict_homol.items():
                    if key in cles_inliers:
                        fs_px=(vi[i*2,0]**2+vi[i*2+1,0]**2)**0.5
                        fs_list.append(fs_px)
                    i+=1
                vi_moy=np.mean(fs_list)
                
                # print( np.linalg.det(self.R))
                if vi_moy<bestErr:

                    bestErr=vi_moy
                    bestFit=np.copy(Li)
                    bestKey_inliers=cles_inliers
                    vi_best=vi

                        
        if bestFit is not None:
            self.DLT_res_to_param(bestFit)
            print(f"Sx: {self.S[0,0]}")
            print(f"Sy: {self.S[1,0]}")
            print(f"Sz: {self.S[2,0]}")
            print(f"f: {self.f}")
            print(f"cx: {self.cx}")
            print(f"cy: {self.cy}")
            print(f"==========================")
            # print(f"Key: {bestKey_inliers}")
            print(f"nb_inliers: {len(bestKey_inliers)}")
            print(f"Erreur moy inliers : {bestErr}")
            fact=-1
            print(f"det : {np.linalg.det(self.R)}")
            print(self.R)
            r =  Rot.from_matrix(self.R)
            angles = r.as_euler("xyz",degrees=True)
            print(f"Angle de rotation  :{angles}")
            r_retour = Rot.from_euler('xyz', angles*fact, degrees=True)
            print(r_retour.as_matrix())
            # print(f"Erreur : {vi_best}")
            # print(f"Erreur : {np.max(np.abs(vi_best))}")
            
            uv_liste=[]
            uv_outliers=[]
            
            i=0
            for key, val in dict_homol.items():
                u=val["u"]
                v=val["v"]


                if key in bestKey_inliers:
                    uv_liste.append([u,v])
                else:
                    uv_outliers.append([u,v])
                i+=1
                
            plot.show_only_point_in_image(os.path.join(self.pathprojet,"image", self.imagepath), np.array(uv_liste), np.array(uv_outliers))
            # print(f"Nb_l : {vi_best.shape[0]}")
        else:
            print("Aucun modèle permet de résoudre la DLT")
            

                
                
                
                    
        
    
    def DLT_image_cible(self, liste_cles_use=None, dict_homol=None):
        if dict_homol is None:
            dict_homol=self.dict_homol
        nb_l=len(dict_homol)*2
        nb_inc=11
        A=np.zeros((nb_l, nb_inc))
        l=np.zeros((nb_l, 1))
        Qll=np.eye(nb_l)
        
        i=0
        sigma0=1.0
        for key, val in dict_homol.items():
            
            if liste_cles_use is None:
                Qll[i,i]=Qll[i,i]*1**2
            elif key in liste_cles_use:
                Qll[i,i]=Qll[i,i]*1**2
            else:
                Qll[i,i]=Qll[i,i]*99999**2
                
                
            
            mx=val["u"]-self.w/2
            my=-val["v"]+self.h/2
            # mx=val["u_proj"]-self.w/2
            # my=-val["v_proj"]+self.h/2
            l[i*2,0]=mx
            l[i*2+1,0]=my
            
            A[i*2,0]=val["coord3D"][0]-self.local_decalage[0]
            A[i*2,1]=val["coord3D"][1]-self.local_decalage[1]
            A[i*2,2]=val["coord3D"][2]
            A[i*2,3]=1
            A[i*2,8]=-mx*(val["coord3D"][0]-self.local_decalage[0])
            A[i*2,9]=-mx*(val["coord3D"][1]-self.local_decalage[1])
            A[i*2,10]=-mx*val["coord3D"][2]
            
            
            A[i*2+1,4]=val["coord3D"][0]-self.local_decalage[0]
            A[i*2+1,5]=val["coord3D"][1]-self.local_decalage[1]
            A[i*2+1,6]=val["coord3D"][2]
            A[i*2+1,7]=1
            A[i*2+1,8]=-my*(val["coord3D"][0]-self.local_decalage[0])
            A[i*2+1,9]=-my*(val["coord3D"][1]-self.local_decalage[1])
            A[i*2+1,10]=-my*val["coord3D"][2]
            
            
            i+=1
            
        P=np.linalg.inv(Qll)
        Qxx=np.linalg.inv(A.T@P@A)
        
        Li=Qxx@A.T@P@l
        vi=A@Li-l
        
        vi_mean=np.mean(np.abs(vi))
        Qll=np.eye(nb_l)*vi_mean**2
        P=np.linalg.inv(Qll)
        vitpvi=(vi.T@P@vi)
        s0=np.sqrt((vitpvi)/(nb_l-nb_inc))
        Quot=s0/sigma0

        
        
        
        Qvv=Qll-A@Qxx@A.T
        wi=np.zeros((nb_l, 1))

        
        return Li, vi
    def RANSAC_DLT_rot(self, k=50, n_samples=10, threshold=5.0, nb_inliers=30, dict_homol=None):
        if dict_homol is None:
            dict_homol=self.dict_homol
        bestFit=None
        bestErr=9999
        
        bestKey_inliers=[]
        vi_best=None
        bestDet=0
        for i in range(k):
            cles_aleatoires = random.sample(list(dict_homol.keys()), n_samples)
            # print(f"clé aléatoire: {len(cles_aleatoires)}")
            cles_inliers=[]
            R, Ji, vi ,P= self.DLT_simplifiee_rotation(cles_aleatoires, dict_homol)
            vi_mean_pond= np.sum(np.abs(vi.T)@P)/np.sum(P)
            if np.linalg.det(R)>0 and vi_mean_pond<bestErr :
                # print(P)
                # print(np.linalg.det(R))
                r =  Rot.from_matrix(R)
                angles = r.as_euler("xyz",degrees=True)
                self.o=-angles[0]
                self.p=-angles[1]
                self.k=-angles[2]
                r = Rot.from_euler('xyz', [self.o, self.p, self.k], degrees=True)
                # return rot_z(k)@rot_y(p)@rot_x(o)
                self.R= r.as_matrix()
                bestErr=vi_mean_pond
                
        print("Recalcul orientation")
        print(f"angle :  {self.o} {self.p} {self.k}")
        print(f"R : {self.R}")
        
        
        return P
            
    def DLT_simplifiee_rotation(self, liste_cles_use=None, dict_homol=None):
        S=self.S.ravel()
        if dict_homol is None:
            dict_homol=self.dict_homol
        nb_l=len(dict_homol)*2
        nb_inc=9
        A=np.zeros((nb_l, nb_inc))
        l=np.zeros((nb_l, 1))
        Qll=np.eye(nb_l)
        
        i=0
        sigma0=1.0
        for key, val in dict_homol.items():
            mx=val["u"]-self.w/2
            my=val["v"]-self.h/2
            
            diag_max_img=((self.w/2)**2+(self.h/2)**2)**0.5
            diage_m=(mx**2+my**2)**0.5
            p_pond_dist_centre = 0.2*(3)**(diage_m/(diag_max_img/7))
            if liste_cles_use is None:
                Qll[i,i]=Qll[i,i]*(1)**2
            elif key in liste_cles_use:
                Qll[i,i]=Qll[i,i]*(1)**2
            else:
                Qll[i,i]=Qll[i,i]*99999**2
                
                
            
            
            # mx=val["u_proj"]-self.w/2
            # my=val["v_proj"]-self.h/2
            l[i*2,0]=mx
            l[i*2+1,0]=my
            f=self.f
            A[i*2,0]=(val["coord3D"][0]-S[0])*f
            A[i*2,1]=(val["coord3D"][1]-S[1])*f
            A[i*2,2]=(val["coord3D"][2]-S[2])*f
            A[i*2,6]=-mx*(val["coord3D"][0]-S[0])
            A[i*2,7]=-mx*(val["coord3D"][1]-S[1])
            A[i*2,8]=-mx*(val["coord3D"][2]-S[2])
            
            
            A[i*2+1,3]=(val["coord3D"][0]-S[0])*f
            A[i*2+1,4]=(val["coord3D"][1]-S[1])*f
            A[i*2+1,5]=(val["coord3D"][2]-S[2])*f
            A[i*2+1,6]=-my*(val["coord3D"][0]-S[0])
            A[i*2+1,7]=-my*(val["coord3D"][1]-S[1])
            A[i*2+1,8]=-my*(val["coord3D"][2]-S[2])
            
            
            i+=1
            
        P=np.linalg.inv(Qll)
        Qxx=np.linalg.inv(A.T@P@A)
        
        Ji=Qxx@A.T@P@l
        vi=A@Ji-l
        
        
        r11=Ji[0,0]
        r12=Ji[1,0]
        r13=Ji[2,0]
        r21=Ji[3,0]
        r22=Ji[4,0]
        r23=Ji[5,0]
        r31=Ji[6,0]
        r32=Ji[7,0]
        r33=Ji[8,0]
        
        
        R=np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
            ])
        
        return R, Ji, vi, P
    
    
    def RANSAC_DLT_foc(self, k=50, n_samples=50, threshold=5.0, nb_inliers=30, dict_homol=None):
        if dict_homol is None:
            dict_homol=self.dict_homol
        bestFit=None
        bestErr=9999
        
        bestKey_inliers=[]
        vi_best=None
        bestDet=0
        for i in range(k):
            cles_aleatoires = random.sample(list(dict_homol.keys()), n_samples)
            # print(f"clé aléatoire: {len(cles_aleatoires)}")
            cles_inliers=[]
            f, Ji, vi,P = self.DLT_simplifiee_focale_ppa(cles_aleatoires, dict_homol)
            vi_mean_pond= np.sum(np.abs(vi.T)@P)/np.sum(P)
            if vi_mean_pond<bestErr :
                self.f=f
                r =  Rot.from_matrix(self.R)
                angles = r.as_euler("xyz",degrees=True)
                self.cx=Ji[1,0]
                self.cy=Ji[2,0]
                bestErr=vi_mean_pond
        print("Recalcul focale")
        print(f"f :  {self.f}")
        print(f"cx : {self.cx}")
        print(f"cy : {self.cy}")

        return P
        
        
        
    def DLT_simplifiee_focale(self, liste_cles_use=None, dict_homol=None):
        S=self.S.ravel()
        R=self.R.T
        if dict_homol is None:
            dict_homol=self.dict_homol
        nb_l=len(dict_homol)*2
        nb_inc=1
        A=np.zeros((nb_l, nb_inc))
        l=np.zeros((nb_l, 1))
        Qll=np.eye(nb_l)
        
        i=0
        sigma0=1.0
        for key, val in dict_homol.items():
            mx=val["u"]-self.w/2
            my=val["v"]-self.h/2
            
            diag_max_img=((self.w/2)**2+(self.h/2)**2)**0.5
            diage_m=(mx**2+my**2)**0.5
            p_pond_dist_centre = 0.2*(3)**(diage_m/(diag_max_img/7))
            
            if liste_cles_use is None:
                Qll[i,i]=Qll[i,i]*(1+p_pond_dist_centre)**2
            elif key in liste_cles_use:
                Qll[i,i]=Qll[i,i]*(1+p_pond_dist_centre)**2
            else:
                Qll[i,i]=Qll[i,i]*99999**2
                
                
            V=val["coord3D"]-S
            # print(V)
            mx=val["u"]-self.w/2
            my=val["v"]-self.h/2
            # mx=val["u_proj"]-self.w/2
            # my=-val["v_proj"]+self.h/2
            l[i*2,0]=mx
            l[i*2+1,0]=my
            
            A[i*2,0]=((val["coord3D"][0]-S[0])*R[0,0]+(val["coord3D"][1]-S[1])*R[0,1]+(val["coord3D"][2]-S[2])*R[0,2])/(R[2,0]*V[0]+R[2,1]*V[1]+R[2,2]*V[2])
            # A[i*2,1]=1
            
            A[i*2+1,0]=((val["coord3D"][0]-S[0])*R[1,0]+(val["coord3D"][1]-S[1])*R[1,1]+(val["coord3D"][2]-S[2])*R[1,2])/(R[2,0]*V[0]+R[2,1]*V[1]+R[2,2]*V[2])
            # A[i*2+1,2]=1
            
            
            i+=1
            
        P=np.linalg.inv(Qll)
        Qxx=np.linalg.inv(A.T@P@A)
        
        Ji=Qxx@A.T@P@l
        vi=A@Ji-l
        
        
        f=Ji[0,0]


        # print(f"Focale  :{self.f}")
        # print(f"Cx  :{Ji[1,0]}")
        # print(f"Cy  :{Ji[2,0]}")
        return f, Ji, vi, P
        
    def DLT_simplifiee_focale_ppa(self, liste_cles_use=None, dict_homol=None):
        S=self.S.ravel()
        R=self.R.T
        if dict_homol is None:
            dict_homol=self.dict_homol
        nb_l=len(dict_homol)*2
        nb_inc=3
        A=np.zeros((nb_l, nb_inc))
        l=np.zeros((nb_l, 1))
        Qll=np.eye(nb_l)
        
        i=0
        sigma0=1.0
        for key, val in dict_homol.items():
            mx=val["u"]-self.w/2
            my=val["v"]-self.h/2
            
            diag_max_img=((self.w/2)**2+(self.h/2)**2)**0.5
            diage_m=(mx**2+my**2)**0.5
            p_pond_dist_centre = 0.2*(3)**(diage_m/(diag_max_img/7))
            if liste_cles_use is None:
                Qll[i,i]=Qll[i,i]*(1+p_pond_dist_centre)**2
            elif key in liste_cles_use:
                Qll[i,i]=Qll[i,i]*(1+p_pond_dist_centre)**2
            else:
                Qll[i,i]=Qll[i,i]*99999**2
                
                
            V=val["coord3D"]-S
            # print(V)
            mx=val["u"]-self.w/2
            my=val["v"]-self.h/2
            # mx=val["u_proj"]-self.w/2
            # my=val["v_proj"]-self.h/2
            l[i*2,0]=mx
            l[i*2+1,0]=my
            
            A[i*2,0]=((val["coord3D"][0]-S[0])*R[0,0]+(val["coord3D"][1]-S[1])*R[0,1]+(val["coord3D"][2]-S[2])*R[0,2])/(R[2,0]*V[0]+R[2,1]*V[1]+R[2,2]*V[2])
            A[i*2,1]=1
            
            A[i*2+1,0]=((val["coord3D"][0]-S[0])*R[1,0]+(val["coord3D"][1]-S[1])*R[1,1]+(val["coord3D"][2]-S[2])*R[1,2])/(R[2,0]*V[0]+R[2,1]*V[1]+R[2,2]*V[2])
            A[i*2+1,2]=1
            
            
            i+=1
            
        P=np.linalg.inv(Qll)
        Qxx=np.linalg.inv(A.T@P@A)
        
        Ji=Qxx@A.T@P@l
        vi=A@Ji-l
        
                    
    
        # print(f"Focale  :{self.f}")
        # print(f"cx  :{Ji[1,0]}")
        # print(f"cy  :{Ji[2,0]}")
        # return R
        
        return Ji[0,0], Ji, vi, P
        
        
        
        
    def DLT_res_to_param(self, Li):
        Li=Li.ravel()
        L=1/(Li[8]**2+Li[9]**2+Li[10]**2)**0.5
        
        cx=L**2*(Li[0]*Li[8]+Li[1]*Li[9]+Li[2]*Li[10])
        cy=L**2*(Li[4]*Li[8]+Li[5]*Li[9]+Li[6]*Li[10])
        xx=(L**2*(Li[0]**2+Li[1]**2+Li[2]**2)-cx**2)**0.5
        yy=(L**2*(Li[4]**2+Li[5]**2+Li[6]**2)-cy**2)**0.5
        self.f=(xx+yy)/2
        
        self.cx=cx
        self.cy=cy
        
        
        r11=L*(cx*Li[8]-Li[0])/xx
        r12=L*(cx*Li[9]-Li[1])/xx
        r13=L*(cx*Li[10]-Li[2])/xx
        r21=L*(cy*Li[8]-Li[4])/yy
        r22=L*(cy*Li[9]-Li[5])/yy
        r23=L*(cy*Li[10]-Li[6])/yy
        r31=L*Li[8]
        r32=L*Li[9]
        r33=L*Li[10]
        
        
        self.R=np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
            ])
        if np.linalg.det(self.R)>0.0:
            r =  Rot.from_matrix(self.R)
            angles = r.as_euler("xyz",degrees=True)
            self.o=angles[0]
            self.p=angles[1]
            self.k=angles[2]
        else:
            self.R=self.R*-1
            r =  Rot.from_matrix(self.R)
            angles = r.as_euler("xyz",degrees=True)
            self.o=angles[0]
            self.p=angles[1]
            self.k=angles[2]
        # r = Rot.from_euler('xyz', angles, degrees=True)
        # # return rot_z(k)@rot_y(p)@rot_x(o)
        # self.R =r.as_matrix()
        
        #CALCUL SELON https://photogrammetry.fi/?p=257
        # L=1/(Li[8]**2+Li[9]**2+Li[10]**2)**0.5
        
        # ppx=L**2*(Li[0]*Li[8]+Li[1]*Li[9]+Li[2]*Li[10])
        # ppy=L**2*(Li[4]*Li[8]+Li[5]*Li[9]+Li[6]*Li[10])
        # # self.cx=ppx
        # # self.cy=ppy
        # self.f=(L**2*(ppx*Li[8]-Li[0])**2+L**2*(ppx*Li[9]-Li[1])**2+L**2*(ppx*Li[10]-Li[2])**2)**0.5
        
        
        # r11=L*(self.cx*Li[8]-Li[0])/self.f
        # r12=L*(self.cx*Li[9]-Li[1])/self.f
        # r13=L*(self.cx*Li[10]-Li[2])/self.f
        # r21=L*(self.cy*Li[8]-Li[4])/self.f
        # r22=L*(self.cy*Li[9]-Li[5])/self.f
        # r23=L*(self.cy*Li[10]-Li[6])/self.f
        # r31=L*(-Li[8])
        # r32=L*(-Li[9])
        # r33=-L*Li[10]
        
        
        # self.R=np.array([
        #     [r11, r12, r13],
        #     [r21, r22, r23],
        #     [r31, r32, r33]
        #     ])
        
        
        Lmat=np.array([
            [Li[0], Li[1], Li[2]],
            [Li[4], Li[5], Li[6]],
            [Li[8], Li[9], Li[10]]
            ])
        
        Lmat2=np.array([
            [Li[3]],
            [Li[7]],
            [1]
            ])
        
        self.S=-np.linalg.inv(Lmat)@Lmat2+np.array([[self.local_decalage[0]], [self.local_decalage[1]],[0]])
        
        
            
        
        
        
        
        
                
    def show_matches_paires(self, dict_homol, image1, image2=None):
        
        uv_cible=[]
        pts1=[]
        pts2=[]
        uv1=[]
        uv2=[]
        if image2 is None:
            image2=self.imagepath
        
        
        for k, va in dict_homol.items():
            homol_array=np.array(va["homol"], dtype=object)
            if image2==self.imagepath and image1 in homol_array[:,0]:
                image1_homol=homol_array[homol_array[:,0]==image1][0,:]
                pts1.append([image1_homol[1],image1_homol[2]])
                pts2.append([va["u"],va["v"]])
                
                
            elif image1 in homol_array[:,0] and image2 in homol_array[:,0]:
                
                image1_homol=homol_array[homol_array[:,0]==image1][0,:]
                image2_homol=homol_array[homol_array[:,0]==image2][0,:]
                pts1.append([image1_homol[1],image1_homol[2]])
                pts2.append([image2_homol[1],image2_homol[2]])
                
                uv1.append(image1_homol[5].tolist())
                uv2.append(image2_homol[5].tolist())
                
                
            
            uv_cible.append([va["u"],va["v"]])

        draw_matches(os.path.join(self.pathprojet,"image", image1), os.path.join(self.pathprojet,"image", image2), np.array(pts1), np.array(pts2))
        
        plot.show_point_in_image(os.path.join(self.pathprojet,"image", self.imagepath), np.array(uv_cible))
                

            
        

            
            
    def seuil_score_nb_homol(self, nb_points):
        nb=[0,20,40,80,150,1500]
        seuil=[0.5,0.35,0.25,0.2,0.15,0.1]
        
        return float(np.interp(nb_points, nb, seuil))
            
        
        

    def keypoint_tensor_IA(self, image_path, imageload):
        feats0 = extractor.extract(imageload)
        return feats0
    
    def matches_keypoint_tensor_IA(self, feats0, feats1):
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        self.matches01=matches01
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension
        
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        self.kpts1=kpts0
        self.kpts2=kpts1
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        self.m_kpts1=m_kpts0
        self.m_kpts2=m_kpts1
    
    def calcul_position_approximative(self):
        print("test")
    
    
    def calcul_moindre_carre_non_lineaire(self, dict_homol=None, liste_key_none=[], liste_inc=["S", "angles", "f", "k1"]):
        if dict_homol is None:
            dict_homol=self.dict_homol
        M=[]
        uv=[]
        duv=[]
        for key, val in dict_homol.items():
            
            if key not in liste_key_none:
                Mx=val["coord3D"].reshape(3,1)
        
                
                M.append(val["coord3D"])
        
                uv.append([val["u"], val["v"]])
            
        M_loc=np.array(M)-self.S.ravel()
        uv=np.array(uv)
    
        Qxx, x, A, dx, B, dl, wi, vi ,res= mpv.gauss_markov_non_lineaire_photog(np.array(uv), self.w, self.h, self.S.ravel(), np.array(M)-self.S.ravel(), self.o*m.pi/180, self.p*m.pi/180, self.k*m.pi/180, self.f, self.cx, self.cy, self.k1, self.k2, self.k3, self.k4, self.p1, self.p2, self.b1, self.b2, liste_inc=liste_inc)
        self.change_calib_calc_from_resdict(res)
        return Qxx, x, A, dx, B, dl, wi, vi
    
    def change_calib_calc_from_resdict(self,res):
        self.S=res["S"]
        self.o=res["angles"][0]
        self.p=res["angles"][1]
        self.k=res["angles"][2]
        self.k1=res["k1"]
        self.k2=res["k2"]
        self.k3=res["k3"]
        self.k4=res["k4"]
        self.p1=res["p1"]
        self.p2=res["p2"]
        self.b1=res["b1"]
        self.b2=res["b2"]
        r = Rot.from_euler('xyz', [self.o, self.p, self.k], degrees=True)

        self.R= r.as_matrix()
    
    
    def view_keypoint(self):
        axes = viz2d.plot_images([self.imageload1, self.imageload2])
        viz2d.plot_matches(self.m_kpts1, self.m_kpts2, color="lime", lw=0.2)
        # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
        
        kpc0, kpc1 = viz2d.cm_prune(self.matches01["prune0"]), viz2d.cm_prune(self.matches01["prune1"])
        viz2d.plot_images([self.imageload1, self.imageload2])
        viz2d.plot_keypoints([self.kpts0, self.kpts1], colors=[kpc0, kpc1], ps=10)
        
        

def draw_matches(im1_path, im2_path, pts1_in, pts2_in, pts1_out=None, pts2_out=None, mask=None):
    img1 = cv2.imread(im1_path)
    img2 = cv2.imread(im2_path)
    
    # img1=resize_max_width(img1, 1000)
    # img2=resize_max_width(img2, 1000)
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    
    # Diminuer la saturation (par exemple, la réduire de 50 %)
    h, s, v = cv2.split(img1)
    s = np.clip(s * 0.2, 0, 255).astype(np.uint8)  # Réduction à 50 %
    
    # Fusionner et reconvertir en BGR
    hsv_modified = cv2.merge([h, s, v])
    img1 = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)
    
    h, s, v = cv2.split(img2)
    s = np.clip(s * 0.2, 0, 255).astype(np.uint8)  # Réduction à 50 %
    
    # Fusionner et reconvertir en BGR
    hsv_modified = cv2.merge([h, s, v])
    img2 = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)
    
    # Vérifier que les deux images ont la même hauteur
    # if img1.shape[0] != img2.shape[0]:
    new_height = min(img1.shape[0], img2.shape[0], 750)
    img1_re = cv2.resize(img1, (int(img1.shape[1] * new_height / img1.shape[0]), new_height))
    img2_re = cv2.resize(img2, (int(img2.shape[1] * new_height / img2.shape[0]), new_height))
    espace_blanc = 255 * np.ones((new_height, 10, 3), dtype=np.uint8)
    # Concaténation horizontale
    img_matches = np.hstack((img1_re,espace_blanc, img2_re))

    # Décalage x pour les points de la 2e image
    offset = int(img1_re.shape[1])+espace_blanc.shape[1]


        
    for pt1, pt2 in zip(pts1_in, pts2_in):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        pt1 = tuple(np.round(pt1*new_height / img1.shape[0]).astype(int))
        pt2 = tuple(np.round(pt2*new_height / img2.shape[0]).astype(int))
        pt2_offset = (pt2[0] + offset, pt2[1])

        cv2.circle(img_matches, pt1, 5, (0,255,0), -1)
        cv2.circle(img_matches, pt2_offset, 5, (0,255,0), -1)
        cv2.line(img_matches, pt1, pt2_offset, (0,255,0), 2)
    
    if pts1_out is not None and pts2_out is not None:
        for pt1, pt2 in zip(pts1_out, pts2_out):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            pt1 = tuple(np.round(pt1*new_height / img1.shape[0]).astype(int))
            pt2 = tuple(np.round(pt2*new_height / img2.shape[0]).astype(int))
            pt2_offset = (pt2[0] + offset, pt2[1])
    
            cv2.circle(img_matches, pt1, 8, (0,255,0), -1)
            cv2.circle(img_matches, pt2_offset, 8, (0,255,0), -1)
            # cv2.line(img_matches, pt1, pt2_offset, (0,0,255), 4)
    # img_array=np.asarray(img)

    name1 = os.path.splitext(os.path.basename(im1_path))[0]
    name2 = os.path.splitext(os.path.basename(im2_path))[0]
    cv2.imwrite("Ouput/matches_"+name1+"_"+name2+".jpg", img_matches)  
    
    h=img_matches.shape[0]
    w=img_matches.shape[1]
    height_in_inches = 4
    aspect_ratio = w / h
    width_in_inches = height_in_inches * aspect_ratio
    fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches))
    ax.imshow(img_matches)
    ax.axis('off') 

def resize_max_width(image: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image  # Pas besoin de redimensionner

    scale = max_width / w
    new_w = int(max_width)
    new_h = int(round(h * scale))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized
