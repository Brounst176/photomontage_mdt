import numpy as np
from scipy.spatial.transform import Rotation as Rot
import trimesh
import laspy
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
import open3d as o3d;
from sklearn import datasets, linear_model
import plot_module as pltbdc
import pointcloud_module as pcd


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
        x2=self.uv_to_M(photoname,np.array([0,nikon.h]),distance)
        x3=self.uv_to_M(photoname,np.array([nikon.w,nikon.h]),distance)
        x4=self.uv_to_M(photoname,np.array([nikon.w,0]),distance)
        
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

    
    
    def liste_coord_image_pt_homologue(self, pathlas, photoname, fact_reduce=0.5):
        """
        Créer un image avec les valeurs de profondeur depuis un nuage de points las
    
        pathlas: chemin du nuage de points las
        photoname: nom de la photo à tester
        fact_reduce: facteur de réduction de l'image
        
        Retourne une image array et une liste des valeurs de profondeurs
        """
        point_cloud=pcd.readlas_to_numpy(pathlas)
        point_cloud=pcd.remove_statistical_outlier_pcd(point_cloud)
        mesh=self.mesh_pyramide_camera_create(photoname, 250)
        
        points_inside=pcd.points_inside_mesh(point_cloud, mesh)
        
        # points_inside_reduce=pcd.reduction_nuage_nbre(points_inside)
        points_inside_reduce=pcd.reduction_nuage_voxel(points_inside,2)
        print(f"Nbre de points: {points_inside.shape[0]}")
        print(f"Nbre de points après réduction: {points_inside_reduce.shape[0]}")
       
        liste_uv_profondeur=[]
        img_uv_prof=np.zeros((int(self.h*fact_reduce), int(self.w*fact_reduce)))
        for point in points_inside_reduce:
            d=self.dist_MS(point, photoname)
            uv=self.M_to_uv(photoname, point)
            u=math.floor(uv[0]*fact_reduce)
            v=math.floor(uv[1]*fact_reduce)
            if u<int(self.w*fact_reduce) and v<int(self.h*fact_reduce):

                prof_uv=np.array([u, v, d, 0.1])
                liste_uv_profondeur.append(prof_uv)
                img_uv_prof[v, u]=d
                
        return liste_uv_profondeur, img_uv_prof
            
    def dist_MS(self, M, photoname):
        return np.linalg.norm(self.images[photoname]["S"]-M)
    
    # def read_profondeur_depthanything(self, pathDepth):
    #     depthmap=Image.open(pathDepth)
    #     return depthmap
    
    def ajuste_liste_observation_sur_UV_depthanything(self, depthmap_IA, liste_prof):
        nb_l=len(liste_prof)
        depth_array=np.array(depthmap_IA)
        for i in range(nb_l):
            obs=liste_prof[i]
            u=int(obs[0])
            v=int(obs[1])
            # d_proche=obs[3]
            x=depth_array[v,u]
            liste_prof[i]=np.append(liste_prof[i],[x])
            
            
        liste_prof_array=np.array(liste_prof)
        liste_prof_array=self.tri_array_par_rapport_a_une_colonne(liste_prof_array, 4)
        max_value_depthmap=np.max(liste_prof_array[:, 4])
        
        nb_intervalle=25
        intervalle=max_value_depthmap//nb_intervalle
        print(max_value_depthmap)
        
        liste_prof_ajustee=[]
        liste_prof_supprimer=[]
        means=[]
        
        
        
        for i in range(nb_intervalle):
            if nb_intervalle*(i+1)>(max_value_depthmap-nb_intervalle):
                masque = (liste_prof_array[:, 4] >= intervalle*i) & (liste_prof_array[:, 4] <= max_value_depthmap)
            else:
                masque = (liste_prof_array[:, 4] >= intervalle*i) & (liste_prof_array[:, 4] < intervalle*(i+1))
            sous_ensemble = liste_prof_array[masque]
            mean_local=np.mean(sous_ensemble[:,2])
            mean_std=np.std(sous_ensemble[:,2])
            mean=[]
            print(len(sous_ensemble))

            if len(means)>0:
                for j, value in enumerate(sous_ensemble):
                    if value[2]<means[-1] and value[4]!=0 and mean_local+2.5*mean_std>value[2]>mean_local-2.5*mean_std:
                        liste_prof_ajustee.append(value)
                        mean.append(value[2])
                    else:
                        liste_prof_supprimer.append(value)

            else:
                for j, value in enumerate(sous_ensemble):
                    if value[4]!=0 and mean_local+2.5*mean_std>value[2]>mean_local-2.5*mean_std:
                        liste_prof_ajustee.append(value)
                        mean.append(value[2])
                    else:
                        liste_prof_supprimer.append(value)
            if len(mean)>0:
                means.append(np.mean(mean))

        # CALCUL RANSAC
        # ================================

        liste_prof_array=np.array(liste_prof_ajustee)
        liste_ajustee_RANSAC=[]
        liste_supprimee_RANSAC=[]
        for i in range(nb_intervalle):
            if nb_intervalle*(i+1)>(max_value_depthmap-nb_intervalle):
                masque = (liste_prof_array[:, 4] >= intervalle*i) & (liste_prof_array[:, 4] <= max_value_depthmap)
            else:
                masque = (liste_prof_array[:, 4] >= intervalle*i) & (liste_prof_array[:, 4] < intervalle*(i+1))
            sous_ensemble = liste_prof_array[masque]
            X=sous_ensemble[:,4].reshape(-1, 1)
            Y=sous_ensemble[:,2].flatten()
            if len(sous_ensemble)>10:
            # Robustly fit linear model with RANSAC algorithm
                ransac = linear_model.RANSACRegressor()
                ransac.fit(X, Y)
                inlier_mask = ransac.inlier_mask_
                outlier_mask = np.logical_not(inlier_mask)
                inlier_indices = np.where(inlier_mask)[0]
                outlier_indices = np.where(outlier_mask)[0]
                inliers = sous_ensemble[inlier_indices]
                outlier = sous_ensemble[inlier_indices]
                liste_ajustee_RANSAC=liste_ajustee_RANSAC+[row for row in inliers]
                liste_supprimee_RANSAC.append(inliers)
            else:
                liste_ajustee_RANSAC=liste_ajustee_RANSAC+[row for row in sous_ensemble]
                    
            

        return liste_ajustee_RANSAC, liste_prof_supprimer


        
    def tri_array_par_rapport_a_une_colonne(self, array, colonne):
        liste_prof_array_tri = array[array[:, colonne].argsort()]
        return liste_prof_array_tri
    
    def tranformation_depthanything_gauss(self, depthmap_IA, liste_prof, img_prof, rob=3.5):
        
        nbre_valeur=100
        nb_i=10
        nb_l=len(liste_prof)
        A=np.zeros((nb_l,nb_i))
        B=np.zeros((nb_l,1))
        Qll=np.eye(nb_l)
        depth_array=np.array(depthmap_IA)
        X=np.zeros((nb_l,1))
        
        
        
        for i in range(nb_l):
            obs=liste_prof[i]
            u=int(obs[0])
            v=int(obs[1])
            B[i,0]=obs[2]
            sigma=obs[3]
            # d_proche=obs[3]
            x=depth_array[v,u]

            X[i,0]=x

            Qll[i,i]=sigma
            # elif  d_proche>(mean_d+std_d):
            #     print(f"Mesure {i} dépondérée car distance éloignée")
        #         Qll[i,i]=99999
            
            
            
            for j in range(nb_i):
                A[i, j]=x**(nb_i-1-j)
                
                
        inc, v, wi, B_calc=self.gauss_markov(Qll, A, B)
        
        
        for i in range(5):
            # print(i)
            inc_i=inc
            indices = np.argsort(np.abs(wi)[:, 0])[-5:][::-1]
            for indice in indices:
                
                if np.abs(wi[indice])>rob:
                    
                    # print(f"L'observation {indice} a une wi de {wi[indice]} et une v de {v[indice]}")
                    Qll[indice, indice]=9999
            inc, v, wi, B_calc=self.gauss_markov(Qll, A, B)

            if max(np.abs(inc-inc_i))[0]<0.005:
                break
            
        
                
        return A, B,Qll, inc, wi, v, X, B_calc
    
    def gauss_markov(self,Qll, A, B):
        
        nb_i=A.shape[1]
        nb_l=A.shape[0]
        
        P=np.linalg.inv(Qll)
        Qxx=np.linalg.inv(A.T@P@A)
        inc=Qxx@A.T@P@B
        v=A@inc-B
        s0=np.sqrt(v.T@P@v/(nb_l-nb_i))

        Qvv=Qll-A@Qxx@A.T
        wi=np.zeros((nb_l, 1))
        for i in range(nb_l):
            wi[i,0]=v[i,0]/(s0*np.sqrt(Qvv[i,i]))
            
        B_calc=A@inc
            
        return inc, v, wi, B_calc



class depthmap:
    def __init__(self, depthmap_IA, photoname, camera, max_megapixel=1.5,  IA_modele="depthanythingV2"):
        self.depthmap_IA=self.create_depthmap(depthmap_IA)
        self.IA_modele=IA_modele
        self.photoname=photoname
        self.camera=camera
        # self.fact_reduce=self.calc_fact_reduce(max_megapixel)
        self.fact_reduce=0.1
        
    def create_depthmap(self, pathDepth):
        depthmap=Image.open(pathDepth)
        return depthmap
        







nikon=camera("NIKON D7500", 5568, 3712, -55.72194146069377, 23.384728774951231,4231.8597199545738,-0.14699248766634521,0.1616307088544226,-0.023431362387691224,0.0,-0.00074350800851607905,0.00052372914373731753,0.0,0.0)


depthmap_DSC6987=depthmap("_DSC6987_556-371.tif", "_DSC6987", nikon)


fichier_path = 'camera_ORIENTAITON.txt'
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
list_uv_prof, img_prof=nikon.liste_coord_image_pt_homologue("point_homologue.las", "_DSC6987",0.1)
depthmap=nikon.read_profondeur_depthanything("_DSC6987_556-371.tif")



A,B, Qll, inc, wi, v, X,B_calc=nikon.tranformation_depthanything_gauss(depthmap, list_uv_prof, img_prof)
pltbdc.plot_mesure_calcule(X, B, B_calc)

list_uv_ajustee, liste_uv_supprimer=nikon.ajuste_liste_observation_sur_UV_depthanything(depthmap, list_uv_prof)
# list_uv_ajustee, liste_uv_supprimer=nikon.RANSAC_liste_observation_sur_UV_depthanything(depthmap,np.delete(list_uv_ajustee, 4, axis=1))
A,B, Qll, inc, wi, v, X,B_calc=nikon.tranformation_depthanything_gauss(depthmap, list_uv_ajustee, img_prof)
pltbdc.plot_mesure_calcule(X, B, B_calc)

# pltbdc.input_10_wi_to_image("_DSC7050_556-371.jpg",list_uv_prof,B,valeur, wi)
# nikon.suppression_point_isole("point_homologue.las")
# nikon.remove_statistical_outlier_pcd("point_homologue.las")
