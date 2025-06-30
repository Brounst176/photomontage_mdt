# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 17:21:09 2025

@author: Bruno
"""
import numpy as np
from numba import njit, jit #OPTIMISATION DES CALCULS
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
import module_python.mesh_module as mm
# import warnings
# warnings.simplefilter("ignore", category=NumbaWarning)


# @njit
def initialisation_projet_sur_image(nbre_de_calcul, i, pourc_prec,u,v,fact_reduce_depthajuste, fact,depthmap_IA,depthmap_IA_backup):
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
    uv=np.array([u/fact_reduce_depthajuste,v/fact_reduce_depthajuste])
    depth_IA_value=depthmap_IA[v_IA, u_IA]
    
    depth_IA_value_origine=depthmap_IA_backup[v_IA, u_IA]
    
    return depth_IA_value,depth_IA_value_origine, uv,pourc_prec,u_IA,v_IA

@njit
def calcul_value_prov(profondeur_valeur, depth_IA_value):
    deptha=profondeur_valeur[:, 4][:, np.newaxis]
    deptha=depth_IA_value-deptha
    
    array_prof=np.append(profondeur_valeur,deptha, axis=1)
    array_prof = array_prof[array_prof[:, 5].argsort()]
    
    mask=array_prof[:, 5]>0
    mask_array=array_prof[mask]
    value_prov=mask_array[0,4]
    return value_prov
# optimisée_pour_jit

@njit
def uv_to_M_by_dist_prof(S,R, uv, dist_proj, k1,k2,k3,k4,p1,p2,b1,b2, w, h,cx,cy,f):
    
    mx=(uv[0]-w/2-cx)/(f+b1+b2)
    my=(uv[1]-h/2-cy)/f
    m_prime=np.array([mx,my, 0.0], dtype=np.float64)

    
    
    x=m_prime[0]
    y=m_prime[1]
    for i in range(100):
        r=(x**2+y**2)**0.5
        x_prime = x*(1+k1*r**2+k2*r**4+k3*r**6+k4*r**8)+(p1*(r**2+2*x**2)+2*p2*x*y)
        y_prime = y*(1+k1*r**2+k2*r**4+k3*r**6+k4*r**8)+(p2*(r**2+2*y**2)+2*p1*x*y)
        u=w*0.5+cx+x_prime*f+x_prime*b1+y_prime*b2
        v=h*0.5+cy+y_prime*f
        uv_prov=np.array([u,v],dtype=np.float64)
        diff_uv=uv-uv_prov
        # print(i)
        if abs(diff_uv[0])>0.3 or abs(diff_uv[1])>0.3:
            x+=diff_uv[0]/4000
            y+=diff_uv[1]/4000
        else:
            break
    X=-f*x
    Y=f*y
    F=np.array([0,0,-f])
    m=np.array([X,Y,0])

    M_prime=np.dot(R.T,(m-F))
    # d_M_prime=np.linalg.norm(M_prime)
    # d_h=M_prime[2]
    # N=self.vect_dir_camera(photoname)
    
    fact=dist_proj/f
    M=S-fact*M_prime
    return M, M-S
@njit
def distorsion_frame_brown_agisoft_from_xy(x, y, k1,k2,k3,k4,p1,p2,b1,b2, w, h,cx,cy,f):

    r=(x**2+y**2)**0.5
    x_prime = x*(1+k1*r**2+k2*r**4+k3*r**6+k4*r**8)+(p1*(r**2+2*x**2)+2*p2*x*y)
    y_prime = y*(1+k1*r**2+k2*r**4+k3*r**6+k4*r**8)+(p2*(r**2+2*y**2)+2*p1*x*y)
    u=w*0.5+cx+x_prime*f+x_prime*b1+y_prime*b2
    v=h*0.5+cy+y_prime*f
    uv=np.array([u,v],dtype=np.float64)
    return uv

@njit
def init_calcul_intersection(boundaries, fact_reduce_depthajuste :np.float64,S,R, k1,k2,k3,k4,p1,p2,b1,b2, w, h,cx,cy,f):
    pixel_coords = []
    directions_list = []
    closest_rays = {}
    for v in range(boundaries[1,0],boundaries[1,1]):
        for u in  range(boundaries[0,0],boundaries[0,1]):
            uv=np.array([u/fact_reduce_depthajuste,v/fact_reduce_depthajuste], dtype=np.float64)
            M,vect = uv_to_M_by_dist_prof(S,R, uv, 120.0, k1,k2,k3,k4,p1,p2,b1,b2, w, h,cx,cy,f)
            vect = vect / np.linalg.norm(vect)
            directions_list.append(vect)
            pixel_coords.append((v, u))
            vect = vect / np.linalg.norm(vect)
            closest_rays[(v, u)] = {'vect': vect}
            
    return pixel_coords, directions_list, closest_rays

@njit
def prep_donnee_intersection(locs, ray_ids, tri_ids, origins, directions_list):
    

    # locs, ray_ids, tri_ids = mesh.ray.intersects_location(origins, directions)

    deltas = locs - origins[ray_ids]
    distances2 = (deltas * deltas).sum(axis=1)
    closest_dist = np.full(len(directions_list), np.inf)
    closest_hit_idx = np.full(len(directions_list), -1, dtype=np.int32)

    for i in range(len(ray_ids)):
        r_id = ray_ids[i]
        if distances2[i] < closest_dist[r_id]:
            closest_dist[r_id] = distances2[i]
            closest_hit_idx[r_id] = i
    return closest_dist, closest_hit_idx
@njit
def calcul_proj_cam(S, R, M):
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
    P=M
    PS=P-S
    # print(PS)

    N=np.dot(R.T,np.array([0.0,0.0,-1.0]))
    # print(R)
    # print(N)
    norm_N=np.linalg.norm(N)
    norm_PS=np.linalg.norm(PS)
    H=P-(PS@N)/(norm_PS*norm_N)*norm_PS/norm_N*N
    norm_PH=np.abs((PS@N)/(norm_PS*norm_N)*norm_PS)
    return H, norm_PH

@njit
def return_array_epurer_from(u_IA, v_IA, value_depth, array_prof, larg_image):
    
    
    uv_decalage=larg_image//54


    u_min = u_IA - uv_decalage
    u_max = u_IA + uv_decalage + 1
    mask_u = (array_prof[:, 0] >= u_min) & (array_prof[:, 0] < u_max)
    array_prof = array_prof[mask_u]
    



    
    # plot.plot_from_liste_prof(liste_prof)
    
    # print(f"Valeur depthIA: {value_depth} et ")

    if array_prof.shape[0]< 5:
        return None
    
    diff_depth=np.abs(array_prof[:,4]-value_depth)

    array_prof=np.column_stack((array_prof, diff_depth))

    array_prof=array_prof[array_prof[:, 5].argsort()]


    mask_dist=array_prof[:,5]<1.5
    array_prof_15dm=array_prof[mask_dist]

    if array_prof_15dm.shape[0]< 5:
        return None

    

    
    
    return array_prof_15dm

def dbscan_non_optimise(array_prof_15dm):
    X=array_prof_15dm[:,[2, 4]]
    neigh = NearestNeighbors(n_neighbors=4)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,4-1]
    # plt.plot(distances);
    # mean=np.mean(distances)
    pourc90=int(distances.shape[0]*0.90)

    y_pred = DBSCAN(eps = distances[pourc90], min_samples=2).fit_predict(X)
    unique_clusters = np.unique(y_pred)
    unique_clusters=np.delete(unique_clusters, 0)
    
    # clusters = [X[y_pred == cluster_id] for cluster_id in unique_clusters]
    
    
    unique_label=np.unique(y_pred)
    
    return y_pred, unique_label
    # x, color=pcd.array_et_colors_set_from_clusters(array_prof_15dm, y_pred)
    # plt.scatter(x[:,2], x[:,4], c=color)
    # plt.show()
    # plt.close()
@njit
def return_array_calcul_moindre_carre(y_pred, unique_label, value_depth, array_prof_15dm):
    liste_label_include=[]
    for i in unique_label:

        if i!=-1:
            
            mask=y_pred==i
            array_mask=array_prof_15dm[mask]
            # inliers, outlier=self.ransac_simple(array_mask.tolist(), min_samples=2)
            if np.min(array_mask[:,2])<1.5*value_depth:
                if np.min(array_mask[:,4])<value_depth and np.max(array_mask[:,4])>value_depth:
                    liste_label_include.append(i)

    if len(liste_label_include)==1:
        mask=y_pred==liste_label_include[0]

        array_res=array_prof_15dm[mask]

        if array_res.shape[0]>4:
            return array_res[:, 0:5]
        else:
            return None

    return None
@njit
def dict_prof_TO_array(dict_prof):
    liste_prof=[]
    for u, value in dict_prof.items():
        for v, value2 in value.items():
            prof_uv=np.array([u, v, value2["d"], value2["sigma"], value2["depthIA"], value2["point"], value2["normal"]], dtype=np.float64)
            liste_prof.append(prof_uv)
    array_prof=np.array(liste_prof)[:,[0,1,2,3,4]]
    array_prof=np.array(array_prof, dtype=np.float64)
    return array_prof


@njit
def gauss_markov(profondeur_valeur,sigma0=1, robuste=False, iter_robuste=10, delta=2.5, print_res=False):
    nb_l=profondeur_valeur.shape[0]
    B = np.asarray(profondeur_valeur[:,2][:, np.newaxis], dtype=np.float64)
    X=np.asarray(profondeur_valeur[:,4][:, np.newaxis], dtype=np.float64)
    
    A=np.ones((nb_l,2))
    A[:,0]=X[:,0]
    Qll=np.eye(nb_l)*0.1
    nb_i=A.shape[1]
    nb_l=A.shape[0]
    
    P=np.linalg.inv(Qll)
    
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    P = np.ascontiguousarray(P)

    AT = np.ascontiguousarray(A.T)
    
    
    Qxx=np.linalg.inv(AT@P@A)
    inc=Qxx@AT@P@B
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

    
