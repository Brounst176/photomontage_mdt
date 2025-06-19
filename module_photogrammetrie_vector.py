# -*- coding: utf-8 -*-
"""
Created on Thu May  8 09:41:06 2025

@author: Bruno
"""

import jax.numpy as jnp
from jax import grad, vmap
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import math as m

def euler_xyz_to_matrix(o, p, k, degrees=False):
    if degrees:
        o = jnp.deg2rad(o)
        p = jnp.deg2rad(p)
        k = jnp.deg2rad(k)

    # Rotation autour de x
    Rx = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(o), -jnp.sin(o)],
        [0, jnp.sin(o), jnp.cos(o)]
    ])

    # Rotation autour de y
    Ry = jnp.array([
        [jnp.cos(p), 0, jnp.sin(p)],
        [0, 1, 0],
        [-jnp.sin(p), 0, jnp.cos(p)]
    ])

    # Rotation autour de z
    Rz = jnp.array([
        [jnp.cos(k), -jnp.sin(k), 0],
        [jnp.sin(k), jnp.cos(k), 0],
        [0, 0, 1]
    ])

    # Composition dans l'ordre xyz (Rx, puis Ry, puis Rz)
    R = Rz @ Ry @ Rx
    return R


S=np.array(
    [2528509.7966,1159642.1095587259,508.2601241]
    )

S=np.array(
    [2528510.7966,1159641.1095587259,508.7601241]
    )
S_loc=np.array([0.0,0.0,0.0])
M=np.array([
    [2528528.609711709, 1159666.3519212431, 512.8080390118347],
    [2528520.861760504, 1159665.2991492173, 510.2013334170333],
    [2528528.0947989007, 1159666.4552612484, 512.3082388231593],
    [2528522.9029060146, 1159664.7783246497, 507.5207091760967],
    [2528528.956984558, 1159665.9539721806, 512.764439330739],
    [2528526.188823503, 1159667.2941347128, 515.9369783801958],
    [2528523.69872577, 1159664.338961658, 510.33355505991494],
    [2528520.17873636, 1159658.1719871066, 507.1788939386315],
    [2528520.4187243273, 1159665.759352405, 507.653248720173],
    [2528527.9572194684, 1159652.85639574, 511.3772392461542],
    [2528530.1676661465, 1159655.2417649552, 512.5239440286532],
    [2528518.6072680685, 1159658.7158823144, 509.29967886491795],
    [2528523.351814766, 1159667.0859236072, 510.94426615611883],
    [2528523.1019074526, 1159664.2351828357, 510.3448912698077],
    [2528525.2870314065, 1159667.664450203, 517.8651535001118],
    [2528524.027302912, 1159667.9162440263, 511.3696445178357],
    [2528527.0950037716, 1159667.0955820335, 516.5457714982331],
    [2528531.1971049868, 1159663.523660242, 513.3014825759658],
    [2528526.5598928714, 1159662.4629239317, 508.4350112959067],
    [2528529.4276935337, 1159655.7761971932, 511.6323350172024],
    [2528530.9086673153, 1159668.8707430824, 514.0155854839235],
    [2528524.4028515546, 1159664.4295598932, 509.4283577995666],
    [2528525.718021745, 1159667.4976680719, 516.0267837410187],
    [2528523.972729429, 1159668.2883574637, 515.9727682686644],
    [2528529.5417400305, 1159669.9008877606, 518.9980368026032],
    [2528523.8858369435, 1159668.1700197181, 514.3889947429705],
    [2528526.9762047306, 1159650.9468717952, 512.0681687323377],
    [2528526.8121228344, 1159667.3378914753, 514.4778210109798],
    [2528533.1400802396, 1159652.619514781, 511.7305089686997],
    [2528521.389252629, 1159666.0159216546, 509.63318403359153],
    [2528529.8466733997, 1159669.5940745752, 519.6816030326299],
    [2528529.439304976, 1159655.670497206, 510.45562339166645],
    [2528522.674359001, 1159668.95189554, 513.4264507450592],
    [2528529.75648957, 1159669.4628299146, 518.9717659286342],
    [2528526.708555349, 1159667.181084256, 516.0342406009696],
    [2528525.8893538, 1159667.479025648, 513.4418265830027],
    [2528561.908435935, 1159697.8557829633, 513.341009616619],
    [2528529.3316530865, 1159655.9975937, 510.5629285710165],
    [2528524.0506270453, 1159663.8526156612, 508.3208447598159],
    [2528528.4639015775, 1159666.158716334, 515.1967018884607],
    [2528524.750691877, 1159664.2641384015, 508.34303087124135],
    [2528524.971611336, 1159660.1147059025, 507.6860109088036],
    [2528516.4419419845, 1159655.3478764081, 507.1238914450805],
    [2528521.902167329, 1159669.578890692, 514.8060839843359],
    [2528532.8358031204, 1159663.1868840456, 507.79190895159627],
    [2528535.7114225514, 1159671.119924605, 507.21424224425573],
    [2528533.2529055355, 1159667.2616611123, 507.8923782877173],
    [2528525.0087858215, 1159668.136274857, 516.0823900530619],
    [2528522.5413421853, 1159665.5596106388, 507.77615604380844],
    [2528527.9633309077, 1159666.0364734288, 514.7281146133319],
    [2528526.976030168, 1159663.9265813213, 508.44692965134163],
    [2528521.646161238, 1159659.3256788547, 506.9076745514758],
    [2528522.5072443, 1159664.3234341578, 508.4396353866905],
    [2528511.285689549, 1159665.8136223608, 514.0018525085179]
    ])
M_loc=np.array(M-S)

uv=np.array([
    [3084.1996681257297, 1615.4549139763096],
    [2183.2104434920034, 1944.2185005643132],
    [3021.3791655675777, 1681.6961845492256],
    [2531.0198587095406, 2381.4508896373673],
    [3155.9215400753856, 1616.5447132618547],
    [2723.8815377474075, 1186.964223770036],
    [2666.640535422305, 1918.933464909123],
    [2747.2382178214893, 2497.390470166455],
    [2089.9732642336876, 2373.3056797414856],
    [4772.647717318814, 1504.3186947700224],
    [4575.308313815577, 1396.181986532822],
    [2366.9147792439558, 2027.064391060273],
    [2403.8708884026373, 1860.0011750452713],
    [2592.4734991777655, 1913.7752357831612],
    [2585.93284104834, 925.6169011888629],
    [2431.4074918964843, 1812.7675492117078],
    [2840.225564228095, 1114.2765495391045],
    [3613.812348604144, 1512.5945143040506],
    [3231.431743017814, 2210.464911586997],
    [4418.198504882841, 1565.3376637555834],
    [3114.608449672211, 1528.31504644341],
    [2758.1654384593626, 2066.1525056336827],
    [2653.6472257166884, 1173.696116490962],
    [2388.8245652242285, 1177.9015459932878],
    [2885.52022460583, 943.352674665927],
    [2389.015189303985, 1391.071535307321],
    [5051.830839275466, 1248.064896446866],
    [2797.457751194545, 1393.3157963128178],
    [5355.123520784413, 1504.0073684841618],
    [2211.8110958045613, 2045.7668290666102],
    [2934.5069154301527, 859.1993444220285],
    [4453.0045766713465, 1781.2557153473872],
    [2187.1754113504708, 1530.7315786045065],
    [2937.4365833090287, 935.9443755508674],
    [2791.824317301477, 1178.6616163218823],
    [2683.5125014925975, 1527.6502394732847],
    [3490.7988091575653, 1943.5747176750601],
    [4386.360303565399, 1767.3221442255494],
    [2766.9990171433687, 2243.084259258273],
    [3072.5467145803445, 1287.0318501359272],
    [2824.340569821569, 2238.1289046597453],
    [3285.941731061722, 2342.6484273974515],
    [2281.8272611724565, 2599.9427437796453],
    [2049.079945131245, 1345.6678689856287],
    [3853.341516375241, 2284.3930579575313],
    [3411.3361510062978, 2349.803799090139],
    [3501.5013127671045, 2276.698167192566],
    [2524.3542720054284, 1174.805307383017],
    [2416.7145789888978, 2341.100455804317],
    [3029.8256881626944, 1340.1894125769118],
    [3137.13867199832, 2213.0649836989833],
    [2872.8956666629533, 2529.1317490280458],
    [2507.9260152337197, 2230.2658818405807],
    [434.3227856805797, 1166.2104679878325]
    ])


# array_alea = np.random.uniform(low=-5, high=5, size=(54, 2))

# uv=uv+array_alea
w=5568
h=3712
R=jnp.array([[ 0.81983357, -0.57172788, -0.03162513],
       [-0.04335696, -0.1170544 ,  0.99217863],
       [-0.57095804, -0.81205018, -0.12075351]])
o = -98.4580197561788282*m.pi/180
p=34.8170601*m.pi/180
k=-3.02727*m.pi/180
f=4249.599
cx=105.21
cy=-122.60
k1=-0.09336646508547678
k2=0.19393583737559544
k3=-0.3177858547519
k4=0.0
p1=0.0085400029854600205
p2=-0.00210701549061142
b1=0.0
b2=0.0


o = -101.4580197561788282*m.pi/180
p=37.8170601*m.pi/180
k=-2.02727*m.pi/180
f=3000.599
cx=0.0
cy=0.0
k1=0.0
k2=0.0
k3=0.0
# k4=0.0
p1=0.0
p2=0.0
# b1=0.0
# b2=0.0
# r = Rot.from_euler('xyz', [o, p, k])
#  # return rot_z(k)@rot_y(p)@rot_x(o)
# R_scipy= r.as_matrix()

# R_tst=np.array(euler_xyz_to_matrix(o, p, k))

def photogra_xm(Sx,Sy,Sz, M, o, p, k, f, cx, cy, k1, k2, k3, k4, p1,p2,b1,b2, Rinit=None):
    S=jnp.array([
        Sx,
        Sy,
        Sz
        ])
    
    R=euler_xyz_to_matrix(o,p,k)
    
    if Rinit is not None:
        R=Rinit
    
    V=M-S
    # print(V)
    F=jnp.array([0,0,-f])

    rms=jnp.dot(R,V)

    m_cam=F-F[2]*rms/rms[2]
    # print(m_cam)
    X=-m_cam[0]
    Y=m_cam[1]
    
    
    Z=f
    x=X/Z

    y=Y/Z
    
    r=(x**2+y**2)**0.5
    x_prime = x*(1+k1*r**2+k2*r**4+k3*r**6+k4*r**8)+(p1*(r**2+2*x**2)+2*p2*x*y)
    y_prime = y*(1+k1*r**2+k2*r**4+k3*r**6+k4*r**8)+(p2*(r**2+2*y**2)+2*p1*x*y)
    ucam=w*0.5+cx+x_prime*f+x_prime*b1+y_prime*b2
    # vcam=cy+y_prime*f
    
    return ucam

def photogra_ym(Sx,Sy,Sz, M, o, p, k, f, cx, cy, k1, k2, k3, k4, p1,p2,b1,b2):
    S=jnp.array([
        Sx,
        Sy,
        Sz
        ])
    R=euler_xyz_to_matrix(o,p,k)
    
    V=M-S

    F=jnp.array([0,0,-f])

    rms=jnp.dot(R,V)

    m_cam=F-F[2]*rms/rms[2]
    
    X=-m_cam[0]
    Y=m_cam[1]
    
    
    Z=f
    x=X/Z

    y=Y/Z
    
    r=(x**2+y**2)**0.5
    # x_prime = x*(1+k1*r**2+k2*r**4+k3*r**6+k4*r**8)+(p1*(r**2+2*x**2)+2*p2*x*y)
    y_prime = y*(1+k1*r**2+k2*r**4+k3*r**6+k4*r**8)+(p2*(r**2+2*y**2)+2*p1*x*y)
    # ucam=cx+x_prime*f+x_prime*b1+y_prime*b2
    vcam=h*0.5+cy+y_prime*f
    
    return vcam

def gauss_markov_non_lineaire_photog(uv,w, h, S, M, o, p, k, f, cx, cy, k1, k2, k3, k4, p1,p2,b1,b2, liste_inc=["S", "angles", "f", "k1"]):
    
    dict_inc={
        "S": {"num": [0,1,2], "val":  [0.0,0.0,0.0]},
        "angles": {"num": [4,5,6], "val": [o,p,k]},
        "cx": {"num": [8], "val": [cx]},
        "cy": {"num": [9], "val": [cy]},
        "f": {"num": [7], "val": [f]},
        "k1":{"num": [10], "val": [k1]},
        "k2":{"num": [11], "val": [k2]},
        "k3":{"num": [12], "val": [k3]},
        "k4":{"num": [13], "val": [k4]},
        "p1":{"num": [14], "val": [p1]},
        "p2":{"num": [15], "val": [p2]},
        "b1":{"num": [16], "val": [b1]},
        "b2":{"num": [17], "val": [b2]}
        }
    

    
    dec=S
    S_loc=jnp.array([0.0,0.0,0.0])
    
    num_param_inc=[]
    
    for inc_name in liste_inc:
        for j in range(len(dict_inc[inc_name]["num"])):
            num_param_inc.append(dict_inc[inc_name]["num"][j])
    
    num_param_inc.sort()
    x=np.zeros((len(num_param_inc),1))
    
    index_x=0
    for index in range(len(num_param_inc)):
        for key, val in dict_inc.items():
            for j in range(len(val["num"])):
                if val["num"][j]==num_param_inc[index]:
                    x[index, 0]=val["val"][j]
                    break
                    
            
    


    grad_xm   = grad(photogra_xm, argnums=(num_param_inc)) 
    grad_xm_batched = vmap(grad_xm, in_axes=(None,None,None,0, None, None, None, None, None,None, None, None, None, None,None, None, None, None))

    Axm=grad_xm_batched(S_loc[0],S_loc[1],S_loc[2], M, o, p, k, f, cx, cy, k1, k2, k3, k4, p1,p2,b1,b2)
    Axm_numpy=np.array(Axm).T

    grad_ym   = grad(photogra_ym, argnums=(num_param_inc)) 
    grad_ym_batched = vmap(grad_ym, in_axes=(None,None,None,0, None, None, None, None, None,None, None, None, None, None,None, None, None, None))

    Aym=grad_ym_batched(S_loc[0],S_loc[1],S_loc[2], M, o, p, k, f, cx, cy, k1, k2, k3, k4, p1,p2,b1,b2)
    Aym_numpy=np.array(Aym).T
    

    A= np.concatenate((Axm_numpy, Aym_numpy), axis=0)
    
    B = np.vstack((uv[:, 0], uv[:, 1])).reshape(-1, 1)
    
    
    Qll=np.eye(B.shape[0])
    P=np.linalg.inv(Qll)
    Qxx=np.linalg.inv(A.T@P@A)
    l=np.zeros((B.shape[0],1))
    
    
    nb_obs=int(M.shape[0])
    for i in range(nb_obs):
        l[i,0]=photogra_xm(S_loc[0],S_loc[1],S_loc[2], M[i,:], o, p, k, f, cx, cy, k1, k2, k3, k4, p1,p2,b1,b2)
        l[i+nb_obs,0]=photogra_ym(S_loc[0],S_loc[1],S_loc[2], M[i,:], o, p, k, f, cx, cy, k1, k2, k3, k4, p1,p2,b1,b2)
    
    

    
    for j in range(50):
        Qxx=np.linalg.inv(A.T@P@A)
        dl=B-l
        dx=Qxx@A.T@P@dl

        x=x+dx
        
        for index in range(len(num_param_inc)):
            for key, val in dict_inc.items():
                for j in range(len(val["num"])):
                    if val["num"][j]==num_param_inc[index]:
                        dict_inc[key]["val"][j]=x[index, 0]
                        break
        Sx=dict_inc["S"]["val"][0]
        Sy=dict_inc["S"]["val"][1]
        Sz=dict_inc["S"]["val"][2]
        o = dict_inc["angles"]["val"][0]
        p=dict_inc["angles"]["val"][1]
        k=dict_inc["angles"]["val"][2]
        f=dict_inc["f"]["val"][0]
        cx=dict_inc["cx"]["val"][0]
        cy=dict_inc["cy"]["val"][0]
        k1=dict_inc["k1"]["val"][0]
        k2=dict_inc["k2"]["val"][0]
        k3=dict_inc["k3"]["val"][0]
        k4=dict_inc["k4"]["val"][0]
        p1=dict_inc["p1"]["val"][0]
        p2=dict_inc["p2"]["val"][0]
        b1=dict_inc["b1"]["val"][0]
        b2=dict_inc["b2"]["val"][0]
        
        if np.max(np.abs(dx[0:6]))<0.00005 and np.max(np.abs(dx[6:9]))<0.01:
            print(f"Convergence à l'itération {j}")
            break
        
        for i in range(nb_obs):
            l[i,0]=photogra_xm(Sx,Sy,Sz, M[i,:], o, p, k, f, cx, cy, k1, k2, k3, k4, p1,p2,b1,b2)
            l[i+nb_obs,0]=photogra_ym(Sx,Sy,Sz, M[i,:], o, p, k, f, cx, cy, k1, k2, k3, k4, p1,p2,b1,b2)
        
    #     # grad_xm   = grad(photogra_xm, argnums=(0,1,2,4,5,6,7,8,9,10,11,12,14,15)) 
    #     # grad_xm_batched = vmap(grad_xm, in_axes=(None,None,None,0, None, None, None, None, None,None, None, None, None, None,None, None, None, None))

        Axm=grad_xm_batched(Sx, Sy, Sz, M, o, p, k, f, cx, cy, k1, k2, k3, k4, p1,p2,b1,b2)
        Axm_numpy=np.array(Axm).T

    #     # grad_ym   = grad(photogra_ym, argnums=(0,1,2,4,5,6,7,8,9,10,11,12,14,15)) 
        # grad_ym_batched = vmap(grad_ym, in_axes=(None,None,None,0, None, None, None, None, None,None, None, None, None, None,None, None, None, None))

        Aym=grad_ym_batched(Sx, Sy, Sz, M, o, p, k, f, cx, cy, k1, k2, k3, k4, p1,p2,b1,b2)
        Aym_numpy=np.array(Aym).T
        
        A= np.concatenate((Axm_numpy, Aym_numpy), axis=0)
    vi=-dl

    s0=np.sqrt(vi.T@P@vi/(dl.shape[0]-x.shape[0]))
    

        
    # vi=-dl
    Qvv=Qll-A@Qxx@A.T
    
    wi=np.zeros((vi.shape[0],1))
    
    
    # s0=np.sqrt(vi.T@P@vi/(dl.shape[0]-x.shape[0]))
    

    for i in range(vi.shape[0]):
        wi[i,0]=vi[i,0]/(s0[0,0] *np.sqrt(Qvv[i,i]))
    print("RESULTAT")
    print("=======================")
    print(f"S={Sx+S[0]} / {Sy+S[1]} / {Sz+S[2]}")
    print(f"omega / phi / kappa={x[3,0]*180/m.pi} / {x[4,0]*180/m.pi} / {x[5,0]*180/m.pi}")
    print(f"f={f}")
    print(f"cx={cx}")
    print(f"cy={cy}")
    print(f"k1={k1}")
    print(f"k2={k2}")
    print(f"k3={k3}")
    print(f"k4={k4}")
    print(f"p1={p1}")
    print(f"p2={p2}")
    print("ANALYSE")
    print("=======================")
    print(f"s0={s0}")


    res ={
        "S": np.array([[Sx+S[0]], [Sy+S[1]], [Sz+S[2]]]),
        "angles": [x[3,0]*180/m.pi,x[4,0]*180/m.pi,x[5,0]*180/m.pi],
        "cx": cx,
        "cy": cy,
        "f": f,
        "f": f,
        "k1":k1,
        "k2":k2,
        "k3":k3,
        "k4":k4,
        "p1":p1,
        "p2":p2,
        "b1":b1,
        "b2":b2,
        }
    return Qxx, x, A, dx, B, dl, wi, vi, res
if __name__ == '__main__':
    liste_inc=["S", "angles", "f", "cx", "cy", "k1", "k2", "k3", "p1", "p2"]
    Qxx, x, A, dx, B, dl, wi, res=gauss_markov_non_lineaire_photog(uv,w, h, S, M_loc, o, p, k, f, cx, cy, k1, k2, k3, k4, p1,p2,b1,b2)
    


