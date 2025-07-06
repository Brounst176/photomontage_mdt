from flask import Flask, jsonify, Response, send_from_directory
from flask import render_template
from flask import request
from flask import url_for
import json
import os
import requests
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import io
import os
import sys
from datetime import datetime
import time
import numpy as np
import cv2
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import module_python.ori_photo_cacl_module as lg

import module_python.photomontage_module as ph
import module_python.plot_module as plot
import module_python.camera_module as cm




app = Flask(__name__)

# Config options - Make sure you created a 'config.py' file.
# app.config.from_object('flask_config')
# To get one variable, tape app.config['MY_VARIABLE']
base_dir = os.path.dirname(os.path.abspath(__file__))



@app.route('/')
def index():
        return render_template("accueil.html", active_page='accueil')

@app.route('/liste_projet')
def liste_projet():
        projets_json = load_projets()
        return render_template("liste_projet.html", active_page='liste_projet', projets=projets_json)

@app.route('/projet/<path:projetname>')
def projet_show(projetname):
        projets_json = load_projets()
        return render_template("projet.html", projetname=projetname, projet=projets_json[projetname])

@app.route('/source')
def source_show():
        return render_template("source.html", active_page='source')

@app.route('/tutoriel')
def tutoriel_show():
        return render_template("tutoriel.html", active_page='tutoriel')


@app.route("/calcul_photomontage", methods=["POST"])
def calcul_photomontage():
        modecalcul= request.form.get("modecalcul")
        projetname = request.form.get("projetname")
        est = request.form.get("est")
        print(est)
        nord = request.form.get("nord")
        print(nord)
        if "image" not in request.files:
                return "Aucune image", 400

        image = request.files["image"]
        if image.filename == "":
                return "Nom de fichier vide", 400

        

        # Générer le nom basé sur la date et l'heure
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        ext = os.path.splitext(image.filename)[1]  # ex: .jpg, .png
        filename = f"{timestamp}{ext}"
        
        folder = os.path.join("data_projet", projetname)
        os.makedirs(folder, exist_ok=True)

        
        # Chemin complet
        filepath = os.path.join("data_projet",projetname, "image",filename)

        # Enregistrer l'image
        image.save(filepath)




        return render_template("calcul_photomontage.html", projetname=projetname, filename=filename, est=est, nord=nord, modecalcul=modecalcul)

#CALCUL DU PHOTOMONTAGE
#=============================================
@app.route("/traitement_photomontage")
def traitement_photomontage():
        projetname = request.args.get("projetname")
        filename = request.args.get("filename")
        pathprojet = os.path.join("data_projet",projetname)
        pathimage= os.path.join("data_projet",projetname,"image",filename)
        pathoutput= os.path.join("data_projet",projetname,"output")
        image_sans_ext = os.path.splitext(filename)[0]
        est = request.args.get("est")
        nord = request.args.get("nord")
        modecalcul = request.args.get("modecalcul")
        print(est)
        x=float(est)
        y=float(nord)

        img = cv2.imread(pathimage)
        mio_pixel=img.shape[1]*img.shape[0]
        mio_seuil=float(modecalcul)
        fact=round(mio_seuil/mio_pixel, 1)
        
        print(f"pixeliamge : {mio_pixel}")
        print(f"seuilpixel : {mio_seuil}")
        if fact>1.0:
                fact=1.0
        elif fact<0.1:
               fact=0.1
        print(f"facteur de réduction : {fact}")
        

        position_approchee = np.array([x, y])

        def generate():
                yield sse_message("Démarrage du traitement")
                

                # Simuler lecture d'une image depuis dossier projet
                time.sleep(1)
                yield f"<p>Image importée sous le nom de {filename}</p>"
                

                #%% Importation des données de base et préparation des class de base
                #===================================================================================
                yield sse_message("<p>Chargement des données de référence</p>")
                modele_photogra=cm.camera("foldernotexite")
                calibliste=get_calib(projetname)
                for i in range(len(calibliste)):
                        modele_photogra.import_calib(calibliste[i], pathprojet+"/"+calibliste[i]+".xml")  #POUR BREMBLENS
                modele_photogra.import_image_from_omega_phi_kappa_file(pathprojet+"/position_orientation.txt")

                cacl_photo=lg.calcul_orientation_photo_homol(filename, position_approchee, modele_photogra, pathprojet, show_plot=False)
                time.sleep(1)
                # Étape 1 : redimensionner, détecter, etc.
                yield sse_message("<p>Recherche des points homologues (durée estimée à 5-10 minutes)</p>")
                cacl_photo.homol_array=None
                cacl_photo.image_traitee=[]
                liste_homol=cacl_photo.first_feats_matches_calc()

                dict_homol_filtre, array_homol_tri=cacl_photo.calcul_iteratif_homol_matches()

                dict_valide,dict_supprimer= cacl_photo.calcul_approximatif_homol_3D(dict_homol_filtre)
                uv=cacl_photo.dict_homol_to_uv()
                img=plot.show_only_point_in_image_pillow(pathimage, uv)
                plot.save_image(img, os.path.join(pathoutput, image_sans_ext+"_homol.jpg"))


                time.sleep(1)
                # Étape 2 : traitement IA ou autre
                yield sse_message("<img src='../image_output/"+projetname+"/"+image_sans_ext+"_homol.jpg' width='500'><p>Calcul de la position de l'image (durée estimée à 5 minutes)<p>")
                #POSITION APPROXIMATIVE PAR DLT
                cacl_photo.RANSAC_DLT(k=200, n_samples=6, nb_inliers=8)
                cacl_photo.distorsion_nulle()
                #PREMIERE DETERMINATION SIMPLE (f, S, orientation, k1) PAR MOINDRE CARRE NON LINEAIRE
                Qxx, x, A, B, wi, vi = cacl_photo.calcul_moindre_carre_non_lineaire(cacl_photo.dict_homol)

                #RECHERCHE DES POINTS FAUX PAR WI
                key_pt_homol_faux = cacl_photo.calcul_dictkey_point_faux_selon_wi(B, wi)

                #DEUXIEME DETERMINATION ("S", "angles", "f", "cx", "cy", "k1", "k2"1) PAR MOINDRE CARRE NON LINEAIRE
                cacl_photo.distorsion_nulle()
                liste_inc=["S", "angles", "f", "cx", "cy", "k1", "k2"]

                Qxx, x, A, B, wi, vi  =  cacl_photo.calcul_moindre_carre_non_lineaire(cacl_photo.dict_homol, key_pt_homol_faux, liste_inc)
                
                #CREATION DU MODEL CAMERA DU RESULAT
                image_cible=cm.camera("foldernot")
                image_cible.import_from_class_calc_photo_homol(cacl_photo)

                time.sleep(1)
                yield sse_message("<p>Création d'une carte de profondeur (durée estimée à 10 minutes)</p>")
                photomontage=ph.photomontage_depthmap(filename, pathprojet, image_cible, fact_reduce_IA=0.5, fact_reduce_photomontage=fact, show_plot=False)

                photomontage.import_projet_obj(os.path.join(pathprojet, "projet.obj"))
                photomontage.import_projet_emprise_obj(os.path.join(pathprojet, "emprise.obj"))
                # 
                photomontage.importer_image_origine(pathimage)
                photomontage.create_depthmap(pathimage)
                photomontage.export_image_from_deptharray(photomontage.depthmap_IA, photomontage.image, os.path.join(pathoutput, image_sans_ext+"_depth.png"), pred_only=True, grayscale=True)


                photomontage.depthmap_IA=photomontage.depthmap_IA_backup
                photomontage.dict_prof={}
                photomontage.liste_coord_image_pt_homologue(os.path.join(pathprojet, "pointcloud.las"), True, True,adapte_basique_prof=True)
                photomontage.transformation_simple_depthmap_IA()
                photomontage.transformation_seconde_depthmap_IA()
                time.sleep(1)
                yield sse_message("<img src='../image_output/"+projetname+"/"+image_sans_ext+"_depth.png' width='500'><p>Création du photomontage (durée pouvant aller jusqu'à 4h)</p>")
                time.sleep(2)
                image_montage, image_projet=photomontage.calcul_position_projet_sur_images()

                photomontage.save_image(image_montage, os.path.join(pathoutput, image_sans_ext+"_res.jpg") )
                photomontage.save_image(image_projet, os.path.join(pathoutput, image_sans_ext+"_projet.png") )
                
                
                yield sse_message("<div><img src='../image_output/"+projetname+"/"+image_sans_ext+"_res.jpg' width='500'></div><div><a href='../image_output/"+projetname+"/"+image_sans_ext+"_res.jpg' download='mon_image.jpg'><button>Télécharger le photomontage</button></a></div><p>Traitement terminé !</p>")

        return Response(generate(), mimetype='text/event-stream')
        




#API 
#=============================================
@app.route('/api/points', methods=['GET'])
def get_points():
        return jsonify(load_projets())


def get_calib(projetname):
        projetjson=load_projets()
        # print(projetjson)
        return projetjson[projetname]["calibration"]


@app.route('/images_projet/<path:projetname>')
def image_projet(projetname):
    image_dir = os.path.join('../data_projet', projetname)
    return send_from_directory(image_dir, "image.png")

@app.route('/image_output/<path:projetname>/<path:imagename>')
def imageoutput(projetname, imagename):
    print(imagename)
    image_dir = os.path.join('../data_projet', projetname, "output")
    return send_from_directory(image_dir, imagename)


@app.route('/extract_exif', methods=['POST'])
def extract_exif():
        if 'image' not in request.files:
                return jsonify({'error': 'Aucun fichier reçu'}), 400

        file = request.files['image']
        print(file)
        
        image = Image.open(io.BytesIO(file.read()))

#         # Extraction EXIF
        try: 
                exif_raw = image._getexif()
                print(exif_raw)
                if not exif_raw:
                        return jsonify({'exif': 'Aucune donnée EXIF trouvée'}), 200
                
                exif_data = {
                        ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()
                }

                gps_data = get_gps_info(exif_data)
                
                print(gps_data)

                url = "http://geodesy.geo.admin.ch/reframe/wgs84tolv95"
                params = {
                        "easting": gps_data["longitude"],
                        "northing": gps_data["latitude"],
                        "format": "json"
                }
                try:
                        response = requests.get(url, params=params)
                        if response.status_code == 200:
                                return response.json()
                        else:
                                return "Erreur API: {response.status_code}"
                except Exception as e:
                        return "Erreur lors de la transformation vers lv95"
        except:
               return 'Erreur lors du traitement de l\'image'



#fonction interne 
#=============================================
def load_projets():
    json_path = os.path.join(base_dir, 'projet.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_to_degrees(value):
        """Convertir les coordonnées GPS en degrés décimaux"""
        d = value[0]
        m = value[1]
        s = value[2]
        return d + (m / 60.0) + (s / 3600.0)

def get_gps_info(exif_data):
        gps_info = exif_data.get('GPSInfo')
        if not gps_info:
                return None

        gps_tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
        
        try:
                lat = convert_to_degrees(gps_tags['GPSLatitude'])
                if gps_tags['GPSLatitudeRef'] != 'N':
                        lat = -lat

                lon = convert_to_degrees(gps_tags['GPSLongitude'])
                if gps_tags['GPSLongitudeRef'] != 'E':
                        lon = -lon
                if 'GPSAltitude' in gps_tags:
                        alt=gps_tags['GPSAltitude']
                else:
                       alt=550.0
                

                return {
                        'latitude': lat,
                        'longitude': lon,
                        'altitude': alt
                }

        except Exception as e:
                return {'error': f'Erreur conversion GPS: {str(e)}'}

def sse_message(text):
    # Ajoute 1 Ko pour forcer le navigateur et Flask à flusher
    padding = ' ' * (1024 - len(text))
    return f"data: {text}{padding}\n\n"


if __name__ == "__main__":
        app.run()