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


#API 
#=============================================
@app.route('/api/points', methods=['GET'])
def get_points():
        return jsonify(load_projets())


@app.route('/images_projet/<path:projetname>')
def image_projet(projetname):
    image_dir = os.path.join('data_projet', projetname)
    return send_from_directory(image_dir, "image.png")

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


if __name__ == "__main__":
        app.run()