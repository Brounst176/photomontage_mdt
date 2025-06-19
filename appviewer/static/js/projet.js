// Vecteur pour le marqueur
const imageSource = new ol.source.Vector();
const imageLayer = new ol.layer.Vector({ source: imageSource });

map.addLayer(imageLayer);

map.on('click', function (e) {
    // Récupérer les coordonnées en EPSG:2056 (car projection déjà définie)
    const coords = e.coordinate; // [east, north]
    const east = Math.round(coords[0]);
    const north = Math.round(coords[1]);

    console.log('Coordonnées cliquées (LV95):', east, north);
    update_position_image(east, north)

});




fetch('/api/points')
.then(response => response.json())
.then(data => {
  
    const item = data[projetname];


    const geometrypoly = wktFormat.readGeometry(item.projet, {
      dataProjection: 'EPSG:2056', // ta projection source
      featureProjection: 'EPSG:2056' // projection de la carte
    });

    const featurepoly = new ol.Feature({
      geometry: geometrypoly,
      name: item.objet
    });

    featurepoly.setStyle(new ol.style.Style({
      stroke: new ol.style.Stroke({
        color: 'black',
        width: 3,
      }),
      fill: new ol.style.Fill({
        color: 'rgba(255, 0, 0, 0.8)',
      })
    }));

    const polyLayer = new ol.layer.Vector({
      source: new ol.source.Vector({
        features: [featurepoly]
      }),
      visible: true
    });
    const geometry = wktFormat.readGeometry(item.coordinate, {
      dataProjection: 'EPSG:2056',
      featureProjection: 'EPSG:2056'
    });

    const center = geometry.getCoordinates();


    map.getView().animate({
      center: center,
      zoom: 14, // ou fit() si tu veux une emprise
      duration: 2000, // en millisecondes
    });
    map.addLayer(polyLayer);




});



 $('#imageInput').on('change', function () {
            const file = this.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('image', file);

            $.ajax({
                url: '/extract_exif',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function (response) {
                    console.log(response)
                    
                    const east = response["easting"];
                    const north = response["northing"];
                    if (east){
                        update_position_image(east, north)
                    }
                    
                    
                }
                 
      });
            });

function update_position_image(east, north){
    const feature = new ol.Feature({
        geometry: new ol.geom.Point([east, north])
    });

    // Style (optionnel)
    feature.setStyle(new ol.style.Style({
        image: new ol.style.Circle({
        radius: 6,
        fill: new ol.style.Fill({ color: 'red' }),
        stroke: new ol.style.Stroke({ color: 'white', width: 2 })
        })
    }));

    // Ajouter à la couche
    imageSource.clear();
    imageSource.addFeature(feature);
    const formatCoord = (value) => {
        return Math.round(value).toLocaleString('fr-CH'); // ou 'de-CH' pour aussi avoir l'apostrophe
    };

    $("#coordimage").val(formatCoord(east)+" / "+formatCoord(north));
    east_image=east
    north_image=north
}