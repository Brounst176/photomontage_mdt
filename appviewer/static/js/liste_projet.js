const polySource = new ol.source.Vector();
const pointSource = new ol.source.Vector();
const polyLayer = new ol.layer.Vector({
      source: polySource,
      visible: false
    });

const pointLayer = new ol.layer.Vector({
      source: pointSource
    });
map.addLayer(pointLayer);
map.addLayer(polyLayer);


map.on('moveend',  function(e) {
      zoom = map.getView().getZoom();
      if(zoom>13){
        pointLayer.setVisible(false);
        polyLayer.setVisible(true);
      }else{
        pointLayer.setVisible(true);
        polyLayer.setVisible(false);
      }
});


fetch('/api/points')
.then(response => response.json())
.then(data => {
  for (const key in data) {
    const item = data[key];

    // Convertir le WKT en géométrie OL
    const geometrypt = wktFormat.readGeometry(item.coordinate, {
      dataProjection: 'EPSG:2056', // ta projection source
      featureProjection: 'EPSG:2056' // projection de la carte
    });

    const featurept = new ol.Feature({
      geometry: geometrypt,
      name: key
    });

    featurept.setStyle(new ol.style.Style({
      image: new ol.style.Circle({
        radius: 10,
        fill: new ol.style.Fill({ color: 'red' }),
        stroke: new ol.style.Stroke({ color: 'white', width: 2 })
      }),
      text: new ol.style.Text({
        text: key,
        offsetY: -20,
        fill: new ol.style.Fill({ color: 'black' }),
        stroke: new ol.style.Stroke({ color: 'white', width: 6 }),
    font: 'bold 14px sans-serif',
      })
    }));


    const geometrypoly = wktFormat.readGeometry(item.projet, {
      dataProjection: 'EPSG:2056', // ta projection source
      featureProjection: 'EPSG:2056' // projection de la carte
    });

    const featurepoly = new ol.Feature({
      geometry: geometrypoly,
      name: key
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

    polySource.addFeature(featurepoly);
    pointSource.addFeature(featurept);




  }
});


$('.projet_li').click(function(){
  var projetname = $(this).attr('id');

  active_projet(projetname)
  console.log("Nom du projet cliqué :", projetname);
  $.get('api/points', { id: projetname }, function(data) {
          zoomToProjet(projetname, data)
          console.log(data[projetname])
          create_projet_view(projetname)
          
      });
}); 


map.on('singleclick', function(evt) {
      let found = false

      map.forEachFeatureAtPixel(evt.pixel, function(feature) {
        const projetname = feature.get('name');
        console.log(projetname)
        active_projet(projetname)
        $.get('api/points', { id: projetname }, function(data) {
          zoomToProjet(projetname, data)
          console.log(data[projetname])
          create_projet_view(projetname)
          
        });
        found=true;

      });

      if (!found){
        $('.projet_li').removeClass('active');
        $('#projet_detail').empty()
      }
    });


function active_projet(projetname){
        $('.projet_li').removeClass('active');
        $("#"+projetname).addClass('active'); 
}

function create_projet_view(projetname){
  let imageUrl = "/images_projet/"+projetname;
          
  var titre = $('<h3>').text("Projet : "+projetname);
  var img = $('<img>').attr('src', imageUrl).attr('alt', 'Image chargée').attr('width', 500);


  const btnContainer = $('<div>').css({
    display: 'flex',
    'justify-content': 'center',
    'margin-top': '15px'
  });
  var btn = $('<button>').text('Ouvrir le projet').attr('id', 'button_ouvrerture').css({
    backgroundColor: '#333',
    color: '#eee',
    border: 'none',
    padding: '10px 20px',
    fontSize: '16px',
    borderRadius: '6px',
    marginBottom: '20px',
    cursor: 'pointer',
    transition: 'background-color 0.3s ease',
    boxShadow: '0 2px 5px rgba(0,0,0,0.5)'
  }).hover(
    function() { $(this).css('background-color', '#555'); },
    function() { $(this).css('background-color', '#333'); }
  ).on('click', function() {
      window.open('/projet/' + projetname);
  });

  btnContainer.append(btn);
  $('#projet_detail').empty().append(titre, img, btnContainer);
}
function zoomToProjet(id, json_data) {

    const item = json_data[id];
    if (!item) {
      console.warn("Projet non trouvé :", id);
      return;
    }

    // Lire la géométrie du point
    const geometry = wktFormat.readGeometry(item.coordinate, {
      dataProjection: 'EPSG:2056',
      featureProjection: 'EPSG:2056'
    });

    const center = geometry.getCoordinates(); // [x, y] en EPSG:2056
    let zoom = map.getView().getZoom();
    // Centrer la carte sur ce point
    if (isProjetnameVisibleFeatures(pointSource, id)){
      map.getView().animate({
      center: center,
      zoom: 14, // ou fit() si tu veux une emprise
      duration: 2000, // en millisecondes
    });
    }else{
      const centermap = map.getView().getCenter();

      map.getView().animate(
        { zoom: 7, duration: 2000, center: center},
        // { center: center, duration: 500 },
        { zoom: 14, duration: 2000 },
      );
    }

      
    

}

function isProjetnameVisibleFeatures(vectorSource, projetname) {
  const extent = map.getView().calculateExtent(map.getSize());
  const visibleFeatures = vectorSource.getFeaturesInExtent(extent);
  console.log(visibleFeatures)

  for (const feature of visibleFeatures) {
    const name = feature.get('name');
    if (name === projetname) {
      return true; // trouvé
    }
  }

  return false

}