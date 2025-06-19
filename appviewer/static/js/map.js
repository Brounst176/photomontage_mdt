
const wktFormat = new ol.format.WKT();

var resolutions = [
4000, 3750, 3500, 3250, 3000, 2750, 2500, 2250, 2000, 1750, 1500, 1250,
1000, 750, 650, 500, 250, 100, 50, 20, 10, 5, 2.5, 2, 1.5, 1, 0.5
];
ol.proj.get('EPSG:2056')
var extent = [2420000, 130000, 2900000, 1350000];
var projection = ol.proj.get('EPSG:2056');
projection.setExtent(extent);

var matrixIds = [];
for (var i = 0; i < resolutions.length; i++) {
matrixIds.push(i);
}


var wmtsCadastre = new ol.layer.Tile({
source: new ol.source.WMTS(({
layer: 'asitvd.fond_cadastral',
crossOrigin: 'anonymous',
url:'//wmts.asit-asso.ch/wmts/1.0.0/{Layer}/default/default/0/2056/{TileMatrix}/{TileRow}/{TileCol}.png',
tileGrid: new ol.tilegrid.WMTS({
  origin: [extent[0], extent[3]],
  resolutions: resolutions,
  matrixIds: matrixIds
}),
requestEncoding: 'REST'
})),
visible: false
});

var osm = new ol.layer.Tile({
source: new ol.source.OSM(),
visible: false
});
var view = new ol.View({
  center: [2519500, 1149600],
  projection: projection,
  zoom: 5,
  maxZoom: 16,
      minZoom:5
});

var map = new ol.Map({
target: "map", 
view: view,

});

map.addLayer(wmtsCadastre);
map.addLayer(osm);
wmtsCadastre.setVisible(true);

