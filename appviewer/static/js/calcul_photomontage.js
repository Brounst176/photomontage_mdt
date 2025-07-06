console.log(projetname)

function lancerTraitement() {
  const source = new EventSource('/traitement_photomontage?projetname='+projetname+'&filename='+filename+"&est="+est+"&nord="+nord+"&modecalcul="+modecalcul);

  source.onmessage = function(event) {
    const log = document.getElementById("log");
    log.innerHTML += `<div>${event.data}</div>`;
  };

  source.onerror = function() {
    source.close();
    const log = document.getElementById("log");
    log.innerHTML += `<div style="color:red;">Erreur ou traitement terminé</div><p>Si les photomontages n'a pas marché, merci de transmettre l'image à b-dc@windowslive.com</p>`;
  };
}

lancerTraitement() 