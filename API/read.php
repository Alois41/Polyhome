<?php
    $bdd = new PDO("mysql:dbname=polyhome;host=localhost", "root", "polyhome");

    $array  = $bdd->query('SELECT * FROM User')->fetchAll(PDO::FETCH_ASSOC);
    echo '{"User":'.json_encode($array);
    $array  = $bdd->query('SELECT * FROM Music')->fetchAll(PDO::FETCH_ASSOC);
    echo ',"Music":'.json_encode($array);
    $array  = $bdd->query('SELECT * FROM MusicFavorite')->fetchAll(PDO::FETCH_ASSOC);
    echo ',"MusicFavorite":'.json_encode($array);
    $array  = $bdd->query('SELECT * FROM Requete')->fetchAll(PDO::FETCH_ASSOC);
    echo ',"Requete":'.json_encode($array)."}";
?>