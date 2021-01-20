<?php
    $bdd = new PDO("mysql:dbname=polyhome;host=localhost", "root", "polyhome");

    if(isset($_GET["id"])){
    $req = $bdd->prepare("DELETE FROM User WHERE id_user=:id");
    $req->bindParam(':id', $_GET["id"]);
    if($req->execute())
        echo "success";
    else
        echo "failed";
    }
?>