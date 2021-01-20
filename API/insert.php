<?php
$bdd = new PDO("mysql:dbname=polyhome;host=localhost", "root", "polyhome");

if (isset($_GET["name"])) {
    $req = $bdd->prepare("INSERT INTO User(name, path_image, is_admin) VALUES (:name, :path, :is_admin)");
    $req->bindParam(':name', $_GET["name"]);
    $path = "/home/pi/data/videos/" . $_GET["name"];
    $req->bindParam(':path', $path);
    $is_admin = 0;
    $req->bindParam(':is_admin', $is_admin);
    if ($req->execute())
        echo "success";
    else
        echo "failed";
}
?>