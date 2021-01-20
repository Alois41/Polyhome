<?php
echo "start";

use Mosquitto\Client;

$mid = 0;
echo "test";
define('CLIENT_ID', "pubclient_" + getmypid());
$c = new Mosquitto\Client(CLIENT_ID);
echo "test2";
$c->onLog('var_dump');
echo "test3";
$c->onConnect(function () use ($c, &$mid) {
    $bdd = new PDO("mysql:dbname=polyhome;host=localhost", "root", "polyhome");

    if (isset($_GET["id"]) and isset($_GET["name"])) {
        $id = $_GET["id"];
        $array = $bdd->query("SELECT name FROM User WHERE id_user=$id")->fetch(PDO::FETCH_ASSOC);

        $req = $bdd->prepare("UPDATE User SET name=:name, path_image=:path, is_admin=:is_admin WHERE id_user=:id");
        $req->bindParam(':name', $_GET["name"]);
        $path = "/home/pi/data/videos/" . $_GET["name"];
        $req->bindParam(':path', $path);
        $is_admin = 0;
        $req->bindParam(':is_admin', $is_admin);
        $req->bindParam(':id', $_GET["id"]);
        if ($req->execute()) {
            $old_name = $array["name"];
            $new_name = $_GET["name"];
            rename("/home/pi/data/videos/$old_name]", "/home/pi/data/videos/$new_name");
            $homepage = file_get_contents('/home/pi/user.txt');
            if ($homepage = $old_name)
                file_put_contents('/home/pi/user.txt', $new_name);

            rename("/home/pi/data/face_images/$old_name]", "/home/pi/data/face_images/$new_name");

            echo "success";
        } else
            echo "failed";
    }
    $mid = $c->publish("\$SYS/PYTHON/FACE", "changeName: $old_name, $new_name", 0);

    echo "Finished";
});

$c->onPublish(function ($publishedId) use ($c, $mid) {
    if ($publishedId == $mid) {
        $c->disconnect();

        echo "Finished";
        echo "disconnected";
    }
});

echo "connect";
$c->connect("localhost");
echo "loop";

$seconds = 5;
while($seconds>0){
$c->loop();
sleep(1);
$seconds--;
}

?>
<?php
//use Mosquitto\Client;
//
//$mid = 0;
//echo "test";
//define('CLIENT_ID', "pubclient_" + getmypid());
//$c = new Mosquitto\Client(CLIENT_ID);
//
//$c->onLog('var_dump');
//
//
//$c->onConnect(function() use ($c, &$mid) {
//    echo "connected";
//    $bdd = new PDO("mysql:dbname=polyhome;host=localhost", "root", "polyhome");
//
//    if (isset($_GET["id"]) and isset($_GET["name"])) {
//        $id = $_GET["id"];
//        $array  = $bdd->query("SELECT name FROM User WHERE id_user=$id")->fetch(PDO::FETCH_ASSOC);
//
//        $req = $bdd->prepare("UPDATE User SET name=:name, path_image=:path, is_admin=:is_admin WHERE id_user=:id");
//        $req->bindParam(':name', $_GET["name"]);
//        $path = "/home/pi/data/videos/" . $_GET["name"];
//        $req->bindParam(':path', $path);
//        $is_admin = 0;
//        $req->bindParam(':is_admin', $is_admin);
//        $req->bindParam(':id', $_GET["id"]);
//        if ($req->execute()) {
//            $old_name = $array["name"];
//            $new_name = $_GET["name"];
//            rename("/home/pi/data/videos/$old_name]", "/home/pi/data/videos/$new_name");
//            $homepage = file_get_contents('/home/pi/user.txt');
//            if($homepage = $old_name)
//                file_put_contents('/home/pi/user.txt', $new_name);
//
//            $mid = $c->publish("\$SYS/PYTHON/FACE", "changeName: $old_name\,$new_name", 2);
//            echo "success";
//        } else
//            echo "failed";
//    }
//    echo "Finished";
//});
//
//$c->onPublish(function($publishedId) use ($c, $mid) {
//    if ($publishedId == $mid) {
//        $c->disconnect();
//        echo "disconnected";
//    }
//});
//
//echo "connect";
//$c->connect("localhost");
//echo "loop";
//$c->loopForever();
