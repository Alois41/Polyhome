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
$c->onConnect(function() use ($c, &$mid) {
    $mid = $c->publish("\$SYS/PHP", "train", 2);
    echo "Finished";
});

$c->onPublish(function($publishedId) use ($c, $mid) {
    if ($publishedId == $mid) {
        $c->disconnect();
        echo "disconnected";
    }
});

echo "connect";
$c->connect("localhost");
echo "loop";
$c->loopForever();
