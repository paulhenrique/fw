<?php
session_start();
$nome_arquivo = $_POST["nome_arquivo"];
include "conn.php";
$insert = "INSERT into charts (id_tipo_chart, dados, id_user) VALUES ('".$_POST["tipoFW"]."', '".$nome_arquivo."', '".$_SESSION["user_id"]."')";
$result = mysqli_query($conn, $insert);
?>