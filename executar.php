<?php
session_start();

$tipo           = "tipo=".$_POST["tipoFW"];
$step           = "step=".$_POST["passo"];
$rangeInicial   = "rangeInicial=".$_POST["rangeInicial"];
$rangeFinal     = "rangeFinal=".$_POST["rangeFinal"];
$option         = "option=".$_POST["opt"];
$ordem          = "ordem=".$_POST["ordemFeixes"];
$user           = "usuario=".$_SESSION["user_name"];
$comando = 'python fw.py '.$tipo.' '.$step." ".$option." ".$ordem." ".$user;
echo $comando;
$retorno_python= shell_exec($comando);
echo $retorno_python;

include "conn.php";
    $insert = "INSERT into charts (id_tipo_chart, dados, id_user) VALUES ('".$_POST["tipoFW"]."', '".$nome_arquivo."', '".$_SESSION["user_id"]."')";
    $result = mysqli_query($conn, $insert);        

if(strpos($retorno_python,"]") !== false)
    echo "<script>javascript:history.back()</script>";
else
    echo "string";
?>
