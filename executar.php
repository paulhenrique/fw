<?php
    session_start();
    
    $tipo           = "tipo=".$_POST["tipoFW"];
    $step           = "step=".$_POST["passo"];
    $rangeInicial   = "rangeInicial=".$_POST["rangeInicial"];
    $rangeFinal     = "rangeFinal=".$_POST["rangeFinal"];
    $option         = "option=".$_POST["opt"];
    $ordem          = "ordem=".$_POST["ordemFeixes"];
    $user           = "user=".$_SESSION["user_name"];
    $comando = 'python python/fw.py '.$tipo.' '.$step." ".$option." ".$ordem." ".$user;
    echo $comando;
    $retorno_python= shell_exec($comando);
    echo $retorno_python;
    if($retorno_python){
        header("location:admin.php");
    }

    //echo $retorno_python;
    //$decodificado = json_decode($retorno_python);

//    echo $decodificado;
?>
