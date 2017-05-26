<?php
    $step           = "step=".$_POST["passo"];
    $tipo           = "tipo=".$_POST["tipoFW"];
    $rangeInicial   = "rangeInicial=".$_POST["rangeInicial"];
    $rangeFinal     = "rangeFinal=".$_POST["rangeFinal"];
    $option         = "option=".$_POST["opt"];
    $ordem          = "ordem=".$_POST["ordemFeixes"];
    $comando = 'python fw.py '.$tipo.' '.$step." ".$option." ".$ordem;
    echo $comando;
    $retorno_python= shell_exec($comando);
    //echo $retorno_python;
    //$decodificado = json_decode($retorno_python);

//    echo $decodificado;
?>
