<?php
	include 'conn.php';
	$insert = "INSERT INTO usuario(user, nome, email, senha) VALUES('".$_POST["user_name"]."', '".$_POST["nome"]."', '".$_POST["email"]."', '".$_POST["senha"]."')";
	echo $insert;
	$query = mysqli_query($conn, $insert);
	if(!$query){
		echo "<script>javascript:history.go(-1)'</script>";
		shell_exec("mkdir users/".$_POST["user_name"]);
	}
	else{
		header("location:login.php");
	}
	mysql_close($conn);
?>