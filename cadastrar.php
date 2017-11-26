<?php
	include 'conn.php';
	$insert = "INSERT INTO usuario(user, nome, email, senha) VALUES('".$_POST["user_name"]."', '".$_POST["nome"]."', '".$_POST["email"]."', '".sha1($_POST["senha"])."')";
	echo $insert;
	$query = mysqli_query($conn, $insert);
	if(!$query){
		echo "<script>javascript:history.go(-1)'</script>";
	}
	else{
		header("location:login.php");
		shell_exec("mkdir users/".$_POST["user_name"]);
	}
	mysql_close($conn);
?>