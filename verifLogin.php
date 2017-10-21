<?php
include'lib.php';
include 'conn.php';
if(isset($_POST["senha"]) && isset($_POST["email"])){
	$senha = $_POST["senha"];
	$email = $_POST["email"];
	$result = mysqli_query($conn,"SELECT * FROM usuario WHERE email = '$email'");

	if ($result->num_rows > 0) {
		while($row = $result->fetch_assoc()) {
			if ($senha == $row["senha"]) {
				// setcookie("usuario", "teste", time() + 360 );
				$_SESSION["user_id"] = $row["id"];
				$_SESSION["user_name"] = $row["nome"];
				header("location: admin.php");
			}else{
				setAlert("login.php",2);
			}
		}
	}else {
		setAlert("login.php",4);
	}
}else{
	setAlert("login.php",1);
}
$conn->close();
?>

