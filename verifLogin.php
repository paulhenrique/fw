<?php
include'lib.php';
include 'conn.php';
if(isset($_POST["senha"]) && isset($_POST["email"])){

	$senha = $_POST["senha"];
	$login = $_POST["email"];

	$result = mysqli_query($conn,"SELECT * FROM usuarios WHERE login = '$login'");

	if ($result->num_rows > 0) {
		while($row = $result->fetch_assoc()) {
			if ($senha == $row["senha"]) {
				setcookie("usuario", "teste", time() + 360 );
				$_SESSION["usuario"] = $row["id"];
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

