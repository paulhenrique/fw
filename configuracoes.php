<?php
session_start();
include'lib.php';
if (!isset($_SESSION["user_id"])){
	setAlert("login.php", 3);		
	die();
}
?>
<!DOCTYPE html>
<html manifest="manifest.appcache">
<head>
	<title>Frozen Waves</title>
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<meta charset="utf-8">
	<link rel="manifest" href="manifest.json">

	<!-- Add to homescreen for Chrome on Android -->
	<meta name="mobile-web-app-capable" content="yes">
	<link rel="icon" sizes="200x200" href="img/ico.png">
	<meta name="theme-color" content="#292b2c">

	<!-- Add to homescreen for Safari on iOS -->
	<meta name="apple-mobile-web-app-capable" content="yes">
	<meta name="apple-mobile-web-app-title" content="Project">
	<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
	<link rel="apple-touch-icon-precomposed" href="img/ico.png">

	<!-- Tile icon for Win8 (144x144 + tile color) -->
	<meta name="msapplication-TileImage" content="img/ico.png">
	<meta name="msapplication-TileColor" content="#292b2c">
	
	<link rel="stylesheet" type="text/css" href="view/css/tether.min.css">
	<link rel="stylesheet" type="text/css" href="view/css/bootstrap.min.css">
	<link rel="stylesheet" type="text/css" href="view/css/font-awesome.min.css">
	<link rel="stylesheet" type="text/css" href="view/css/c3.min.css">
	<link rel="stylesheet" type="text/css" href="view/css/estilo.css">
</head>
<body>
	<div class="container mt-5 pt-5">
		<h3 class="thin">Configurações do Perfil</h3>
		<div class="row mt-5">
			<div class="col-4">
			</div>
		</div>
		<nav class="navbar navbar-expand-lg navbar-light bg-light fixed-top ">
			<div class="container">
				<a class="navbar-brand" href="#"><i class="fa fa-user-circle" aria-hidden="true"></i> <?php echo $_SESSION["user_name"]; ?></a>
				<div class="collapse navbar-collapse" id="navbarText">
					<ul class="navbar-nav flex-row ml-md-auto d-none d-md-flex">
						<li class="nav-item">
							<a class="nav-link" href="start.php">Inicio</a>
						</li>
						<li class="nav-item active">
							<a class="nav-link" href="configuracoes.php">Configurações<span class="sr-only">(current)</span></a>
						</li>
						<li class="nav-item">
							<a class="nav-link" href="logout.php">Sair</a>
						</li>
					</ul>
				</div>
			</div>
		</nav>
	</body>
	<script type="text/javascript" src="view/js/jquery.min.js"></script>
	<script type="text/javascript" src="view/js/popper.min.js"></script>
	<script type="text/javascript" src="view/js/bootstrap.min.js"></script>
	<script type="text/javascript" src="view/js/d3.min.js"></script>
	<script type="text/javascript" src="view/js/c3.min.js"></script>
	<script type="text/javascript" src="view/js/spin.js"></script>
	<script type="text/javascript" src="view/js/app.js"></script>
	<script type="text/javascript" src="view/js/install.js"></script>
	<script type="text/javascript" src="service-worker.js"></script>
	</html>

