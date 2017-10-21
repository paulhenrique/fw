<?php
include'lib.php';
if (isset($_SESSION["user_name"])) {
	header("location: admin.php");
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
	<link rel="stylesheet" type="text/css" href="view/css/estilo.css">
</head>
<body>
	<nav class="fixed-top navbar navbar-toggleable-md navbar-inverse bg-inverse">
		<button class="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
			<span class="navbar-toggler-icon"></span>
		</button>
		<a class="navbar-brand f300"  href="#">project_frozen</a>

		<div class="collapse navbar-collapse" id="navbarNav">
			<ul class="navbar-nav offset-md-10">
				<li class="nav-item active">
					<a class="nav-link f300" href="index.php">Inicio <span class="sr-only">(current)</span></a>
				</li>
				<li class="nav-item">
					<a class="nav-link f300" href="#">Sobre</a>
				</li>
				<!-- <li class="nav-item">
					<a class="nav-link f300" href="#">Pricing</a>
				</li> -->
				<!-- <li class="nav-item">
					<a class="nav-link f300" href="logout.php">Sair</a>
				</li> -->
			</ul>
		</div>
	</nav>
	<div class="container-fluid mt-5">
		<div class="row head">
			<div class="offset-md-6 col-12 col-md-6 text-right text-white">
				<h1 class="display-1 mt-5 pt-5">frozen project</h1>
				<p class="lead">
					Vivamus sagittis lacus vel augue laoreet rutrum faucibus dolor auctor. Duis mollis, est non commodo luctus.
				</p>

				<a href="login.php" class="btn btn-primary btn-lg mt-3">Entrar</a>
				<a href="login.php" class="btn btn-secondary btn-lg mt-3">Fazer Parte</a>
			</div>
		</div>	
		<div class="row cont mt-5">
			<div class="container mt-5 text-center">
				<h1 class="display-4 f300 ">Introdução</h1>
				<p class="lead mt-3">
					Vivamus sagittis lacus vel augue laoreet rutrum faucibus dolor auctor. <br> Duis mollis, est non commodo luctus.
				</p>
				<div class="row f300 mt-5">
					<div class="col-12 col-md-4">
						<h1 class="display-1"><i class="fa fa-bar-chart" aria-hidden="true"></i></h1>
						<h1 class="f300">Sed do Eiusmod</h1>
						<p class="mt-3">
							Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
							tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
							quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea 
						</p>
					</div>
					<div class="col-12 col-md-4">
						<h1 class="display-1"><i class="fa fa-user" aria-hidden="true"></i></h1>
						<h1 class="f300">Consectetur Adipisicing Elit</h1>
						<p class="mt-3">
							Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
							tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam.
						</p>
					</div>
					<div class="col-12 col-md-4">
						<h1 class="display-1"><i class="fa fa-mobile" aria-hidden="true"></i></h1>
						<h1 class="f300">Lorem ipsum sit amet</h1>
						<p class="mt-3">
							Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
							tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim 
						</p>
					</div>
				</div>

			</div> <!-- container -->
		</div> <!-- row -->
		<!-- <div class="row  mt-5 pt-5">
			<div class="container text-center">
				<h1 class="display-4 f300" >Exemplos</h1>
				<p>Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
				tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
				quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
				consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse
				cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non
				proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
			</div>
		</div> -->
	</div> <!-- container fluid -->
	<footer class="bg-inverse text-white  py-3  mt-5">
		<div class="col-12 col-md-6 offset-md-6 text-center">
			<p class="f300">
				Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
			</p>
		</div>
	</footer>
</body>
<script type="text/javascript" src="view/js/jquery.min.js"></script>
<script type="text/javascript" src="view/js/tether.min.js"></script>
<script type="text/javascript" src="view/js/bootstrap.min.js"></script>
<script type="text/javascript" src="view/js/install.js"></script>
<script type="text/javascript" src="service-worker.js"></script>
</html>