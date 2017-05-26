<?php
include'lib.php';
?>
<!DOCTYPE html>
<html>
<head>
	<title>Frozen Waves</title>
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<meta charset="utf-8">
	<link rel="stylesheet" type="text/css" href="css/tether.min.css">
	<link rel="stylesheet" type="text/css" href="css/bootstrap.min.css">
</head>
<body>

	<div class="container">
		<div class="w-25 mx-auto mt-5 pt-5">
			<h3 class="mt-5">Login</h3>
			<form action="verifLogin.php" method="post">
				<div class="input-group mt-4">
					<input type="text" class="form-control" placeholder="E-mail" name="email" required autofocus>
				</div>
				<div class="input-group">
					<input type="password" name="senha" class="form-control" placeholder="Senha" required >
				</div>
				<button type="submit" class="btn mt-4 btn-primary btn-block">Entrar</button>
			</form>

			<div class="alert alert-info alert-dismissible mt-5" role="alert">
				<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>
				<?php
				if (!isset($_GET["alert"])) {
					echo "Bem Vindo!";
				}
				?>
				<strong><?php echo getAlert() ?></strong>
			</div>
		</div>
	</div>	
</body>
<script type="text/javascript" src="js/jquery.min.js"></script>
<script type="text/javascript" src="js/tether.min.js"></script>
<script type="text/javascript" src="js/bootstrap.min.js"></script>
</html>