<!DOCTYPE html>
<html>
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
	<link rel="apple-touch-icon-precomposed" href="view/img/ico.png">

	<!-- Tile icon for Win8 (144x144 + tile color) -->
	<meta name="msapplication-TileImage" content="img/ico.png">
	<meta name="msapplication-TileColor" content="#292b2c">
	
	<link rel="stylesheet" type="text/css" href="view/css/tether.min.css">
	<link rel="stylesheet" type="text/css" href="view/css/bootstrap.min.css">

</head>
<body class="text-dark">
	<div class="container">
		<div class="col-8 col-md-4 mx-auto mt-5 pt-5">
			<h2 class="mt-5 thin">Cadastro</h2>
			<form action="cadastrar.php" method="post">
				<div class="input-group mt-4">
					<input type="text" class="form-control" placeholder="Nome" name="nome" required autofocus>
				</div>
				<div class="input-group mt-2">
					<input type="text" class="form-control" placeholder="Nome de usuário" name="user_name" required autofocus>
				</div>
				<div class="input-group mt-2">
					<input type="email" class="form-control" placeholder="E-mail" name="email" required>
				</div>
				<div class="input-group mt-2">
					<input type="password" id="txtSenha" name="senha" class="form-control" placeholder="Senha" required>
				</div>
				<div class="input-group mt-2">
					<input oninput="validaSenha(this)" type="password" name="senha_confirm" class="form-control" placeholder="Confirmar Senha" required >
				</div>
				<button type="submit" class="btn mt-4 btn-primary btn-block">Cadastrar</button>
				<a href="index.php" class="btn mt-1 btn-secondary btn-block">Voltar</a>
			</form>
		</div>
	</div>	
</body>
<script type="text/javascript" src="js/jquery.min.js"></script>
<script type="text/javascript" src="js/tether.min.js"></script>
<script type="text/javascript" src="js/bootstrap.min.js"></script>
<script type="text/javascript" src="js/install.js"></script>
<script type="text/javascript" src="service-worker.js"></script>
<script type="text/javascript">
	function validaSenha (input){ 
		if (input.value != document.getElementById('txtSenha').value) {
			input.setCustomValidity('Repita a senha corretamente');
		} else {
			input.setCustomValidity('');
		}
	} 
</script>
</html>