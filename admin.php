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
<body class="">
	<nav class="navbar navbar-toggleable-md navbar-inverse bg-inverse">
		<button class="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
			<span class="navbar-toggler-icon"></span>
		</button>
		<a class="navbar-brand f300"  href="#"><?php echo $_SESSION["user_name"]?></a>

		<div class="collapse navbar-collapse" id="navbarNav">
			<ul class="navbar-nav offset-md-8">
				<li class="nav-item active">
					<a class="nav-link f300" data-toggle="tab" href="#graphs" role="tab">Inicio</a>
				</li>
				<li class="nav-item ">
					<a class="nav-link f300" data-toggle="tab" href="#generateGraph" role="tab">Gerar Frozen</a>
				</li>
				<li class="nav-item">
					<a class="nav-link f300 disabled" href="#">Configurações</a>
				</li>
				<li class="nav-item">
					<a class="nav-link f300" href="logout.php">Sair</a>
				</li>
			</ul>
		</div>
	</nav>
	<div class="tab-content">
		<div class="container-fluid mt-5 tab-pane fade show active" role="tabpanel" id="graphs">
			<div class="row">
				<div class="col-12 col-md-6 bg-white rounded">
					<div id="chart">
					</div>
					<div class="descChart col-12 col-md-4 offset-md-8 text-right text-muted">
						<?php getLastChart(2); ?>	
					</div>
				</div>
				<div class="col-12 col-md-6">
					<h1 class="f300">Chart List</h1>
					<ul class="list-group mt-3 listChart">
						<?php listChart(); ?>
					</ul>
				</div>
			</div>
		</div>	<!-- container graphs-->
		<div class="container-fluid mt-5 tab-pane fade" role="tabpanel" id="generateGraph">
			<div class="row">
				<div class="container">
					<h1 class="f300">Novo Gráfico</h1>
					<form method="post" action="executar.php">
						<div class="row mt-5 ">
							<div class="col-12 col-md-6">
								<div class="row">
									<div class="col-12 col-md-6">
										<div class="form-group">
											<label for="tipoFW">Tipo Frozen Wave</label>
											<select class="form-control" id="tipoFW" name="tipoFW">
												<option value="1">1</option>
												<option value="2">2</option>
												<option value="3">3</option>
												<option value="4">4</option>
												<option value="5">5</option>
												<option value="6">6</option>
												<option value="7">7</option>
												<option value="8">8</option>
											</select>
										</div>
										<div class="form-group">
											<label for="ordemFeixes">Ordem dos Feixes de Bessel</label>
											<select class="form-control" id="ordemFeixes" name="ordemFeixes">
												<option value="0">0</option>
												<option value="1">1</option>
												<option value="2">2</option>
												<option value="3">3</option>
												<option value="4">4</option>
												<option value="5">5</option>
											</select>
										</div>
									</div>
									<div class="col-12 col-md-6">
										<legend class="col-form-legend">Opções </legend>
										<div class="form-check">
											<label class="form-check-label">
												<input class="form-check-input" type="radio" name="opt" value="1" checked>
												Bessel
											</label>
										</div>
										<div class=" row-check">
											<label class="form-check-label">
												<input class="form-check-input" type="radio" name="opt" value="2" >
												PSI
											</label>
										</div>
										<div class="form-check">
											<label class="form-check-label">
												<input class="form-check-input" type="radio" name="opt" value="3" >
												GNM-Magnético
											</label>
										</div>
										<div class="form-check">
											<label class="form-check-label">
												<input class="form-check-input" type="radio" name="opt" value="4" >
												Anbn
											</label>
										</div>
									</div>
								</div>
							</div>

							<!-- Coluna Direita -->
							<div class="col-12 col-md-6">
								<label for="rangeInicial">Range</label>
								<div class="row">
									<div class="col-12 col-md-6">
										<input type="text" class="form-control  mb-sm-0 mt-0 mt-sm-2 mb-2" id="rangeInicial" placeholder="Inicial" name="rangeInicial">
									</div>
									<div class="col-12 col-md-6">
										<input type="text" class="form-control  mb-sm-0 mt-0 mt-sm-2 mb-2" id="rangeFinal" placeholder="Final" name="rangeFinal">
									</div>
								</div>
								<input type="text" class="col-12 col-md-3 form-control mt-0 mt-sm-2" id="passo" placeholder="Passo" name="passo"  data-toggle="tooltip" data-placement="bottom" title="Passo maior que 100">
								<button type="submit"  class="btn btn-primary mt-2 mx-auto mx-sm-auto" >Processar</button>
							</div>

						</div>
					</form>
				</div>
			</div>
		</div>	<!-- container generate graph -->
	</div><!-- Tab  content-->
	<div class="view-hidden">
		<span id="foo"></span>
	</div>
</body>
<script type="text/javascript" src="view/js/jquery.min.js"></script>
<script type="text/javascript" src="view/js/tether.min.js"></script>
<script type="text/javascript" src="view/js/bootstrap.min.js"></script>
<script type="text/javascript" src="view/js/d3.min.js"></script>
<script type="text/javascript" src="view/js/c3.min.js"></script>
<script type="text/javascript" src="view/js/spin.js"></script>
<script type="text/javascript" src="view/js/app.js"></script>
<script type="text/javascript" src="view/js/install.js"></script>
<script type="text/javascript" src="service-worker.js"></script>
<script type="text/javascript">
	// var cont = <?php getLastChart(1); ?>;
	
	var chart = c3.generate({
		data: {
			url:"./json/teste1.json",
			type: 'spline'
		}
	});

// var cont = lerJson("./json/teste1.json");
// function lerJson(url){
// 	$.getJSON(url, function(dados){		
// 		 return dados[0];
// 	});
// }
</script>
</html>
