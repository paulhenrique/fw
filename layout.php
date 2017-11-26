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
	<div class="sidebar">
		<ul class="tab-cog">
			<li><a href="#"><i class="fa fa-user-circle" aria-hidden="true"></i> <?php echo $_SESSION["user_name"]; ?></a></li>
			<li><a href="#inicio"><i class="fa fa-bar-chart" aria-hidden="true"></i> Inicio</a></li>
			<li><a href="#generate-chart"><i class="fa fa-plus-circle" aria-hidden="true"></i> Novo Gráfico</a></li>
			<li><a href="#graficos"><i class="fa fa-cog" aria-hidden="true"></i> Gerenciar Gráficos</a></li>
		</ul>
		<ul class="user-cog">
			
			<li><a href="configuracoes.php"><i class="fa fa-cog" aria-hidden="true"></i> Configurações</a></li>
			<li><a href="logout.php"><i class="fa fa-sign-out" aria-hidden="true"></i> Sair</a></li>
		</ul>
	</div>
	<div class="bar-top">
		<p class="thin"><i class="fa fa-area-chart" aria-hidden="true"></i> Frozen Waves - Inicio</p>
	</div>
	<div class="main">
		
		<div id="inicio" class="show tab-content">
			<div class="container tab-pane fade show active" role="tabpanel" id="graphs">
				<div class="row justify-content-end pt-5">
					<div class="col-12 col-md-8 rounded">
						<div class="m-3 text-center" id="chart-dados"></div>
						<div id="chart">
						</div>	
					</div>
					<div class="col-12 col-md-4">
						<h4 class="thin">Chart List</h4>
						<ul class="list-group mt-3 list-chart">
							<?php 
							listCharts();
							?>
						</ul>
					</div>
				</div>
			</div>
		</div>
		<div id="generate-chart" class="tab-content">
			<div class="container mt-5">
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
											<input class="form-check-input" type="radio" name="opt" value="10" >
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
</div>
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
<script type="text/javascript">
	function gerarGrafico(file){

		var col1 = ['col1'],
		col2 = ['col2'];

		var user = "<?php echo $_SESSION["user_name"]; ?>";	
		$.getJSON("./users/"+user+"/"+file+"", function (resultado) {

			$.each(resultado[0], function(index, value){
				$("#chart-dados").html(value);
			});
			$.each(resultado[1], function (index, value){	
				// console.log(value);
				col2.push(value[0]);
				col1.push(value[1]);

			});
			var chart = c3.generate({
				data: {
					columns:[
					col1,
					col2
					]
				}
			});
		});
	};

	$(".btn-PreviewGraph").on("click", function (e) {
		e.preventDefault();
		$(".list-graph").removeClass("active");
		$(this).children("li").addClass("active");
		var file = $(this).attr("data-file");
		gerarGrafico(file);
	});
	$(function () {
		$(".list-chart a:eq(0)").click();
	});

	$(".sidebar .tab-cog li a").on("click", function(){
		var div = $(this).attr("href");
		div = $(div);
		$(".tab-content").removeClass("show");
		div.addClass("show");
	});
</script>
</html>

