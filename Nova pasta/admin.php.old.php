<?php
include'lib.php';
checklogin();
?>
<!DOCTYPE html>
<html>
<head>
	<title>Frozen Waves</title>
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<meta charset="utf-8">
	<link rel="stylesheet" type="text/css" href="css/tether.min.css">
	<link rel="stylesheet" type="text/css" href="css/bootstrap.min.css">
	<link rel="stylesheet" type="text/css" href="css/font-awesome.min.css">
	<link rel="stylesheet" type="text/css" href="css/estilo.css">
</head>
<body>
	<div class="container">
		<h1 class="f300">Gerar Gráfico</h1>
		<form method="post" action="testes.php">
			<div class="row mt-5 ">
				<!-- Coluna esquerda -->
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
								</select>
							</div>
							<div class="form-group">
								<label for="ordemFeixes">Ordem dos Feixes de Bessel</label>
								<select class="form-control" id="ordemFeixes" name="ordemFeixes">
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
									<input class="form-check-input" type="radio" name="opt" value="bessel" checked>
									Bessel
								</label>
							</div>
							<div class="form-check">
								<label class="form-check-label">
									<input class="form-check-input" type="radio" name="opt" value="psi" >
									PSI
								</label>
							</div>
							<div class="form-check">
								<label class="form-check-label">
									<input class="form-check-input" type="radio" name="opt" value="gnm-magnetico" >
									GNM-Magnético
								</label>
							</div>
							<div class="form-check">
								<label class="form-check-label">
									<input class="form-check-input" type="radio" name="opt" value="anbn" >
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
				</form>
				<span type="text" id="processar" class="btn btn-primary mt-2 mx-auto mx-sm-auto" >Processar</span>
				<!-- <span id="foo"></span> -->
			</div>
		</div> <!-- row -->
		
	</div>	<!-- container-fluid -->
</body>
<script type="text/javascript" src="js/jquery.min.js"></script>
<script type="text/javascript" src="js/tether.min.js"></script>
<script type="text/javascript" src="js/bootstrap.min.js"></script>
<script type="text/javascript" src="js/spin.js"></script>
<script type="text/javascript" src="js/app.js"></script>
</html>