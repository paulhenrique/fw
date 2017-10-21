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
	<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
	<link rel="stylesheet" type="text/css" href="view/css/estilo.css">
</head>
<body class="bg-light">
	<div class="container-fluid mt-5">
		<div class="row">
			<div class="col-12  text-center text-dark p-5 mt-5">
				<h1 class="thin mt-5 pt-5">frozen project</h1>
				<a href="login.php" class="btn btn-primary mt-3">Entrar</a>
				<a href="cadastro.php" class="btn btn-secondary mt-3">Fazer Parte</a>
			</div>
		</div>	
	</div> 
	
	<div class="container-fluid mt-5 p-md-5">	
		<div class="container p-md-5">
			<div class="row justify-content-center">
				<div class="col-md-5 col-12">
					<div class="phone-view"></div>
				</div>
				<div class="col-md-4 thin"> 
					<h2 class="thin pb-3 mt-5">O que é isso?</h2>
					<p class="text-justify">
						A complexidade dos métodos numéricos, a variedades na quantidade e uso de seus atributos de entrada e a quantidade de saída de dados pode conduzir os pesquisadores à errôneas interpretações e até limitar o uso das ferramentas para os que possuem menor conhecimento teórico em determinado assunto e que pretendem aplica-­lo a um certo fim. Nessa afirmação se enquadram as modelagem de sobreposições de Feixe de Bessel, que vêm sendo exploradas em várias das comunicações e computação quântica, assim como na detecção de tumores na medicina. Nesse contexto, esse projeto é a continuidade de uma parceria já em andamento entre pesquisadores do IFSP e da EESC/USP para o desenvolvimento de interfaces computacionais que visem simplificar a inserção de dados nesse complexos modelos numéricos e também a visualização de seus resultados em gráficos em uma solução fechada, portável e gratuita, visando sua possível distribuição futuramente.
					</p>
				</div>
			</div>
		</div> 
	</div>

	<div class="p-5 p-md-0">
		<div class="monitor"></div>
	</div>
	<div class="container-fluid">
		<div class="container">
			<h2 class="thin p-3 text-center">Objetivos</h2>
			<div class="row justify-content-center">
				<p class="thin text-justify col-md-6">
					Implementaçã o de uma interface humano-­computador, focando-­se na usabilidade para sofisticados métodos numéricos voltados às telecomunicações e bio-­engenharia, agrupando de maneira sistêmica os atributos nessa interface. Essa generalidade de interface traz como consequências as especificidades de estudos sobre técnicas e tecnologias para o desenvolvimento de interfaces de software, usabilidade e, neste caso, também a visualização de dados cientı́ficos, podendo-­se considerar diferentes tipos de gráficos e dados à serem visualizados.
				</p>
			</div>
		</div>
	</div>
	<div class="container-fluid p-md-5 mt-md-5">
		<div class="container p-5 mt-md-5 text-center">
			<h2 class="thin">Técnologias</h2>
			<div class="row justify-content-center mt-5 align-items-center">
				<div class="col">
					<img class="img-fluid" src="view/img/php_logo.jpg">	
				</div>
				<div class="col">
					<img class="img-fluid" src="view/img/python_logo.png">	
				</div>
				<div class="col">
					<img class="img-fluid" src="view/img/jquery-logo.jpg">	
				</div>
				
			<!-- </div>
			<div class="row justify-content-center p-md-5 align-items-center"> -->
				</div>
			<div class="row justify-content-center align-items-center">
				<div class="col">
					<img class="img-fluid" src="view/img/html5_logo.png">	
				</div>
				<div class="col">
					<img class="img-fluid" src="view/img/css3_logo.png">	
				</div>
				<div class="col">
					<img class="img-fluid" src="view/img/js_logo.png">	
				</div>
			
				<div class="col">
					<img class="img-fluid" src="view/img/d3_logo.png">	
				</div>
				<div class="col">
					<img class="img-fluid" src="view/img/c3_logo.png">	
				</div>
				</div>
			<div class="row justify-content-center align-items-center">
				<div class="col ">
					<img class="img-fluid" src="view/img/ajax_logo.jpg">	
				</div>
				<div class="col ">
					<img class="img-fluid" src="view/img/mysql_logo.jpg">	
				</div>

				<div class="col ">
					<img class="img-fluid" src="view/img/bootstrap_logo.png">	
				</div>
			</div>
		</div>
	</div>

	<div class="container text-center">
		<h2 class="thin">Uma parceria</h2>
		<div class="row p-5 justify-content-center">
			<div class="col-2">
				<img src="view/img/logo_ifsp.jpg" class="img-fluid">
			</div>
			<div class="col-2">
				<img src="view/img/logo_eesc.png" class="img-fluid">
			</div>
		</div>
	</div>
	<div class="text-center pr-1 mt-5">
		<p>design and layout: @phvcandido</p>
	</div>
</body>
<script type="text/javascript" src="view/js/jquery.min.js"></script>
<script type="text/javascript" src="view/js/tether.min.js"></script>
<script type="text/javascript" src="view/js/bootstrap.min.js"></script>
<script type="text/javascript" src="view/js/install.js"></script>
<script type="text/javascript" src="service-worker.js"></script>
</html>