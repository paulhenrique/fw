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
	<meta name="mobile-web-app-capable" content="yes">
	<link rel="icon" sizes="200x200" href="img/ico.png">
	<meta name="theme-color" content="#292b2c">
	<meta name="apple-mobile-web-app-capable" content="yes">
	<meta name="apple-mobile-web-app-title" content="Project">
	<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
	<link rel="apple-touch-icon-precomposed" href="img/ico.png">
	<meta name="msapplication-TileImage" content="img/ico.png">
	<meta name="msapplication-TileColor" content="#292b2c">
	<link rel="stylesheet" type="text/css" href="view/css/tether.min.css">
	<link rel="stylesheet" type="text/css" href="view/css/bootstrap.min.css">
	<link rel="stylesheet" type="text/css" href="view/css/font-awesome.min.css">
	<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
	<link rel="stylesheet" type="text/css" href="view/css/estilo.css">
	<style type="text/css">
	.border-header{
		border-top: 5px #4daec3 solid;
	}
	.text-blue{
		color: #4daec3 !important;
	}
	.text-dark {
		color: #2f4657 !important;
	}
	.text-dark-2{
		color: #9a9898 !important;
	}
	.header::after{
		content: "";
	}
	.exemplos{
		height: 400px;
		text-align: center;
	}
	.exemplos div{
		width:300px;
		height: 100%;
		z-index: 3;
		width: 580px;
		border-radius: 20px;
	}
	.exemplos div.center{
		display: inline-block;
		margin: 0 auto;
		width: 700px;
	}
	.exemplos div:first-child{
		position: absolute;
		left: 10px;
		width: 580px;
		z-index: 2;
		top: 50px;
	}
	.exemplos div:last-child{
		position: absolute;
		right: 10px;
		width: 580px;
		z-index: 2;
		top: 50px;
	}
</style>
</head>
<body class="text-dark">
	<nav class="navbar navbar-expand-lg navbar-white bg-white p-3 p-5 border-header">
		<div class="container position-relative">
			<a class="navbar-brand text-dark" href="index.php">Frozen Waves</a>
			<button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
				<i class="fa fa-bars text-dark" aria-hidden="true"></i>
			</button>
			<div class="collapse navbar-collapse position-relative" id="navbarSupportedContent">
				<ul class="navbar-nav mr-auto position-absolute r-0">
					<li class="nav-item">
						<a class="nav-link text-dark" href="login.php">Home</a>
					</li>
					<li class="nav-item">
						<a class="nav-link text-dark" href="cadastro.php">Link</a>
					</li>
				</ul>
			</div>
		</div>
	</nav>
	<div class="container-fluid pt-md-5 header">
		<div class="col-12 col-md-8 m-auto text-center">
			<div class="col-md-4 m-auto">
				<img class="img-fluid mt-md-5" src="view/img/logo-frozen-vetor.png">
			</div>
			<h2 class="text-dark f300 mt-5 pt-md-5">uma solução fechada, portável e gratuita</h2>
		</div>
	</div>

	<div class="container mt-5 pt-5">
		<div class="row justify-content-center">
			<div class="col-3">
				<div class="phone-view"></div>
			</div>
			<div class="col-12 col-md-6  rounded    p-5">
				<h2 class="f300"><i class="fa fa-question-circle-o" aria-hidden="true"></i> O que é isso?</h2>
				<p class="text-justify lead thin text-dark-2">
					<small>
						A complexidade dos métodos numéricos, a variedades na quantidade e uso de seus atributos de entrada e a quantidade de saída de dados pode conduzir os pesquisadores à errôneas interpretações e até limitar o uso das ferramentas para os que possuem menor conhecimento teórico em determinado assunto e que pretendem aplica-­lo a um certo fim. Nessa afirmação se enquadram as modelagem de sobreposições de Feixe de Bessel, que vêm sendo exploradas em várias das comunicações e computação quântica, assim como na detecção de tumores na medicina. Nesse contexto, esse projeto é a continuidade de uma parceria já em andamento entre pesquisadores do IFSP e da EESC/USP para o desenvolvimento de interfaces computacionais que visem simplificar a inserção de dados nesse complexos modelos numéricos e também a visualização de seus resultados em gráficos em uma solução fechada, portável e gratuita, visando sua possível distribuição futuramente.


						<!-- Nessa afirmação se enquadram as modelagem de sobreposições de Feixe de Bessel, que vêm sendo exploradas em várias das comunicações e computação quântica, assim como na detecção de tumores na medicina. Nesse contexto, esse projeto é a continuidade de uma parceria já em andamento entre pesquisadores do IFSP e da EESC/USP para o desenvolvimento de interfaces computacionais que visem simplificar a inserção de dados nesse complexos modelos numéricos e também a visualização de seus resultados em gráficos em uma solução fechada, portável e gratuita, visando sua possível distribuição futuramente. -->
					</small>
				</p>
			</div>
		</div>
	</div>
	
	<!-- <div class="container mt-5 ">
		<div class="col-md-6 m-auto col-12">
			<img src="view/img/schema-comunicacao3.png" class="img-fluid">
		</div>
	</div>  -->
	
<!--
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
			<h2 class="thin">Tecnologias</h2>
			<div class="row justify-content-center mt-5 align-items-center">
				<div class="col-md-2">
					<img class="img-fluid" src="view/img/php_logo.jpg">	
				</div>
				<div class="col-md-2">
					<img class="img-fluid" src="view/img/python_logo.png">	
				</div>
				<div class="col-md-2">
					<img class="img-fluid" src="view/img/jquery-logo.jpg">	
				</div> 
				
			</div>
			<div class="row justify-content-center p-md-5 align-items-center">
			</div>
			<div class="row justify-content-center align-items-center">
				<div class="col-md-1">
					<img class="img-fluid" src="view/img/html5_logo.png">	
				</div>
				<div class="col-md-1">
					<img class="img-fluid" src="view/img/css3_logo.png">	
				</div>
				<div class="col-md-1">
					<img class="img-fluid" src="view/img/js_logo.png">	
				</div>

				<div class="col-md-1">
					<img class="img-fluid" src="view/img/d3_logo.png">	
				</div>
				<div class="col-md-1">
					<img class="img-fluid" src="view/img/c3_logo.png">	
				</div>
			</div>
			<div class="row justify-content-center align-items-center">
				<div class="col-md-2 ">
					<img class="img-fluid" src="view/img/ajax_logo.jpg">	
				</div>
				<div class="col-md-2 ">
					<img class="img-fluid" src="view/img/mysql_logo.jpg">	
				</div>

				<div class="col-md-2 ">
					<img class="img-fluid" src="view/img/bootstrap_logo.png">	
				</div>
			</div>
		</div>
	</div> 
-->
</body>
<script type="text/javascript" src="view/js/jquery.min.js"></script>
<script type="text/javascript" src="view/js/tether.min.js"></script>
<script type="text/javascript" src="view/js/bootstrap.min.js"></script>
<script type="text/javascript" src="view/js/install.js"></script>
<script type="text/javascript" src="service-worker.js"></script>
</html>
