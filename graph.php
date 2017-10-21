<?php
include 'lib.php';
if (!isset($_GET['graph'])) {
	echo "<h1>Houve um erro ao acessar este gráfico</h1>";
}else{
	$id = $_GET['graph'];
}
?>
<div id="chart">
</div>

<div id="#descChart">
	<?php getIndividualChart($id, 2); ?>	
</div>

<script type="text/javascript">
	var cont = <?php getIndividualChart($id,1); ?>;
	var chart = c3.generate({
		data: {
			columns: cont,
			type: 'spline'
		}
	});
</script>
</html>
