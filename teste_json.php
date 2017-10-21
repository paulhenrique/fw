<!DOCTYPE html>
<html>
<head>
	<title>Frozen Waves</title>
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<meta charset="utf-8">
	<link rel="manifest" href="manifest.json">

	<link rel="stylesheet" type="text/css" href="view/css/c3.min.css">
	<link rel="stylesheet" type="text/css" href="view/css/estilo.css">
</head>
<body class="">
	<div id="chart">
		
	</div>
</body>
<script type="text/javascript" src="view/js/jquery.min.js"></script>
<script type="text/javascript" src="view/js/d3.min.js"></script>
<script type="text/javascript" src="view/js/c3.min.js"></script>
<script type="text/javascript">
	var col1 = ['col1'],
	col2 = ['col2'];
	$.getJSON("./users/paul/BesselJ_r152.0.json", function (resultado) {
		// 	console.log(resultado[0]);
		// console.log(resultado[1]);
		$.each(resultado[1], function (index, value){	
				// console.log("index:"+index+"  value: "+value);
				// console.log(index);
				col2.push(value);
				col1.push(index);
				
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
	
		// var chart = c3.generate({
		// 	data: {
		// 		columns:[

		// 		],
		// 		type:'spline'
		// 	}
		// });
	</script>
	</html>
