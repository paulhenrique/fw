<!DOCTYPE html>
<html>
<head>
	<title>Frozen Waves</title>
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<meta charset="utf-8">
	<link rel="manifest" href="manifest.json">

	<link rel="stylesheet" type="text/css" href="css/c3.min.css">
	<link rel="stylesheet" type="text/css" href="css/estilo.css">
</head>
<body class="">
	<div id="chart">
		
	</div>
</body>
<script type="text/javascript" src="js/jquery.min.js"></script>
<script type="text/javascript" src="js/d3.min.js"></script>
<script type="text/javascript" src="js/c3.min.js"></script>
<script type="text/javascript">
	var col1 = ['col1'],
		col2 = ['col2'];
	$.getJSON("./json/teste2.json", function (resultado) {

		$.each(resultado, function (index, value, ){
				// console.log("index:"+index+"  value: "+value);
				
				col1.push(this[0]);
				col2.push(this[1]);
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
