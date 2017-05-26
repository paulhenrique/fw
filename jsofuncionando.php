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
	col2 = ['col2'],
	x 	 = ['x'];
	$.getJSON("./json/teste2.json", function (resultado) {

		$.each(resultado, function (index, value){
				// console.log("index:"+index+"  value: "+value);
				x.push(index);
				col1.push(this[0]);
				col2.push(this[1]);
			});
		var chart = c3.generate({
			data: {
				x: 'x',
				columns:[
				x,
				col1,
				col2
				]
			},
			axis: {
				x: {
					type: 'timeseries',
					tick: {
						values: x
					}
				}
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
