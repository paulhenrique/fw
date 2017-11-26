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
		// $("#chart").fadeOut("slow");
		// $("#chart").load(location, " #chart").fadeIn("slow");
		// $(".descChart").load(location ," #descChart");
	});