$(function () {
	$('[data-toggle="tooltip"]').tooltip();
});

var opts = {
  lines: 15 // The number of lines to draw
, length: 23 // The length of each line
, width: 10 // The line thickness
, radius: 64 // The radius of the inner circle
, scale: 0.75 // Scales overall size of the spinner
, corners: 1 // Corner roundness (0..1)
, color: '#000' // #rgb or #rrggbb or array of colors
, opacity: 0.65 // Opacity of the lines
, rotate: 7 // The rotation offset
, direction: 1 // 1: clockwise, -1: counterclockwise
, speed: 1.3 // Rounds per second
, trail: 17 // Afterglow percentage
, fps: 20 // Frames per second when using setTimeout() as a fallback for CSS
, zIndex: 2e9 // The z-index (defaults to 2000000000)
, className: 'spinner' // The CSS class to assign to the spinner
, top: '50%' // Top position relative to parent
, left: '50%' // Left position relative to parent
, shadow: false // Whether to render a shadow
, hwaccel: false // Whether to use hardware acceleration
, position: 'absolute' // Element positioning
}
var target = document.getElementById('foo');
var spinner = new Spinner(opts).spin(target);

$(".btn-PreviewGraph").on("click", function (e) {
	e.preventDefault();
	$(".list-graph").removeClass("active");
	var location = $(this).attr("href");
	$(this).children("li").addClass("active");
	$("#chart").fadeOut("slow");
	$("#chart").load(location, " #chart").fadeIn("slow");
	$(".descChart").load(location ," #descChart");
});

$("#processar").on("click", function (e) {
//	e.preventDefault();
	$(".view-hidden").fadeIn("slow");
	$(".view-hidden #foo").fadeIn("slow");
});
$(".view-hidden").on("click", function(){
	$(".view-hidden").fadeOut("slow");
	$(".view-hidden #foo").fadeOut("slow");
});
