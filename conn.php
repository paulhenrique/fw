<?php
// $servername = "localhost";
// $username = "root";
// $password = "itp.ifsp.2017";
// $dbname = "project_frozen";

$servername = "localhost";
$username = "root";
$password = "";
$dbname = "project_frozen";

// Create connection
$conn = new mysqli($servername, $username, $password,$dbname);
// Cddheck connection
if ($conn->connect_error) {
	    die("Connection failed: " . $conn->connect_error);
} 
?>
