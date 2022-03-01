<?php

$dbServername = "localhost"; 
$dbUsername = "root"; 
$dbPassword = "Megz242002_"; 
$dbName = "user_study"; 


$conn = mysqli_connect($dbServername, $dbUsername, $dbPassword, $dbName); 

$sql = "SELECT * FROM condition_status;"; 
$result + mysqli_query($conn, $sql);
$resultCheck = mysqli_num_rows($result); 

if($resultCheck > 0) {
    while($row = mysqli_fetch_assoc()) {
        echo $row['condition']; 
    }
}

echo '<script>';
echo 'var condition_serial_number = ' . json_encode($row) . ';';
echo '</script>';
?>



