<?php
// Folder path
$folderPath = 'media/images/';

// Scan the directory for image files
$imageFiles = glob($folderPath . '*.{jpg,jpeg,png,gif}', GLOB_BRACE);

// Output the list of image files
header('Content-Type: application/json');
echo json_encode($imageFiles);
?>
