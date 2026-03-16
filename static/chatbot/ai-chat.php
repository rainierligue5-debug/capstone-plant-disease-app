<?php

$userMessage = $_POST['message'];

$apiKey = "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";

$data = [
 "model" => "mistralai/mistral-7b-instruct",
 "messages" => [
   ["role" => "user", "content" => $userMessage]
 ]
];

$ch = curl_init();

curl_setopt($ch, CURLOPT_URL, "https://openrouter.ai/api/v1/chat/completions");
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, [
 "Authorization: Bearer $apiKey",
 "Content-Type: application/json"
]);

curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));

$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true);

echo $result['choices'][0]['message']['content'];

?>