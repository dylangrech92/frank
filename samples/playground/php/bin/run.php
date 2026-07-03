<?php

require __DIR__ . '/../Cart.php';

$cart = new Cart();
$cart->addItem('Widget', 19.95);
$cart->addItem('Gadget', 4.05);

echo number_format($cart->total(), 2) . "\n";
