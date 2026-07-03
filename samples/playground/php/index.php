<?php

require __DIR__ . '/Cart.php';

$cart = new Cart();
$cart->addItem('Sprocket', 12.50);

echo "Cart total: $" . number_format($cart->total(), 2) . "\n";
