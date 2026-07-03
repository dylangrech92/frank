<?php

use PHPUnit\Framework\TestCase;

final class CartTest extends TestCase
{
    public function testTotalSumsItemPrices(): void
    {
        $cart = new Cart();
        $cart->addItem('Sprocket', 12.50);
        $cart->addItem('Widget', 2.50);
        $this->assertSame(15.00, $cart->total());
    }

    // deliberately failing test for the test-explorer live tests
    public function testEmptyCartTotalIsTen(): void
    {
        $cart = new Cart();
        $this->assertSame(10.00, $cart->total());
    }
}
