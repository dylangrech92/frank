<?php

/**
 * Shopping cart that tracks items and totals.
 */
class Cart
{
    /** @var array<int, array{name: string, price: float}> */
    private array $items = [];

    /**
     * Add an item to the cart.
     */
    public function addItem(string $name, float $price): void
    {
        $this->items[] = ['name' => $name, 'price' => $price];
    }

    /**
     * Calculate and return the total price of all items.
     */
    public function total(): float
    {
        return array_reduce(
            $this->items,
            fn($carry, $item) => $carry + $item['price'],
            0.0
        );
    }
}
