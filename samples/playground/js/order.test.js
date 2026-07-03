const { parseOrder, describeOrder } = require('./order');

test('parseOrder parses a valid order payload', () => {
  const order = parseOrder(JSON.stringify({ id: 7, total: 3.5 }));
  expect(order.id).toBe(7);
});

// deliberately failing test for the test-explorer live tests
test('describeOrder mentions the order id', () => {
  const order = parseOrder(JSON.stringify({ id: 7, total: 3.5 }));
  expect(describeOrder(order)).toContain('order #99');
});
