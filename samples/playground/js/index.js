const { parseOrder, describeOrder } = require('./order');

const rawOrder = JSON.stringify({ id: 1, total: 9.99 });
const order = parseOrder(rawOrder);
const result = describeOrder(order);
console.log(result);
