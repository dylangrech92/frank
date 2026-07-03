/**
 * Base handler class for order processing.
 */
class Handler {
  /**
   * Handle an incoming order object.
   * @param {Object} order 
   * @returns {string}
   */
  handle(order) {
    return 'unhandled';
  }
}

/**
 * Order handler that formats order details.
 */
class OrderHandler extends Handler {
  /**
   * Handle an order and return a formatted string.
   * @param {Object} order 
   * @returns {string}
   */
  handle(order) {
    return `Order #${order.id} — Total: $${order.total.toFixed(2)}`;
  }
}

/**
 * Parse and validate a JSON order string.
 * @param {string} raw 
 * @returns {Object}
 */
function parseOrder(raw) {
  const order = JSON.parse(raw);

  if (!('id' in order)) {
    throw new Error('Missing field: id');
  }
  if (!('total' in order)) {
    throw new Error('Missing field: total');
  }

  return order;
}

/**
 * Describe an order using OrderHandler.
 * @param {Object} order 
 * @returns {string}
 */
function describeOrder(order) {
  const handler = new OrderHandler();
  return handler.handle(order);
}

module.exports = { Handler, OrderHandler, parseOrder, describeOrder };
