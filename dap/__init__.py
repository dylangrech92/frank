
"""DAP (Debug Adapter Protocol) transport-layer client.

Provides ``DAPClient``, a protocol-agnostic JSON-RPC client that speaks the DAP wire format
over stdio or TCP subprocess transports.  Messages are framed with Content-Length headers
(standardised by LSP and DAP base protocols) and routed inside ``DAPClient`` by the ``type``
field:

* ``"request\"`` → a request from the adapter (e.g. "startDebugging") — answered via a reverse-request handler.
* ``"response\"`` → an answer to a previously-sent request — delivered to the corresponding ``_Pending`` waiter.
* ``"event\"``    → a notification from the adapter — dispatched to ``on_event`` listeners and buffered for ``wait_event`` waiters.

See jsonrpc.py for the shared framing primitives.
"""
