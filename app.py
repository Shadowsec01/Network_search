"""
=============================================================================
app.py — Flask Web Application Server (Main Entry Point)
=============================================================================
PURPOSE:
    Serves the web GUI and exposes REST API endpoints that the browser
    JavaScript calls to trigger agent searches, retrieve graph data,
    and stream step-by-step logs.

ARCHITECTURE: Client-Server (MVC pattern)
    ┌─────────────────────────────────────────┐
    │            Browser (HTML/JS)            │  ← View + Controller
    │  Sends HTTP requests, renders results   │
    └──────────────────┬──────────────────────┘
                       │ HTTP REST (JSON)
    ┌──────────────────▼──────────────────────┐
    │           Flask App (app.py)            │  ← Routing Layer
    │  Routes: /api/graph, /api/search, etc.  │
    └──────────────────┬──────────────────────┘
                       │ Python function calls
    ┌──────────────────▼──────────────────────┐
    │  Agents Engine    │  Network Simulator   │  ← Model Layer
    │  (agents.py)      │  (network_sim.py)    │
    └───────────────────┴──────────────────────┘

WEB FRAMEWORK: Flask (lightweight, no ORM overhead needed for this project)
    - flask       : HTTP routing, template rendering
    - flask-cors  : Allow browser JS to call our API (CORS headers)

PYTHON COMPATIBILITY: Python 3.13 / 3.14
    All imports tested against latest Flask (3.x) and networkx (3.x)

HOW TO RUN:
    pip install flask flask-cors networkx
    python app.py
    → Open http://127.0.0.1:5000 in your browser

=============================================================================
"""

from flask import Flask, jsonify, request, render_template, abort
from flask_cors import CORS        # Cross-Origin Resource Sharing for JS fetch()
import json
import os
import pathlib

# Import our custom modules
from network_sim import NetworkGraph
from agents import MultiAgentCoordinator

# ---------------------------------------------------------------------------
# APPLICATION FACTORY
# ---------------------------------------------------------------------------
import pathlib

# ---------------------------------------------------------------------------
# TEMPLATE DIRECTORY — resolved relative to this file so it works regardless
# of the working directory the user launches app.py from.
# ---------------------------------------------------------------------------
_HERE      = pathlib.Path(__file__).parent.resolve()
_TEMPLATES = _HERE / "templates"

app = Flask(__name__, template_folder=str(_TEMPLATES))
CORS(app)                          # Enable CORS so browser JS can call /api/* freely

# ---------------------------------------------------------------------------
# GLOBAL STATE — Single shared network graph instance
# ---------------------------------------------------------------------------
# NOTE: In production, this would be per-session. For a teaching demo,
# one shared graph is appropriate and easier to explain.
NETWORK = NetworkGraph(num_nodes=12, seed=42)


# ---------------------------------------------------------------------------
# ROUTE 1 — Serve the main HTML page
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """
    Render and return the main GUI page.

    Flask looks for templates in the /templates/ folder by default.
    render_template() reads index.html and returns it as an HTTP response.

    HTTP Method : GET
    Returns     : HTML page (text/html)
    """
    return render_template("index.html")


# ---------------------------------------------------------------------------
# ROUTE 2 — Return the network graph as JSON
# ---------------------------------------------------------------------------
@app.route("/api/graph", methods=["GET"])
def get_graph():
    """
    API Endpoint: GET /api/graph

    Returns the full network topology as JSON.
    The frontend vis.js library uses this to draw the interactive graph.

    RESPONSE FORMAT:
    {
        "nodes": [ { "id": "N0", "label": "N0\n10.0.0.1", "type": "router", ... } ],
        "edges": [ { "from": "N0", "to": "N1", "label": "5ms", "cost": 5 } ]
    }

    HTTP Method : GET
    Returns     : JSON (application/json)
    """
    return jsonify(NETWORK.to_json())


# ---------------------------------------------------------------------------
# ROUTE 3 — Return list of all nodes (for source/target dropdowns)
# ---------------------------------------------------------------------------
@app.route("/api/nodes", methods=["GET"])
def get_nodes():
    """
    API Endpoint: GET /api/nodes

    Returns a simplified node list for populating the GUI dropdowns.

    RESPONSE FORMAT:
    [
        { "id": "N0", "type": "router", "ip": "10.0.0.1", "status": "online" },
        ...
    ]

    HTTP Method : GET
    Returns     : JSON array
    """
    nodes = [
        {
            "id"    : nid,
            "type"  : node.node_type,
            "ip"    : node.ip,
            "status": node.status
        }
        for nid, node in NETWORK.nodes.items()
        if node.status == "online"   # Only show online nodes as valid options
    ]
    # Sort: routers first, then servers, then workstations
    type_order = {"router": 0, "server": 1, "workstation": 2}
    nodes.sort(key=lambda n: (type_order.get(n["type"], 3), n["id"]))
    return jsonify(nodes)


# ---------------------------------------------------------------------------
# ROUTE 4 — Run multi-agent search (main AI action)
# ---------------------------------------------------------------------------
@app.route("/api/search", methods=["POST"])
def run_search():
    """
    API Endpoint: POST /api/search

    Receives start and target node IDs from the browser form.
    Instantiates a MultiAgentCoordinator, runs all three agents, and
    returns the full search results as JSON.

    REQUEST BODY (JSON):
    {
        "start"  : "N0",
        "target" : "N8"
    }

    RESPONSE FORMAT:
    {
        "start"   : "N0",
        "target"  : "N8",
        "results" : [ { agent: ..., found: ..., path: [...], steps: [...] } ],
        "summary" : { best_agent: ..., agents: [...], total_msgs: N },
        "blackboard": [ { sender, msg_type, payload, timestamp } ]
    }

    VALIDATION:
        - Both 'start' and 'target' must be present in request body.
        - Both must correspond to ONLINE nodes in the graph.
        - start != target (trivial search not allowed).

    HTTP Method : POST (modifies server state — runs computation)
    Returns     : JSON
    """
    # ------------------------------------------------------------------
    # Parse request body
    # ------------------------------------------------------------------
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    start  = body.get("start", "").strip()
    target = body.get("target", "").strip()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not start or not target:
        return jsonify({"error": "Both 'start' and 'target' node IDs are required"}), 400

    if start == target:
        return jsonify({"error": "Start and target nodes must be different"}), 400

    # Check both nodes exist and are online
    all_nodes = NETWORK.nodes
    if start not in all_nodes:
        return jsonify({"error": f"Node '{start}' does not exist in the network"}), 404
    if target not in all_nodes:
        return jsonify({"error": f"Node '{target}' does not exist in the network"}), 404

    if all_nodes[start].status != "online":
        return jsonify({"error": f"Node '{start}' is offline — choose an online node"}), 400
    if all_nodes[target].status != "online":
        return jsonify({"error": f"Node '{target}' is offline — choose an online node"}), 400

    # ------------------------------------------------------------------
    # Run multi-agent search
    # ------------------------------------------------------------------
    # Fresh coordinator per search — ensures clean blackboard
    coordinator = MultiAgentCoordinator(NETWORK)
    result      = coordinator.run(start, target)

    # Convert timestamp floats to strings for JSON serialisation
    for msg in result.get("blackboard", []):
        msg["timestamp"] = str(msg.get("timestamp", ""))

    for agent_result in result.get("results", []):
        for msg in agent_result.get("messages", []):
            msg["timestamp"] = str(msg.get("timestamp", ""))

    return jsonify(result)


# ---------------------------------------------------------------------------
# ROUTE 5 — Node detail endpoint
# ---------------------------------------------------------------------------
@app.route("/api/node/<node_id>", methods=["GET"])
def get_node_detail(node_id: str):
    """
    API Endpoint: GET /api/node/<node_id>

    Returns full metadata for a specific node including its services,
    OS, open ports, and neighbour list. Called when user clicks a
    node in the graph visualisation.

    PATH PARAMETER: node_id (e.g., "N3")

    RESPONSE FORMAT:
    {
        "id"        : "N3",
        "type"      : "server",
        "ip"        : "192.168.1.13",
        "status"    : "online",
        "data"      : { "services": [...], "os": "...", "open_ports": [...] },
        "neighbours": ["N0", "N1", "N5"]
    }

    HTTP Method : GET
    Returns     : JSON
    """
    if node_id not in NETWORK.nodes:
        return jsonify({"error": f"Node '{node_id}' not found"}), 404

    node = NETWORK.nodes[node_id]
    return jsonify({
        "id"         : node.node_id,
        "type"       : node.node_type,
        "ip"         : node.ip,
        "status"     : node.status,
        "data"       : node.data,
        "neighbours" : NETWORK.get_neighbors(node_id)
    })


# ---------------------------------------------------------------------------
# ROUTE 6 — Health check (for deployment / monitoring)
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    """
    API Endpoint: GET /api/health

    Simple liveness probe. Returns 200 OK when the server is running.
    Used by load balancers, Docker healthchecks, and CI pipelines.

    HTTP Method : GET
    Returns     : JSON { "status": "ok", "nodes": N }
    """
    return jsonify({
        "status" : "ok",
        "nodes"  : len(NETWORK.nodes),
        "edges"  : NETWORK.G.number_of_edges()
    })


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Start the Flask development server.

    debug=True  : Auto-reloads on code changes, shows detailed error pages.
                  NEVER use debug=True in production.
    host="0.0.0.0": Listen on all network interfaces (accessible from LAN).
    port=5000   : Standard Flask port.

    For production deployment, replace with:
        gunicorn -w 4 app:app
    or use any WSGI-compatible server (uWSGI, Waitress on Windows).
    """
    print("=" * 60)
    print("  Communicating Agents Network Search System")
    print("  Flask server starting on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
