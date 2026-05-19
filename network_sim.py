"""
=============================================================================
network_sim.py — Network Topology Simulator
=============================================================================
PURPOSE:
    Builds and manages a simulated computer network as a weighted graph.
    Each node represents a network host (router, server, workstation).
    Each edge represents a physical/logical link with a cost (latency/hops).

AI TYPE USED: Graph-based environment model
    - This is a "World Model" — a structured representation of the environment
      that agents perceive and reason about.
    - The graph acts as the shared knowledge space all agents operate within.

PYTHON COMPATIBILITY: Python 3.13 / 3.14
MODULES USED:
    - networkx  : industry-standard graph library, fully supports Py3.13+
    - random    : built-in, for generating realistic random topologies
    - dataclasses: built-in structured node metadata

=============================================================================
"""

import networkx as nx          # Graph data structure and algorithms
import random                  # Random topology generation
from dataclasses import dataclass, field  # Clean node/edge metadata
from typing import Optional    # Type hints for Python 3.13+


# ---------------------------------------------------------------------------
# NODE METADATA — Each network host has a type and status
# ---------------------------------------------------------------------------
@dataclass
class NetworkNode:
    """
    Represents a single host in the simulated network.

    Attributes:
        node_id   : Unique identifier (e.g., "N1", "Router-A")
        node_type : Role of the host — 'router', 'server', or 'workstation'
        ip        : Simulated IPv4 address
        status    : 'online' or 'offline' — agents skip offline nodes
        hops      : Number of hops from the search origin (filled at runtime)
    """
    node_id   : str
    node_type : str              # 'router' | 'server' | 'workstation'
    ip        : str
    status    : str = "online"   # Default all nodes online
    hops      : int = 0          # Distance from search source (BFS depth)
    data      : dict = field(default_factory=dict)  # Payload the node holds


# ---------------------------------------------------------------------------
# NETWORK GRAPH — The shared environment all agents traverse
# ---------------------------------------------------------------------------
class NetworkGraph:
    """
    Simulates a computer network as a weighted undirected graph.

    DESIGN PATTERN — Environment Model:
        Agents do NOT own this graph. They receive a reference to it and
        query it. This separation of "world" and "actor" is a core principle
        in Multi-Agent Systems (MAS).

    Graph Type: NetworkX Graph (undirected, weighted edges)
        - Nodes: NetworkNode objects stored as node attributes
        - Edges: weighted by 'cost' (simulating link latency in ms)
    """

    def __init__(self, num_nodes: int = 12, seed: int = 42):
        """
        Constructor: Build a random but realistic network topology.

        Args:
            num_nodes : How many hosts to simulate (default 12)
            seed      : Random seed for reproducibility in demos
        """
        random.seed(seed)                  # Reproducible topology
        self.G = nx.Graph()                # Undirected graph — links are bidirectional
        self.nodes: dict[str, NetworkNode] = {}   # Fast lookup by node_id
        self._build_topology(num_nodes)

    # -----------------------------------------------------------------------
    def _build_topology(self, num_nodes: int):
        """
        LOGIC: Generate a realistic tiered network topology.

        TIERS (mimics real enterprise networks):
            Tier 1 — Core Routers   (2 nodes)  : backbone, high connectivity
            Tier 2 — Servers        (4 nodes)  : services (web, DB, file, DNS)
            Tier 3 — Workstations   (rest)     : end-user hosts

        ALGORITHM: Barabási–Albert preferential attachment
            - Real networks are NOT random — popular nodes attract more links.
            - BA model replicates this "rich-get-richer" property seen in
              the internet topology.
            - nx.barabasi_albert_graph(n, m) adds each new node with m edges
              to existing high-degree nodes.
        """
        # Generate scale-free graph (mirrors real internet structure)
        ba_graph = nx.barabasi_albert_graph(num_nodes, m=2, seed=42)

        # Assign node types by index (first 2 = routers, next 4 = servers)
        type_map = {}
        for i in range(num_nodes):
            if i < 2:
                type_map[i] = "router"
            elif i < 6:
                type_map[i] = "server"
            else:
                type_map[i] = "workstation"

        # Build NetworkNode objects and add to our graph
        for i in ba_graph.nodes():
            node_type = type_map[i]
            node_id   = f"N{i}"

            # Simulate realistic IPv4 addresses per tier
            if node_type == "router":
                ip = f"10.0.0.{i+1}"
            elif node_type == "server":
                ip = f"192.168.1.{i+10}"
            else:
                ip = f"192.168.2.{i+100}"

            # Randomly drop ~15% of workstations to simulate offline hosts
            status = "offline" if (node_type == "workstation" and random.random() < 0.15) else "online"

            # Create node with sample searchable data payload
            node_obj = NetworkNode(
                node_id   = node_id,
                node_type = node_type,
                ip        = ip,
                status    = status,
                data      = {
                    "services": random.sample(
                        ["HTTP", "SSH", "FTP", "DNS", "SMTP", "SNMP", "RDP"],
                        k=random.randint(1, 3)
                    ),
                    "os": random.choice(["Linux", "Windows Server", "FreeBSD", "Ubuntu"]),
                    "open_ports": random.sample(range(20, 9000), k=random.randint(2, 5))
                }
            )

            self.nodes[node_id] = node_obj
            # Add node to NetworkX graph with metadata
            self.G.add_node(node_id, **{
                "type"   : node_type,
                "ip"     : ip,
                "status" : status,
                "data"   : node_obj.data
            })

        # Add edges with random latency costs (in ms)
        for u, v in ba_graph.edges():
            cost = random.randint(1, 20)   # Latency: 1–20 ms per hop
            self.G.add_edge(f"N{u}", f"N{v}", cost=cost)

    # -----------------------------------------------------------------------
    def get_neighbors(self, node_id: str) -> list[str]:
        """
        Return the list of neighboring node IDs for a given node.
        Agents call this to know where they CAN move next.
        Only returns ONLINE neighbors (dead hosts are not traversable).
        """
        return [
            nbr for nbr in self.G.neighbors(node_id)
            if self.nodes[nbr].status == "online"
        ]

    # -----------------------------------------------------------------------
    def shortest_path(self, source: str, target: str) -> Optional[list[str]]:
        """
        ALGORITHM: Dijkstra's Shortest Path
            - Finds minimum-cost route between two nodes.
            - Used by agents to plan efficient search routes.
            - Weight = 'cost' (edge latency in ms).
            - Returns None if no path exists (disconnected graph segment).
        """
        try:
            return nx.dijkstra_path(self.G, source, target, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    # -----------------------------------------------------------------------
    def to_json(self) -> dict:
        """
        Serialize the graph to a JSON-compatible dict for the frontend.
        The JavaScript visualization (vis.js / D3) consumes this format.
        """
        nodes_out = []
        for nid, node in self.nodes.items():
            nodes_out.append({
                "id"       : nid,
                "label"    : f"{nid}\n{node.ip}",
                "type"     : node.node_type,
                "ip"       : node.ip,
                "status"   : node.status,
                "data"     : node.data,
                "group"    : node.node_type,   # vis.js group for colour-coding
            })

        edges_out = []
        for u, v, attr in self.G.edges(data=True):
            edges_out.append({
                "from"  : u,
                "to"    : v,
                "label" : f"{attr.get('cost', 1)}ms",
                "cost"  : attr.get("cost", 1)
            })

        return {"nodes": nodes_out, "edges": edges_out}
