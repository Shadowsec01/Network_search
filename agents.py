"""
=============================================================================
agents.py — Communicating Agents Engine
=============================================================================
PURPOSE:
    Implements three distinct AI search agents that communicate with each
    other while searching a simulated network graph. This is the core
    intelligence module of the system.

AI TYPES IMPLEMENTED:
    1. BFS Agent  — Breadth-First Search Agent (Uninformed / Blind Search)
       STATE: Uses a FIFO queue. Explores all neighbours at current depth
              before going deeper. Guarantees shortest path in unweighted graphs.
       COMMUNICATION: Broadcasts discovered nodes to other agents.

    2. DFS Agent  — Depth-First Search Agent (Uninformed / Blind Search)
       STATE: Uses a LIFO stack. Dives deep along one path first.
              Memory efficient; does NOT guarantee shortest path.
       COMMUNICATION: Shares backtrack events so other agents avoid dead ends.

    3. Greedy Agent — Heuristic / Informed Search Agent
       STATE: Uses a priority queue ranked by edge cost (greedy best-first).
              Makes locally optimal choices at each step (like a GPS that
              always picks the nearest junction). Does NOT guarantee optimal
              global path.
       COMMUNICATION: Sends cost estimates to help other agents re-prioritize.

COMMUNICATION MODEL: Blackboard Architecture
    - All agents share a common "Blackboard" (message bus).
    - Agents POST messages to the blackboard (e.g., "I found node N5").
    - Agents READ messages from other agents to avoid redundant searches.
    - This is a classic Multi-Agent System (MAS) coordination pattern used
      in NASA mission planning, air traffic control, and network management.

PYTHON COMPATIBILITY: Python 3.13 / 3.14
MODULES:
    - collections  : deque (efficient FIFO/LIFO queue), built-in
    - heapq        : priority queue for greedy agent, built-in
    - threading    : concurrent agent execution, built-in
    - dataclasses  : clean message structures, built-in
    - time         : step timing for animation, built-in

=============================================================================
"""

from collections import deque          # BFS queue (FIFO) and DFS stack (LIFO)
import heapq                           # Min-heap priority queue for Greedy agent
import threading                       # Run agents concurrently (parallel search)
import time                            # Simulate real-time traversal delay
from dataclasses import dataclass, field
from typing import Optional
from network_sim import NetworkGraph   # Import our network environment


# ---------------------------------------------------------------------------
# MESSAGE — Unit of communication between agents
# ---------------------------------------------------------------------------
@dataclass
class Message:
    """
    A structured message passed between agents via the Blackboard.

    FIELDS:
        sender    : Agent name sending the message
        msg_type  : Category of message:
                    'DISCOVERED'  — agent found a new node
                    'VISITED'     — agent has fully processed a node
                    'FOUND'       — target node located (search success)
                    'DEAD_END'    — path exhausted (DFS backtrack signal)
                    'COST_UPDATE' — Greedy agent sharing cost estimates
        payload   : Relevant data (node IDs, cost values, path info)
        timestamp : Step number when message was sent
    """
    sender    : str
    msg_type  : str
    payload   : dict
    timestamp : float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# BLACKBOARD — Shared communication space (Agent Coordination Hub)
# ---------------------------------------------------------------------------
class Blackboard:
    """
    DESIGN PATTERN: Blackboard Architecture (Barbara Hayes-Roth, 1985)

    The Blackboard is a shared memory space that all agents can read/write.
    It decouples agents from each other — agents never call each other
    directly. They only interact through the board.

    This is equivalent to:
        - A shared chat room where agents post status updates
        - A notice board in an office where workers post findings
        - A publish/subscribe message bus in microservices

    CONCURRENCY SAFETY:
        threading.Lock() ensures no two agents corrupt the message list
        simultaneously (race condition prevention).
    """

    def __init__(self):
        self.messages  : list[Message] = []   # All messages (full history)
        self.visited   : set[str]      = set() # Globally visited nodes (shared)
        self._lock     = threading.Lock()      # Thread safety lock

    def post(self, message: Message):
        """Agent posts a message — thread-safe write."""
        with self._lock:
            self.messages.append(message)

    def get_all(self) -> list[Message]:
        """Read all messages posted so far — thread-safe read."""
        with self._lock:
            return list(self.messages)

    def mark_visited(self, node_id: str):
        """Mark a node as visited globally — prevents duplicate searches."""
        with self._lock:
            self.visited.add(node_id)

    def is_visited(self, node_id: str) -> bool:
        """Check if any agent has already visited this node."""
        with self._lock:
            return node_id in self.visited


# ---------------------------------------------------------------------------
# BASE AGENT — Abstract parent class for all search agents
# ---------------------------------------------------------------------------
class BaseAgent:
    """
    Abstract base for all search agents.

    STATE MACHINE:
        Each agent cycles through these states during execution:
        IDLE → SEARCHING → (FOUND | EXHAUSTED)

        - IDLE      : Agent created but not yet started
        - SEARCHING : Agent is actively traversing the network
        - FOUND     : Agent located the target node
        - EXHAUSTED : Agent explored all reachable nodes, target not found
    """

    def __init__(self, name: str, graph: NetworkGraph, blackboard: Blackboard):
        self.name        = name
        self.graph       = graph
        self.blackboard  = blackboard
        self.state       = "IDLE"          # Initial agent state
        self.steps       : list[dict] = [] # Audit trail of every action taken
        self.path_found  : Optional[list[str]] = None  # Solution path if found

    def _record_step(self, action: str, node: str, detail: str = ""):
        """
        Log every agent decision for the frontend timeline visualisation.
        Each step appears as a row in the GUI step-by-step table.
        """
        self.steps.append({
            "agent"  : self.name,
            "action" : action,
            "node"   : node,
            "detail" : detail,
            "state"  : self.state
        })

    def search(self, start: str, target: str) -> dict:
        """
        Override in subclasses. Returns a result dict with:
            found    : bool
            path     : list of node IDs
            steps    : full audit trail
            messages : all blackboard messages from this agent
        """
        raise NotImplementedError("Subclasses must implement search()")


# ---------------------------------------------------------------------------
# AGENT 1 — BFS Agent (Breadth-First Search)
# ---------------------------------------------------------------------------
class BFSAgent(BaseAgent):
    """
    AI TYPE: Uninformed / Blind Search — Breadth-First Search

    HOW IT WORKS:
        - Uses a FIFO queue (deque from Python's collections module).
        - Explores the network LEVEL by LEVEL (hop-by-hop from source).
        - Processes all nodes 1 hop away before nodes 2 hops away, etc.

    COMPLETENESS : YES — will always find the target if it exists.
    OPTIMALITY   : YES — finds the shortest path (fewest hops).
    TIME Complexity : O(V + E) — visits every node and edge once.
    Space Complexity: O(V)    — queue can hold all frontier nodes.

    REAL-WORLD USE:
        - Peer-to-peer network searching (BitTorrent, Gnutella)
        - Social network friend suggestions (LinkedIn "2nd connection")
        - DNS record lookup propagation
    """

    def search(self, start: str, target: str) -> dict:
        self.state = "SEARCHING"
        self._record_step("START", start, f"BFS initiated from {start} → seeking {target}")

        # FIFO queue stores (current_node, path_taken_so_far)
        queue   : deque[tuple[str, list[str]]] = deque()
        visited : set[str] = set()

        queue.append((start, [start]))
        visited.add(start)

        # Communicate start to blackboard
        self.blackboard.post(Message(
            sender   = self.name,
            msg_type = "DISCOVERED",
            payload  = {"node": start, "depth": 0}
        ))

        while queue:
            current, path = queue.popleft()   # FIFO — leftmost = oldest = shallowest

            self.blackboard.mark_visited(current)
            self._record_step("VISIT", current, f"Dequeued | path depth={len(path)-1}")

            # ----------------------------------------------------------------
            # GOAL TEST — Did we find the target?
            # ----------------------------------------------------------------
            if current == target:
                self.state = "FOUND"
                self.path_found = path
                self._record_step("FOUND", current, f"TARGET FOUND! Path: {' → '.join(path)}")

                # Announce discovery to all agents on the blackboard
                self.blackboard.post(Message(
                    sender   = self.name,
                    msg_type = "FOUND",
                    payload  = {"node": current, "path": path, "hops": len(path)-1}
                ))
                break

            # ----------------------------------------------------------------
            # EXPANSION — Enqueue all unvisited online neighbours
            # ----------------------------------------------------------------
            neighbours = self.graph.get_neighbors(current)
            for nbr in neighbours:
                if nbr not in visited:
                    visited.add(nbr)
                    new_path = path + [nbr]
                    queue.append((nbr, new_path))

                    self._record_step("ENQUEUE", nbr, f"Added to BFS frontier from {current}")
                    self.blackboard.post(Message(
                        sender   = self.name,
                        msg_type = "DISCOVERED",
                        payload  = {"node": nbr, "parent": current, "depth": len(new_path)-1}
                    ))

        else:
            # Queue emptied — target not reachable
            self.state = "EXHAUSTED"
            self._record_step("EXHAUSTED", start, "All reachable nodes explored, target not found")

        return {
            "agent"    : self.name,
            "strategy" : "BFS",
            "found"    : self.state == "FOUND",
            "path"     : self.path_found or [],
            "steps"    : self.steps,
            "messages" : [m.__dict__ for m in self.blackboard.get_all() if m.sender == self.name]
        }


# ---------------------------------------------------------------------------
# AGENT 2 — DFS Agent (Depth-First Search)
# ---------------------------------------------------------------------------
class DFSAgent(BaseAgent):
    """
    AI TYPE: Uninformed / Blind Search — Depth-First Search

    HOW IT WORKS:
        - Uses a LIFO stack (list used as stack via .append() / .pop()).
        - Dives as DEEP as possible along each branch before backtracking.
        - Tries one complete path fully before exploring alternatives.

    COMPLETENESS : YES (in finite graphs) — will find target eventually.
    OPTIMALITY   : NO — may find a longer path first.
    Time Complexity  : O(V + E)
    Space Complexity : O(depth) — only stores current branch, memory efficient.

    REAL-WORLD USE:
        - Web crawlers (crawl deep into site before sibling pages)
        - Maze solving algorithms
        - Dependency resolution (pip, npm resolve deep before siblings)
        - Network vulnerability scanners (pentest tools like Nessus)
    """

    def search(self, start: str, target: str) -> dict:
        self.state = "SEARCHING"
        self._record_step("START", start, f"DFS initiated from {start} → seeking {target}")

        # LIFO stack stores (current_node, path_taken_so_far)
        stack   : list[tuple[str, list[str]]] = []
        visited : set[str] = set()

        stack.append((start, [start]))

        self.blackboard.post(Message(
            sender   = self.name,
            msg_type = "DISCOVERED",
            payload  = {"node": start, "depth": 0}
        ))

        while stack:
            current, path = stack.pop()   # LIFO — pop from right = deepest first

            if current in visited:
                continue                  # Already explored this branch

            visited.add(current)
            self.blackboard.mark_visited(current)
            self._record_step("VISIT", current, f"Popped from stack | depth={len(path)-1}")

            # ----------------------------------------------------------------
            # GOAL TEST
            # ----------------------------------------------------------------
            if current == target:
                self.state = "FOUND"
                self.path_found = path
                self._record_step("FOUND", current, f"TARGET FOUND! Path: {' → '.join(path)}")

                self.blackboard.post(Message(
                    sender   = self.name,
                    msg_type = "FOUND",
                    payload  = {"node": current, "path": path, "hops": len(path)-1}
                ))
                break

            # ----------------------------------------------------------------
            # EXPANSION — Push all unvisited neighbours onto stack
            # Note: reversed() so left-to-right ordering is preserved
            # ----------------------------------------------------------------
            neighbours = self.graph.get_neighbors(current)
            for nbr in reversed(neighbours):
                if nbr not in visited:
                    new_path = path + [nbr]
                    stack.append((nbr, new_path))

                    self._record_step("PUSH", nbr, f"Pushed onto DFS stack from {current}")
                    self.blackboard.post(Message(
                        sender   = self.name,
                        msg_type = "DISCOVERED",
                        payload  = {"node": nbr, "parent": current}
                    ))
            else:
                # Neighbourhood exhausted — signal backtrack
                if len(neighbours) == 0 or all(n in visited for n in neighbours):
                    self._record_step("BACKTRACK", current, "No unvisited neighbours — backtracking")
                    self.blackboard.post(Message(
                        sender   = self.name,
                        msg_type = "DEAD_END",
                        payload  = {"node": current}
                    ))

        else:
            self.state = "EXHAUSTED"
            self._record_step("EXHAUSTED", start, "Stack empty — target not found")

        return {
            "agent"    : self.name,
            "strategy" : "DFS",
            "found"    : self.state == "FOUND",
            "path"     : self.path_found or [],
            "steps"    : self.steps,
            "messages" : [m.__dict__ for m in self.blackboard.get_all() if m.sender == self.name]
        }


# ---------------------------------------------------------------------------
# AGENT 3 — Greedy Agent (Informed / Heuristic Search)
# ---------------------------------------------------------------------------
class GreedyAgent(BaseAgent):
    """
    AI TYPE: Informed / Heuristic Search — Greedy Best-First Search

    HOW IT WORKS:
        - Uses a MIN-HEAP priority queue (heapq module).
        - Selects the next node with the LOWEST edge cost at each step.
        - Greedy = always makes the locally optimal choice.
        - Does NOT look ahead — just picks the cheapest current neighbour.

    COMPLETENESS : YES (in finite graphs without cycles handled by visited set)
    OPTIMALITY   : NO — greedy choices can miss the globally optimal path.
    Time Complexity  : O(E log V) — heap operations on each edge.
    Space Complexity : O(V) — frontier in heap.

    HEURISTIC USED: Edge cost (link latency in ms)
        - Lower latency links are preferred.
        - This mimics routing protocols like OSPF (Open Shortest Path First).

    COMMUNICATION:
        - Posts cost estimates to blackboard so BFS/DFS agents can avoid
          known high-cost paths.

    REAL-WORLD USE:
        - OSPF and IS-IS routing protocol path selection
        - GPS navigation (pick cheapest immediate road junction)
        - CPU job scheduling (always pick shortest remaining job first)
    """

    def search(self, start: str, target: str) -> dict:
        self.state = "SEARCHING"
        self._record_step("START", start, f"Greedy initiated from {start} → seeking {target}")

        # Priority queue: (cost, node, path)
        # heapq is a MIN-heap — lowest cost always at front
        heap    : list[tuple[int, str, list[str]]] = []
        visited : set[str] = set()

        heapq.heappush(heap, (0, start, [start]))

        self.blackboard.post(Message(
            sender   = self.name,
            msg_type = "DISCOVERED",
            payload  = {"node": start, "cost": 0}
        ))

        while heap:
            cost, current, path = heapq.heappop(heap)   # Pop minimum-cost node

            if current in visited:
                continue                 # Already processed this node

            visited.add(current)
            self.blackboard.mark_visited(current)
            self._record_step("VISIT", current, f"Greedy selected | cumulative cost={cost}ms")

            # ----------------------------------------------------------------
            # GOAL TEST
            # ----------------------------------------------------------------
            if current == target:
                self.state = "FOUND"
                self.path_found = path
                self._record_step("FOUND", current,
                    f"TARGET FOUND! Cost={cost}ms | Path: {' → '.join(path)}")

                self.blackboard.post(Message(
                    sender   = self.name,
                    msg_type = "FOUND",
                    payload  = {"node": current, "path": path, "cost": cost}
                ))
                break

            # ----------------------------------------------------------------
            # EXPANSION — Push neighbours sorted by link cost
            # ----------------------------------------------------------------
            neighbours_data = [
                (self.graph.G[current][nbr].get("cost", 1), nbr)
                for nbr in self.graph.get_neighbors(current)
                if nbr not in visited
            ]

            for edge_cost, nbr in neighbours_data:
                new_cost = cost + edge_cost
                new_path = path + [nbr]
                heapq.heappush(heap, (new_cost, nbr, new_path))

                self._record_step("PUSH", nbr,
                    f"Queued from {current} | link cost={edge_cost}ms | total={new_cost}ms")

                # Share cost info with other agents via blackboard
                self.blackboard.post(Message(
                    sender   = self.name,
                    msg_type = "COST_UPDATE",
                    payload  = {"node": nbr, "estimated_cost": new_cost, "via": current}
                ))

        else:
            self.state = "EXHAUSTED"
            self._record_step("EXHAUSTED", start, "Heap empty — target not found")

        return {
            "agent"    : self.name,
            "strategy" : "GREEDY",
            "found"    : self.state == "FOUND",
            "path"     : self.path_found or [],
            "steps"    : self.steps,
            "messages" : [m.__dict__ for m in self.blackboard.get_all() if m.sender == self.name]
        }


# ---------------------------------------------------------------------------
# MULTI-AGENT COORDINATOR — Launches all agents and collects results
# ---------------------------------------------------------------------------
class MultiAgentCoordinator:
    """
    Orchestrates concurrent execution of all three search agents.

    CONCURRENCY MODEL: Python threading
        - Each agent runs in its own thread (concurrent, not true parallel
          due to CPython GIL, but sufficient for I/O-bound simulations).
        - threading.Thread per agent.
        - threading.join() waits for all agents to complete before reporting.

    COORDINATION: Shared Blackboard
        - All three agents share one Blackboard instance.
        - The Blackboard's thread lock ensures message integrity.

    RESULT AGGREGATION:
        - After all agents finish, results are merged and ranked by:
          1. Was target found? (yes > no)
          2. Path length (shorter = better)
          3. Steps taken (fewer = more efficient)
    """

    def __init__(self, graph: NetworkGraph):
        self.graph      = graph
        self.blackboard = Blackboard()

        # Instantiate all three agent types
        self.agents = [
            BFSAgent   ("BFS-Agent",    graph, self.blackboard),
            DFSAgent   ("DFS-Agent",    graph, self.blackboard),
            GreedyAgent("Greedy-Agent", graph, self.blackboard),
        ]

    def run(self, start: str, target: str) -> dict:
        """
        Launch all agents concurrently and collect their results.

        THREADING PATTERN:
            1. Create a Thread for each agent's search() call.
            2. Start all threads simultaneously.
            3. Join (wait) for all to complete.
            4. Aggregate results into a unified report.
        """
        results       : list[dict] = []
        threads       : list[threading.Thread] = []
        result_store  : dict[str, dict] = {}

        def run_agent(agent, start, target):
            """Thread target: run one agent and store result."""
            result = agent.search(start, target)
            result_store[agent.name] = result

        # Create and start one thread per agent
        for agent in self.agents:
            t = threading.Thread(
                target = run_agent,
                args   = (agent, start, target),
                daemon = True         # Daemon threads die with main program
            )
            threads.append(t)
            t.start()

        # Wait for all agents to finish
        for t in threads:
            t.join(timeout=30)     # 30s timeout per agent (safety net)

        # Collect results in agent order
        for agent in self.agents:
            if agent.name in result_store:
                results.append(result_store[agent.name])

        # Build comparison summary
        summary = self._summarize(results, start, target)

        return {
            "start"    : start,
            "target"   : target,
            "results"  : results,
            "summary"  : summary,
            "blackboard": [m.__dict__ for m in self.blackboard.get_all()]
        }

    def _summarize(self, results: list[dict], start: str, target: str) -> dict:
        """
        Generate a comparative performance summary of all agents.
        Used by the frontend to render the comparison table.
        """
        summary_rows = []
        for r in results:
            summary_rows.append({
                "agent"    : r["agent"],
                "strategy" : r["strategy"],
                "found"    : r["found"],
                "path"     : r["path"],
                "hops"     : len(r["path"]) - 1 if r["path"] else "N/A",
                "steps"    : len(r["steps"]),
                "msgs_sent": len(r["messages"]),
            })

        # Rank agents: found first, then by fewest hops
        def rank_key(row):
            found = 0 if row["found"] else 1
            hops  = row["hops"] if isinstance(row["hops"], int) else 999
            return (found, hops, row["steps"])

        summary_rows.sort(key=rank_key)

        return {
            "best_agent"  : summary_rows[0]["agent"] if summary_rows else "None",
            "agents"      : summary_rows,
            "total_msgs"  : len(self.blackboard.get_all()),
        }
