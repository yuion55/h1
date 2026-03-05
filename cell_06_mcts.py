# cell_06_mcts.py
"""
CTRL-MATH v5 — Monte Carlo Tree Search Engine
Replaces MPCPlanner with a full MCTS engine.

Nodes are stored as flat NumPy arrays (NOT Python dicts/objects).
UCT computation is vectorized; backpropagation is @njit.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
from numba import njit

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_NODES       = 2048
EXPANSION_K     = 8
SIMULATION_DEPTH = 10
C_EXPLORE       = 1.4

# Node array column indices
_COL_PARENT   = 0   # int64: parent index (-1 = root)
_COL_N        = 1   # int64: visit count
_COL_Q        = 2   # float64: total reward (stored as float)
_COL_DEPTH    = 3   # int64: depth in tree
_N_COLS       = 4


# ── @njit backpropagation ─────────────────────────────────────────────────────
@njit(cache=True)
def backpropagate_jit(
    parent_col: np.ndarray,   # shape (MAX_NODES,) int64
    n_col:      np.ndarray,   # shape (MAX_NODES,) float64 (visit counts)
    q_col:      np.ndarray,   # shape (MAX_NODES,) float64 (total reward)
    node_idx:   int,
    reward:     float,
) -> None:
    """
    Walk from node_idx to root updating N and Q.
    All arithmetic in NumPy scalars — no Python object overhead.
    """
    idx = node_idx
    while idx >= 0:
        n_col[idx] += 1.0
        q_col[idx] += reward
        idx = int(parent_col[idx])


# ── MCTSNodeStore ─────────────────────────────────────────────────────────────
class MCTSNodeStore:
    """
    Flat NumPy storage for MCTS tree nodes.

    Arrays:
      parent[i]   — parent index (-1 for root)
      n[i]        — visit count
      q[i]        — total reward
      depth[i]    — depth from root
      children[i] — list of child indices (Python list, only for bookkeeping)
      texts[i]    — partial solution text for each node
    """

    def __init__(self, max_nodes: int = MAX_NODES):
        self.max_nodes = max_nodes
        self.parent    = np.full(max_nodes, -1, dtype=np.int64)
        self.n         = np.zeros(max_nodes, dtype=np.float64)
        self.q         = np.zeros(max_nodes, dtype=np.float64)
        self.depth     = np.zeros(max_nodes, dtype=np.int64)
        self._children: List[List[int]] = [[] for _ in range(max_nodes)]
        self._texts: List[str] = [""] * max_nodes
        self._size = 0

    def alloc(self, parent_idx: int, depth: int, text: str = "") -> int:
        """Allocate a new node; return its index. Returns -1 if full."""
        if self._size >= self.max_nodes:
            return -1
        idx = self._size
        self.parent[idx] = parent_idx
        self.n[idx]      = 0.0
        self.q[idx]      = 0.0
        self.depth[idx]  = depth
        self._children[idx] = []
        self._texts[idx]    = text
        self._size += 1
        return idx

    def add_child(self, parent_idx: int, child_idx: int) -> None:
        """Register child_idx as a child of parent_idx."""
        self._children[parent_idx].append(child_idx)

    def uct_score_children(self, parent_idx: int, c: float = C_EXPLORE) -> np.ndarray:
        """
        Vectorized UCT for all children of parent_idx.
        score = Q/N + c * sqrt(ln(N_parent) / N_child)
        Unvisited children get score=inf.
        Returns array of scores aligned with children list.
        """
        children = self._children[parent_idx]
        if not children:
            return np.empty(0, dtype=np.float64)

        idx  = np.array(children, dtype=np.int64)
        n_p  = max(self.n[parent_idx], 1.0)
        n_c  = self.n[idx]
        q_c  = self.q[idx]

        # Unvisited nodes → inf
        unvisited = n_c == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.where(
                unvisited,
                np.inf,
                q_c / n_c + c * np.sqrt(np.log(n_p) / n_c),
            )
        return scores

    def best_child(self, parent_idx: int, c: float = C_EXPLORE) -> int:
        """Return index of child with highest UCT score."""
        children = self._children[parent_idx]
        if not children:
            return -1
        scores = self.uct_score_children(parent_idx, c)
        return children[int(np.argmax(scores))]

    def best_path_leaf(self, root: int = 0) -> int:
        """
        Walk greedily from root to a leaf using best_child.
        Returns the leaf index.
        """
        idx = root
        while self._children[idx]:
            idx = self.best_child(idx)
        return idx

    def size(self) -> int:
        return self._size

    def reset(self) -> None:
        """Reset store to empty state."""
        self.parent[:] = -1
        self.n[:]      = 0.0
        self.q[:]      = 0.0
        self.depth[:]  = 0
        for i in range(self.max_nodes):
            self._children[i] = []
            self._texts[i]    = ""
        self._size = 0


# ── MCTSEngine ────────────────────────────────────────────────────────────────
class MCTSEngine:
    """
    Full MCTS solver for math problems.

    Components:
      - MCTSNodeStore for flat array storage
      - LLM executor for expansion and simulation
      - PRM (optional) for step scoring
    """

    def __init__(
        self,
        llm_executor=None,
        prm=None,
        n_simulations: int = 64,
        max_nodes: int     = MAX_NODES,
        expansion_k: int   = EXPANSION_K,
        sim_depth: int     = SIMULATION_DEPTH,
        c_explore: float   = C_EXPLORE,
    ):
        self.llm         = llm_executor
        self.prm         = prm
        self.n_sim       = n_simulations
        self.store       = MCTSNodeStore(max_nodes)
        self.expansion_k = expansion_k
        self.sim_depth   = sim_depth
        self.c_explore   = c_explore

    def solve(self, problem: str, budget_seconds: float = 60.0) -> Tuple[int, str]:
        """
        Run MCTS to solve `problem`.
        Returns (answer_int, best_text).
        """
        import time

        # Without an LLM, MCTS cannot expand nodes — return early so that
        # the caller's fallback logic (e.g. SymPy) is tried instead of
        # extracting a spurious integer from the problem text itself.
        if self.llm is None:
            return 0, ""

        self.store.reset()
        root = self.store.alloc(-1, 0, problem)

        deadline = time.perf_counter() + budget_seconds
        for _ in range(self.n_sim):
            if time.perf_counter() > deadline:
                break
            if self.store.size() >= self.store.max_nodes - self.expansion_k:
                break

            leaf = self.store.best_path_leaf(root)
            children = self._expand(leaf)
            if not children:
                reward = self._simulate(leaf)
                backpropagate_jit(
                    self.store.parent,
                    self.store.n,
                    self.store.q,
                    leaf,
                    reward,
                )
            else:
                for child in children:
                    reward = self._simulate(child)
                    backpropagate_jit(
                        self.store.parent,
                        self.store.n,
                        self.store.q,
                        child,
                        reward,
                    )

        best_leaf = self.store.best_path_leaf(root)
        text      = self.store._texts[best_leaf]
        answer    = self._extract_answer_from_text(text)
        return answer, text

    def _expand(self, node_idx: int) -> List[int]:
        """
        Use LLM to propose EXPANSION_K continuations of the node's text.
        Returns list of new child indices.
        """
        if self.llm is None:
            return []

        parent_text = self.store._texts[node_idx]
        parent_depth = int(self.store.depth[node_idx])

        try:
            proposals = self.llm.propose_steps_batched(
                parent_text, k=self.expansion_k
            )
        except Exception:
            proposals = []

        children = []
        for step_text in proposals[:self.expansion_k]:
            new_text = parent_text + "\n" + step_text if parent_text else step_text
            child_idx = self.store.alloc(node_idx, parent_depth + 1, new_text)
            if child_idx == -1:
                break
            self.store.add_child(node_idx, child_idx)
            children.append(child_idx)
        return children

    def _simulate(self, node_idx: int) -> float:
        """
        Random rollout from node_idx for sim_depth steps.
        Returns reward in [0, 1].
        """
        text = self.store._texts[node_idx]
        if self.llm is None:
            return 0.0

        for _ in range(self.sim_depth):
            try:
                steps = self.llm.propose_steps_batched(text, k=1)
                if steps:
                    text = text + "\n" + steps[0]
            except Exception:
                break

        if self.prm is not None:
            try:
                steps_list = [s for s in text.split("\n") if s.strip()]
                scores = self.prm.score_batch(steps_list)
                return float(np.mean(scores)) if len(scores) > 0 else 0.5
            except Exception:
                pass

        answer = self._extract_answer_from_text(text)
        return 1.0 if answer != 0 else 0.0

    def _extract_answer_from_text(self, text: str) -> int:
        """Extract integer answer from text. Returns 0 on failure."""
        if not text:
            return 0
        # Try \boxed{N}
        m = re.search(r"\\boxed\{([^}]+)\}", text)
        if m:
            try:
                return int(m.group(1).strip())
            except ValueError:
                pass
        # Try "ANSWER: N"
        m = re.search(r"ANSWER:\s*(-?\d+)", text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        # Last integer fallback
        nums = re.findall(r"-?\d+", text)
        if nums:
            try:
                return int(nums[-1])
            except ValueError:
                pass
        return 0


# ── Warm up JIT ───────────────────────────────────────────────────────────────
def _warmup_jit() -> None:
    """Pre-compile @njit functions at import time."""
    _p  = np.full(8, -1, dtype=np.int64)
    _n  = np.zeros(8, dtype=np.float64)
    _q  = np.zeros(8, dtype=np.float64)
    backpropagate_jit(_p, _n, _q, 0, 1.0)


_warmup_jit()
