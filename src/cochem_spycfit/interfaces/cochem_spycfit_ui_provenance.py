"""DAG Provenance Graph visualizer with Time-Travel reversion mechanics.

Zone C of the SpycFit Interactive Dashboard:
- Directed Acyclic Graph (DAG) state visualization using NetworkX and Plotly.
- Convergence health classification (improving, stable, diverging) based on RMSD differentials.
- Directed edge labels displaying locked assignment count deltas and RMSD shifts.
- Interactive node-click callbacks triggering instant time-travel state reversion.
- High-precision tensor restoration (<100 ms) from SWMR HDF5 Tier 2 cache.
- Ipywidgets container integration for full Jupyter notebook dashboard embedding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Graph and visualization dependencies with graceful degradation
try:
    import networkx as nx
    import plotly.graph_objects as go

    _HAS_GRAPH_DEPS = True
except ImportError:
    _HAS_GRAPH_DEPS = False
    logger.warning("networkx or plotly not available. Provenance graph rendering disabled.")

try:
    import ipywidgets as widgets

    _HAS_WIDGETS = True
except ImportError:
    _HAS_WIDGETS = False
    logger.warning("ipywidgets not available. Interactive widget container disabled.")


# =============================================================================
# 1. EXCEPTION HIERARCHY
# =============================================================================


class ProvenanceGraphError(Exception):
    """Base exception for all DAG provenance graph operations."""


class GraphBuildError(ProvenanceGraphError):
    """Raised when DAG graph construction from CommitManager fails."""


class StateReversionError(ProvenanceGraphError):
    """Raised when time-travel reversion to historical FitState fails."""


# =============================================================================
# 2. PROVENANCE GRAPH RENDERER CLASS (ZONE C)
# =============================================================================


class ProvenanceGraphRenderer:
    """Renders the FitState DAG as an interactive provenance graph with time-travel."""

    # Color mapping for convergence health
    HEALTH_COLORS: dict[str, str] = {
        "improving": "#2ecc71",  # Green — RMSD decreasing (< 0.95x parent)
        "stable": "#f39c12",     # Orange — RMSD stable (0.95x to 1.05x parent)
        "diverging": "#e74c3c",  # Red — RMSD increasing (> 1.05x parent)
    }

    def __init__(
        self,
        commit_manager: object,
        tensor_cache: object,
        on_revert_callback: Callable[[str], None] = lambda _: None,
    ) -> None:
        """Initialize the ProvenanceGraphRenderer.

        Parameters
        ----------
        commit_manager : object
            The DAG state manager holding all FitState nodes.
        tensor_cache : object
            The HDF5 tensor cache for restoring heavy state matrices.
        on_revert_callback : Callable[[str], None], optional
            Callback invoked with state_id when user selects a historical node.
            Defaults to a no-op lambda.
        """
        self._commit_mgr = commit_manager
        self._tensor_cache = tensor_cache
        self._on_revert: Callable[[str], None] = on_revert_callback
        self._graph: Optional[nx.DiGraph] = None
        self._fig: Optional[Any] = None
        self._node_id_registry: List[str] = []

    @property
    def commit_manager(self) -> object:
        """Return the bound CommitManager instance."""
        return self._commit_mgr

    @property
    def tensor_cache(self) -> object:
        """Return the bound TensorCache instance."""
        return self._tensor_cache

    @property
    def graph(self) -> Optional[nx.DiGraph]:
        """Return the underlying NetworkX DiGraph representation."""
        return self._graph

    @property
    def fig(self) -> Optional[Any]:
        """Return the cached Plotly figure object."""
        return self._fig

    @property
    def on_revert_callback(self) -> Callable[[str], None]:
        """Return the registered state reversion callback."""
        return self._on_revert

    def build_graph(self) -> None:
        """Build a NetworkX DiGraph from the CommitManager state registry.

        Each node is a FitState (keyed by state_id).
        Each directed edge points from parent -> child.
        Node attributes include: rmsd, timestamp, n_assignments, health, color.
        Edge attributes include: label, delta_n, delta_rmsd.
        """
        if not _HAS_GRAPH_DEPS:
            logger.error("Cannot build graph: networkx/plotly dependencies are missing.")
            return

        self._graph = nx.DiGraph()

        # Retrieve state dictionary from commit manager
        raw_states = getattr(self._commit_mgr, "states", None)
        if raw_states is None:
            raw_states = getattr(self._commit_mgr, "_states", {})
        states_dict: Dict[str, Any] = dict(raw_states)

        for state_id, state in states_dict.items():
            health = self._classify_health(state, states_dict)
            locked = getattr(state, "locked_assignments", ()) or ()
            n_assign = len(locked)
            rmsd_val = float(getattr(state, "global_rmsd", 0.0))
            ts_str = str(getattr(state, "timestamp", ""))

            self._graph.add_node(
                state_id,
                rmsd=rmsd_val,
                timestamp=ts_str,
                n_assignments=n_assign,
                health=health,
                color=self.HEALTH_COLORS.get(health, self.HEALTH_COLORS["stable"]),
            )

            parent_id = getattr(state, "parent_id", None)
            if parent_id and parent_id in states_dict:
                parent_state = states_dict[parent_id]
                parent_locked = getattr(parent_state, "locked_assignments", ()) or ()
                delta_n = n_assign - len(parent_locked)
                parent_rmsd = float(getattr(parent_state, "global_rmsd", 0.0))
                delta_rmsd = rmsd_val - parent_rmsd
                edge_label = f"Δn={delta_n:+d}, ΔRMSD={delta_rmsd:+.2f} kHz"
                self._graph.add_edge(
                    parent_id,
                    state_id,
                    label=edge_label,
                    delta_n=delta_n,
                    delta_rmsd=delta_rmsd,
                )

    def _classify_health(
        self,
        state: object,
        all_states: Dict[str, Any],
    ) -> str:
        """Classify node convergence health based on RMSD trend relative to parent.

        Parameters
        ----------
        state : object
            FitState to evaluate.
        all_states : dict
            Full dictionary of state objects in DAG.

        Returns
        -------
        str
            'improving', 'diverging', or 'stable'.
        """
        parent_id = getattr(state, "parent_id", None)
        if parent_id is None or parent_id not in all_states:
            return "stable"

        parent_state = all_states[parent_id]
        current_rmsd = float(getattr(state, "global_rmsd", 0.0))
        parent_rmsd = float(getattr(parent_state, "global_rmsd", 0.0))

        if current_rmsd < parent_rmsd * 0.95:
            return "improving"
        elif current_rmsd > parent_rmsd * 1.05:
            return "diverging"
        return "stable"

    def _compute_dag_layout(
        self,
        graph: nx.DiGraph,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute 2D coordinates for DAG nodes using hierarchical tree layout.

        Attempts Graphviz dot layout if available, falling back to a deterministic
        topological level arrangement.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed acyclic graph to position.

        Returns
        -------
        dict
            Mapping from node_id to (x, y) coordinate tuple.
        """
        pos_dict: Optional[Dict[str, Tuple[float, float]]] = None

        if hasattr(nx, "drawing") and hasattr(nx.drawing, "nx_agraph"):
            try:
                raw_pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot")
                if raw_pos and isinstance(raw_pos, dict):
                    pos_dict = {
                        str(k): (float(v[0]), float(v[1])) for k, v in raw_pos.items()
                    }
            except Exception as layout_err:
                logger.debug("Graphviz layout unavailable: %s", layout_err)
                pos_dict = None

        if pos_dict is not None:
            return pos_dict

        # Deterministic hierarchical topological layout fallback
        pos_dict = {}
        if len(graph) == 0:
            return pos_dict

        # Identify root nodes (in-degree == 0)
        roots = [n for n, deg in graph.in_degree() if deg == 0]
        if not roots:
            roots = list(graph.nodes())[:1]

        # Compute depth from root nodes using longest path
        node_depths: Dict[str, int] = {}
        for node in graph.nodes():
            node_depths[node] = 0

        try:
            topo_order = list(nx.topological_sort(graph))
            for u in topo_order:
                for v in graph.successors(u):
                    if node_depths[v] < node_depths[u] + 1:
                        node_depths[v] = node_depths[u] + 1
        except Exception:
            # Fallback if cycles present
            for r in roots:
                for target, length in nx.single_source_shortest_path_length(graph, r).items():
                    if length > node_depths.get(target, 0):
                        node_depths[target] = length

        # Group nodes by depth
        levels: Dict[int, List[str]] = {}
        for n, d in node_depths.items():
            if d not in levels:
                levels[d] = []
            levels[d].append(n)

        for depth_lvl, nodes_at_lvl in levels.items():
            total_at_lvl = len(nodes_at_lvl)
            for idx, n in enumerate(nodes_at_lvl):
                x_coord = (float(idx) - (float(total_at_lvl) - 1.0) / 2.0) * 2.0
                y_coord = -float(depth_lvl) * 2.0
                pos_dict[n] = (x_coord, y_coord)

        return pos_dict

    def render_plotly_figure(self) -> Any:
        """Render the provenance DAG as an interactive Plotly FigureWidget.

        Node color corresponds to convergence health (green/orange/red).
        Node size scales with number of locked assignments.
        Edges show assignment count deltas and RMSD changes.
        Click event triggers instant time-travel state reversion.

        Returns
        -------
        go.FigureWidget or go.Figure
            Interactive Plotly visualization.
        """
        if not _HAS_GRAPH_DEPS:
            logger.error("Cannot render plotly figure: networkx/plotly missing.")
            return None

        if self._graph is None:
            self.build_graph()

        if self._graph is None or len(self._graph) == 0:
            fig_empty = go.Figure()
            fig_empty.update_layout(
                title="🕰️ Provenance Graph — (Empty DAG)",
                template="plotly_white",
            )
            self._fig = fig_empty
            return fig_empty

        pos = self._compute_dag_layout(self._graph)

        # Build node coordinates and styling
        node_x: List[float] = []
        node_y: List[float] = []
        node_colors: List[str] = []
        node_sizes: List[float] = []
        node_labels: List[str] = []
        hover_descriptions: List[str] = []
        self._node_id_registry = []

        for node_id, attrs in self._graph.nodes(data=True):
            coord = pos.get(node_id, (0.0, 0.0))
            node_x.append(coord[0])
            node_y.append(coord[1])
            node_colors.append(attrs.get("color", self.HEALTH_COLORS["stable"]))
            n_assign = attrs.get("n_assignments", 0)
            size_val = max(14.0, min(36.0, 12.0 + float(n_assign) * 2.0))
            node_sizes.append(size_val)
            node_labels.append(node_id[:8])
            self._node_id_registry.append(node_id)

            rmsd_disp = attrs.get("rmsd", 0.0)
            health_disp = attrs.get("health", "stable")
            ts_disp = attrs.get("timestamp", "")
            hover_descriptions.append(
                f"<b>Commit:</b> {node_id[:12]}...<br>"
                f"<b>RMSD:</b> {rmsd_disp:.2f} kHz<br>"
                f"<b>Assignments:</b> {n_assign}<br>"
                f"<b>Health:</b> {health_disp.upper()}<br>"
                f"<b>Created:</b> {ts_disp}"
            )

        # Build edge paths and mid-point hover markers
        edge_x: List[Optional[float]] = []
        edge_y: List[Optional[float]] = []
        edge_mid_x: List[float] = []
        edge_mid_y: List[float] = []
        edge_hover_labels: List[str] = []

        for u, v, edge_data in self._graph.edges(data=True):
            p0 = pos.get(u, (0.0, 0.0))
            p1 = pos.get(v, (0.0, 0.0))
            edge_x.extend([p0[0], p1[0], None])
            edge_y.extend([p0[1], p1[1], None])
            edge_mid_x.append((p0[0] + p1[0]) / 2.0)
            edge_mid_y.append((p0[1] + p1[1]) / 2.0)
            edge_hover_labels.append(edge_data.get("label", ""))

        try:
            fig = go.FigureWidget()
        except Exception:
            fig = go.Figure()

        # 1. Edge lines trace
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1.5, color="#888888"),
                hoverinfo="none",
                showlegend=False,
            )
        )

        # 2. Edge delta mid-point hover trace
        if edge_mid_x:
            fig.add_trace(
                go.Scatter(
                    x=edge_mid_x,
                    y=edge_mid_y,
                    mode="markers",
                    marker=dict(size=6, color="#888888", opacity=0.0),
                    hovertext=edge_hover_labels,
                    hoverinfo="text",
                    showlegend=False,
                )
            )

        # 3. Node markers trace
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=1.5, color="#2c3e50"),
                ),
                text=node_labels,
                hovertext=hover_descriptions,
                hoverinfo="text",
                textposition="top center",
                showlegend=False,
            )
        )

        # Layout styling
        fig.update_layout(
            title="🕰️ Provenance Graph — Decision Tree & Time-Travel",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
            hovermode="closest",
        )

        # Attach click event listener for time-travel state reversion
        if len(fig.data) > 0 and hasattr(fig.data[-1], "on_click"):
            fig.data[-1].on_click(self._handle_node_click)

        self._fig = fig
        return fig

    def _handle_node_click(self, trace: Any, points: Any, state: Any) -> None:
        """Handle user click on a graph node to dispatch state reversion."""
        if points and hasattr(points, "point_inds") and points.point_inds:
            idx = int(points.point_inds[0])
            if 0 <= idx < len(self._node_id_registry):
                target_state_id = self._node_id_registry[idx]
                logger.info("Time-travel reversion initiated to state: %s", target_state_id)
                self._on_revert(target_state_id)

    def revert_to_state(self, state_id: str) -> Dict[str, np.ndarray]:
        """Execute Time-Travel state reversion.

        1. Set CommitManager HEAD to target historical state_id.
        2. Restore high-precision tensors from HDF5 cache via TensorCache.
        3. Return restored tensor dictionary.

        Parameters
        ----------
        state_id : str
            SHA-256 hash of target historical FitState.

        Returns
        -------
        dict[str, np.ndarray]
            Restored tensors (covariance, hamiltonian, gp_weights, etc.).

        Raises
        ------
        KeyError
            If state_id does not exist in CommitManager or TensorCache.
        StateReversionError
            If commit manager or tensor cache interface is invalid.
        """
        if not hasattr(self._commit_mgr, "branch_from"):
            raise StateReversionError(
                "CommitManager does not implement required branch_from method."
            )

        self._commit_mgr.branch_from(state_id)

        if not hasattr(self._tensor_cache, "restore_state_tensors"):
            raise StateReversionError(
                "TensorCache does not implement required restore_state_tensors method."
            )

        restored_tensors = self._tensor_cache.restore_state_tensors(state_id)
        logger.info(
            "Reverted to state %s... (%d tensors restored)",
            state_id[:12],
            len(restored_tensors),
        )
        return restored_tensors

    def get_widget(self) -> Any:
        """Assemble and return the provenance graph as an ipywidgets container.

        Returns
        -------
        widgets.VBox or None
            Ipywidgets container with dashboard title and Plotly graph.
        """
        if not _HAS_WIDGETS:
            logger.warning("ipywidgets not available for get_widget.")
            return None

        fig = self.render_plotly_figure()
        if fig is None:
            return widgets.VBox(
                [widgets.HTML("<p>Provenance graph rendering unavailable.</p>")]
            )

        if isinstance(fig, widgets.Widget):
            return widgets.VBox(
                [
                    widgets.HTML("<h3>🕰️ Provenance Graph — Time-Travel</h3>"),
                    fig,
                ],
                layout=widgets.Layout(padding="10px"),
            )

        return widgets.VBox(
            [
                widgets.HTML("<h3>🕰️ Provenance Graph — Time-Travel</h3>"),
                widgets.HTML(
                    fig.to_html(include_plotlyjs="cdn", full_html=False)
                ),
            ],
            layout=widgets.Layout(padding="10px"),
        )
