"""
CoChem-SpycFit Interfaces Module.

Exports:
- cochem_spycfit_ui: Decision Dashboard (HITL), Plotly WebGL dual-pane residuals, bipartite Hungarian auto-propose, offline PDF/HTML rendering, What-If parameter sliders, semantic commits.
- cochem_vibspyc_snap: Publication-ready snapshot export, AASTeX/LaTeX longtable rendering, CrossRef BibTeX citations, dataset hashing, and artifact packaging.
"""

from .cochem_spycfit_ui import (
    AssignmentStatus,
    BipartiteConfig,
    BipartiteMatchingError,
    DashboardConfig,
    HITLApprovalError,
    HITLAssignmentManager,
    OfflineRenderError,
    PeakAssignment,
    RenderBackend,
    SemanticCommit,
    SemanticCommitError,
    SliderComputationError,
    SpectralPeak,
    UIDashboardError,
    WhatIfParameterState,
    WhatIfSpectrumEngine,
    auto_propose_assignments,
    compute_bipartite_cost_matrix,
    compute_huber_loss_excluding_ghosts,
    render_dual_pane_dashboard,
    render_static_offline_html,
    render_static_offline_pdf,
    translate_mathematical_change_to_git_commit,
)
from .cochem_vibspyc_snap import (
    FitProvenancePayload,
    evaluate_compression_strategy,
    export_spycfit_snapshot,
    format_citations_to_bib,
    generate_aastex_longtables,
    get_required_dois,
    get_spycfit_processed_dir,
    hash_dataset_iteratively,
    package_fit_artifacts,
    resolve_processed_workspace_dir,
    seal_artifact_read_only,
)

__all__ = [
    # UI Dashboard
    "AssignmentStatus",
    "BipartiteConfig",
    "BipartiteMatchingError",
    "DashboardConfig",
    "HITLApprovalError",
    "HITLAssignmentManager",
    "OfflineRenderError",
    "PeakAssignment",
    "RenderBackend",
    "SemanticCommit",
    "SemanticCommitError",
    "SliderComputationError",
    "SpectralPeak",
    "UIDashboardError",
    "WhatIfParameterState",
    "WhatIfSpectrumEngine",
    "auto_propose_assignments",
    "compute_bipartite_cost_matrix",
    "compute_huber_loss_excluding_ghosts",
    "render_dual_pane_dashboard",
    "render_static_offline_html",
    "render_static_offline_pdf",
    "translate_mathematical_change_to_git_commit",
    # Snapshot & Export
    "FitProvenancePayload",
    "evaluate_compression_strategy",
    "export_spycfit_snapshot",
    "format_citations_to_bib",
    "generate_aastex_longtables",
    "get_required_dois",
    "get_spycfit_processed_dir",
    "hash_dataset_iteratively",
    "package_fit_artifacts",
    "resolve_processed_workspace_dir",
    "seal_artifact_read_only",
]
