"""
CLI / Streamlit entry point for single-unit offline analysis.

Implementation lives in ``NPXL_offline_analysis.main_refactored``; this module
keeps a stable import path for ``Analysis.NPXL_analysis.single_unit_offline_analysis.main``.
"""
from __future__ import annotations

from collections.abc import Callable

from Analysis.NPXL_analysis.NPXL_offline_analysis.main_refactored import (
    main as _run_refactored_pipeline,
)


def main(
    parent_dir: str | None = None,
    *,
    progress_fn: Callable[[int, str], None] | None = None,
    status_fn: Callable[[str], None] | None = None,
) -> None:
    """
    Run the full offline analysis for one catgt (parent) folder.

    ``progress_fn`` / ``status_fn`` are optional UI hooks (e.g. Streamlit). The
    refactored pipeline does not emit fine-grained progress; we only bracket the run.
    """
    if status_fn is not None:
        status_fn("Running analysis pipeline…")
    if progress_fn is not None:
        progress_fn(10, "Running analysis pipeline…")

    _run_refactored_pipeline(parent_dir=parent_dir)

    if progress_fn is not None:
        progress_fn(90, "Analysis pipeline finished")
