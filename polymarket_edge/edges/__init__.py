"""Edge discovery modules."""
from polymarket_edge.edges.consistency_arb import run as run_consistency_arb
from polymarket_edge.edges.microstructure_mm import run as run_microstructure_mm
from polymarket_edge.edges.resolution_mechanics import run as run_resolution_mechanics

__all__ = [
    "run_consistency_arb",
    "run_microstructure_mm",
    "run_resolution_mechanics",
]
