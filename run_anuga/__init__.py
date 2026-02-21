"""
run_anuga — Run ANUGA flood simulations from Hydrata scenario packages.

Modules:
    run         Main simulation driver (run_sim, main)
    run_utils   Input parsing, mesh generation, post-processing
    defaults    Simulation constants (Manning's n, yieldstep limits, etc.)
    config      Pydantic ScenarioConfig model for scenario.json
    callbacks   SimulationCallback protocol and implementations
    cli         CLI entry point with subcommands
    _imports    Lazy import helper for optional dependencies
"""

__version__ = "1.1.0"
