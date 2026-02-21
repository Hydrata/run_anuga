"""
run_anuga — Run ANUGA flood simulations from Hydrata scenario packages.

Modules:
    run         Main simulation driver (run_sim, main)
    run_utils   Input parsing, mesh generation, post-processing
    defaults    Simulation constants (Manning's n, yieldstep limits, etc.)
    schema      JSON Schema validation for scenario.json
"""

__version__ = "1.0.0"
