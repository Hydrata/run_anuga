"""TASK-4351: the scenario's ANUGA flow algorithm (the solver).

scenario.json ``flow_algorithm`` picks the slope-limiting and time-stepping pair ANUGA
runs. Absent means ``'DE0'``, anuga_core's own default (Euler, CFL 0.9), so every
package built before the key behaves exactly as before. There is NO environment
override: scenario.json is the only source, so the solver a run records is the solver
it ran.
"""

#: The scenario.json key.
FLOW_ALGORITHM_KEY = "flow_algorithm"
#: anuga_core's default, used when the key is absent or null.
DEFAULT_FLOW_ALGORITHM = "DE0"
#: anuga_core's own six values (shallow_water_domain.py set_flow_algorithm, pin 3712ea6f).
#: hydrata offers only DE0, DE1 and DE_ader2; the engine accepts them all.
FLOW_ALGORITHMS = ("DE0", "DE1", "DE2", "DE0_7", "DE1_7", "DE_ader2")


def resolve_flow_algorithm(input_data) -> str:
    """The run's flow algorithm: ``scenario_config['flow_algorithm']``, else ``'DE0'``.

    ``input_data`` is run_sim's input dict. An unknown value raises ``ValueError`` naming
    the allowed values: falling back to DE0 would mislabel a comparison run.
    """
    scenario_config = (input_data or {}).get("scenario_config") or {}
    value = scenario_config.get(FLOW_ALGORITHM_KEY)
    if value is None:
        return DEFAULT_FLOW_ALGORITHM
    if not isinstance(value, str) or value not in FLOW_ALGORITHMS:
        raise ValueError(
            f"scenario {FLOW_ALGORITHM_KEY!r} is {value!r}; use one of "
            f"{', '.join(FLOW_ALGORITHMS)} (DE0 is the default)."
        )
    return value
