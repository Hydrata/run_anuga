"""Tests for the rank-0 MPI_Abort tear-down in run_sim's broad except handler.

When rank-0-only work (e.g. post_process_sww) raises an exception inside the
main try block, the other ranks are blocked at the next barrier(). Without
MPI_Abort, rank 0 unwinds through finally -> _finalize_with_timeout (~30s)
while ranks 1-3 keep CPUs pinned. MPI_Abort(1) tears down COMM_WORLD so the
container exits fast and the entrypoint EXIT trap can report /error/.

Regression for canary-15 wedge (TASK-1048 W6.4 / TASK-1080).
"""

import sys
from unittest import mock

import pytest


def _install_fake_mpi():
    """Return (fake_mpi_module, fake_comm) and inject into sys.modules.

    The mock replaces both `mpi4py` and `mpi4py.MPI` so that the lazy
    `from mpi4py import MPI` inside run.py's except handler resolves to our
    mock and we can assert on COMM_WORLD.Abort.
    """
    fake_comm = mock.MagicMock(name="COMM_WORLD")
    fake_mpi_submodule = mock.MagicMock(name="mpi4py.MPI")
    fake_mpi_submodule.COMM_WORLD = fake_comm
    fake_mpi_pkg = mock.MagicMock(name="mpi4py")
    fake_mpi_pkg.MPI = fake_mpi_submodule
    sys.modules["mpi4py"] = fake_mpi_pkg
    sys.modules["mpi4py.MPI"] = fake_mpi_submodule
    return fake_mpi_pkg, fake_comm


@pytest.fixture
def fake_mpi(monkeypatch):
    """Inject a mock mpi4py so we can observe COMM_WORLD.Abort without a real
    MPI tear-down. Restores any pre-existing mpi4py modules on teardown."""
    saved = {k: sys.modules.get(k) for k in ("mpi4py", "mpi4py.MPI")}
    fake_pkg, fake_comm = _install_fake_mpi()
    try:
        yield fake_pkg, fake_comm
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _stub_setup_input_data_for_inside_try_failure(monkeypatch):
    """Make setup_input_data return a dict that raises KeyError on the first
    access inside the try block (`input_data['run_label']` at run.py:76).

    This forces an exception to be raised INSIDE the try -> reach the broad
    except at line 275 -> trigger MPI_Abort path. setup_input_data itself
    runs before the try, so we can't just have it raise.
    """
    class _BoomDict(dict):
        def __getitem__(self, key):
            if key == "run_label":
                raise RuntimeError("synthetic-rank0-boom")
            return super().__getitem__(key)

    payload = _BoomDict()
    payload["scenario_config"] = {}
    payload["checkpoint_directory"] = "/tmp/nonexistent"

    monkeypatch.setattr("run_anuga.run.setup_input_data", lambda *a, **kw: payload)


def _stub_logger_and_callback(monkeypatch):
    """Bypass setup_logger (writes HTTP) + _finalize_with_timeout (would tear
    down the real MPI state initialised by anuga import in other test files,
    causing finalize() to hang or fault across the suite). callback is
    supplied via run_sim's `callback=` param."""
    monkeypatch.setattr("run_anuga.run.setup_logger", lambda *a, **kw: mock.MagicMock())
    monkeypatch.setattr("run_anuga.run._finalize_with_timeout", lambda *a, **kw: None)


class TestRank0AbortOnException:
    def test_rank0_exception_calls_mpi_abort(self, monkeypatch, tmp_path, fake_mpi):
        """When an exception fires inside run_sim's try block,
        MPI.COMM_WORLD.Abort(1) must be called before the exception propagates,
        so other ranks tear down quickly instead of spinning at barriers."""
        from run_anuga.callbacks import NullCallback
        from run_anuga.run import run_sim

        _stub_setup_input_data_for_inside_try_failure(monkeypatch)
        _stub_logger_and_callback(monkeypatch)

        _, fake_comm = fake_mpi

        with pytest.raises(RuntimeError, match="synthetic-rank0-boom"):
            run_sim(str(tmp_path), callback=NullCallback())

        fake_comm.Abort.assert_called_once_with(1)

    def test_rank0_exception_falls_through_when_abort_raises(self, monkeypatch, tmp_path, fake_mpi):
        """If MPI.COMM_WORLD.Abort itself fails, the original exception must
        still propagate. The inner try/except in run.py guarantees Abort
        failure cannot mask the actual run failure."""
        from run_anuga.callbacks import NullCallback
        from run_anuga.run import run_sim

        _stub_setup_input_data_for_inside_try_failure(monkeypatch)
        _stub_logger_and_callback(monkeypatch)

        _, fake_comm = fake_mpi
        fake_comm.Abort.side_effect = RuntimeError("abort-also-broken")

        with pytest.raises(RuntimeError, match="synthetic-rank0-boom"):
            run_sim(str(tmp_path), callback=NullCallback())

        fake_comm.Abort.assert_called_once_with(1)
