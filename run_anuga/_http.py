"""HTTP helper for control-server callbacks.

Used by:

* ``run.py::_report_run_error`` (POST /api/v2/anuga/runs/<id>/error/)
* ``callbacks.py::HydrataCallback._post`` (POST /api/v2/anuga/runs/<id>/{log,progress}/)
* ``run_utils.py::update_web_interface`` (legacy V1 PATCH path, kept for
  backwards-compat with the 80-day-stale Batch image)

Callers pass an owned ``requests.Session`` via ``session=`` to realise
connection-reuse on hot paths (e.g. evolve loop 100+ POSTs per run).
When ``session`` is omitted a fresh Session is created and closed per call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from run_anuga._imports import import_optional

if TYPE_CHECKING:  # pragma: no cover — typing only
    import requests
    from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)


def make_internal_session(token: str) -> "requests.Session":
    """Return a Session with ``X-Internal-Token`` set to ``token`` (RAW, no Bearer prefix).

    ``IsInternalComputeCaller`` in ``apps/gn_anuga/permissions.py`` reads the
    header verbatim, so a ``Bearer`` prefix would 401. Caller owns close().
    """
    requests = import_optional("requests")
    session = requests.Session()
    session.headers['X-Internal-Token'] = token
    return session


def post_to_control_server(
    url: str,
    *,
    auth: "HTTPBasicAuth | None" = None,
    method: str = "POST",
    data: dict | None = None,
    files: dict | None = None,
    timeout: int | None = None,
    session: "requests.Session | None" = None,
) -> "requests.Response":
    """POST or PATCH to the control server.

    TASK-1049 (W1): a caller-owned ``session`` may be supplied so a long-lived
    Session (with pre-set headers like ``X-Internal-Token``) can be reused
    across many callback POSTs without rebuilding TCP/TLS. When ``session`` is
    provided the helper does NOT close it (caller owns lifecycle). When
    omitted, the helper falls back to a single-shot ``with requests.Session()``
    block, which closes the session on exit.

    Returns the response object. Logs an error (does not raise) on status >= 400.
    Callers are responsible for URL construction (templates vary across sites).

    Parameters
    ----------
    url
        Fully constructed URL (callers own templating).
    auth
        Optional ``HTTPBasicAuth`` instance. When provided alongside an owned
        Session, it is set on the session (overwriting any prior auth) for the
        legacy BasicAuth callers. When a caller-supplied ``session`` is used
        with its own pre-set auth/headers, leave ``auth=None``.
    method
        ``"POST"`` or ``"PATCH"``.  Case-insensitive.
    data
        Form payload (passed verbatim to ``requests``).
    files
        File payload (passed verbatim to ``requests``).
    timeout
        Per-request timeout in seconds. Defaults to ``None`` (no timeout) to
        preserve the pre-refactor behavior, which is important for the
        PATCH-with-``files`` callers (mesh/result artifact uploads from a
        worker on a slow link can easily exceed any short bound). Callers
        that want a timeout should pass one explicitly.
    session
        Optional ``requests.Session`` owned by the caller. When provided, the
        helper uses it directly and does NOT close it (caller's responsibility,
        typically via a ``close()`` method paired with a ``try/finally`` block
        at the run-loop site).

    Returns
    -------
    requests.Response
        The response object.  Caller decides what to do with it.

    Raises
    ------
    ValueError
        If ``method`` is anything other than ``"POST"`` or ``"PATCH"``
        (case-insensitive).
    """
    requests = import_optional("requests")

    def _do_request(client) -> "requests.Response":
        verb = method.lower()
        if verb == "post":
            return client.post(url, data=data, files=files, timeout=timeout)
        elif verb == "patch":
            return client.patch(url, data=data, files=files, timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method!r}. Use 'POST' or 'PATCH'.")

    if session is not None:
        # Caller-owned Session: do NOT close. Only set auth if the caller
        # explicitly passed one (callers using header-based auth like
        # X-Internal-Token leave auth=None and pre-set session.headers).
        if auth is not None:
            session.auth = auth
        response = _do_request(session)
    else:
        with requests.Session() as client:
            if auth is not None:
                client.auth = auth
            response = _do_request(client)

    if response.status_code >= 400:
        logger.error(
            "Error posting to control server %s. HTTP code: %d - %s",
            url,
            response.status_code,
            response.text,
        )
    return response
