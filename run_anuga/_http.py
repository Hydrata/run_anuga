"""HTTP helper for control-server callbacks.

Three sites used to duplicate the requests.Session + HTTPBasicAuth +
status-check + error-log pattern:

* ``run.py::_report_run_error`` (POST /api/v2/anuga/runs/<id>/error/)
* ``run_utils.py::update_web_interface`` (PATCH anuga/api/<p>/<s>/run/<id>/)
* ``callbacks.py::HydrataCallback._patch`` (PATCH anuga/api/<p>/<s>/run/<id>/)

This module collapses them onto a single helper.  Callers still own URL
construction since the templates vary across sites.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from run_anuga._imports import import_optional

if TYPE_CHECKING:  # pragma: no cover — typing only
    import requests
    from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)


def post_to_control_server(
    url: str,
    *,
    auth: "HTTPBasicAuth",
    method: str = "POST",
    data: dict | None = None,
    files: dict | None = None,
    timeout: int | None = None,
) -> "requests.Response":
    """POST or PATCH to the control server using a requests.Session with pre-set BasicAuth.

    Returns the response object. Logs an error (does not raise) on status >= 400.
    Callers are responsible for URL construction (templates vary across sites).

    Parameters
    ----------
    url
        Fully constructed URL (callers own templating).
    auth
        ``HTTPBasicAuth`` instance, set on the session, not per-request.
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

    with requests.Session() as client:
        client.auth = auth
        verb = method.lower()
        if verb == "post":
            response = client.post(url, data=data, files=files, timeout=timeout)
        elif verb == "patch":
            response = client.patch(url, data=data, files=files, timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method!r}. Use 'POST' or 'PATCH'.")

    if response.status_code >= 400:
        logger.error(
            "Error posting to control server %s. HTTP code: %d - %s",
            url,
            response.status_code,
            response.text,
        )
    return response
