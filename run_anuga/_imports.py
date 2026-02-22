"""
Lazy import helper for optional dependencies.

Provides clear error messages when heavy dependencies (GDAL, anuga, etc.)
are not installed, guiding users to the correct pip extra.
"""

from __future__ import annotations

_EXTRA_MAP = {
    "anuga": "full",
    "numpy": "sim",
    "pandas": "sim",
    "dill": "sim",
    "psutil": "sim",
    "shapely": "sim",
    "osgeo": "sim",
    "rasterio": "sim",
    "cv2": "viz",
    "matplotlib": "viz",
    "requests": "platform",
    "boto3": "platform",
    "pystac": "platform",
    "celery": "platform",
    "django": "platform",
}


def import_optional(module_name: str, *, extra: str | None = None):
    """
    Import and return a module, raising a helpful ImportError if missing.

    Parameters
    ----------
    module_name : str
        Dotted module path, e.g. ``"osgeo.ogr"`` or ``"anuga"``.
    extra : str or None
        pip extra name (e.g. ``"sim"``).  If *None*, looked up from ``_EXTRA_MAP``
        using the top-level package name.

    Returns
    -------
    module
        The imported module object.

    Raises
    ------
    ImportError
        With a message telling the user which pip extra to install.
    """
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        top_level = module_name.split(".")[0]
        # Check if the top-level package is installed but fails to import
        # (e.g. anuga is installed but missing transitive deps like matplotlib)
        from importlib.metadata import PackageNotFoundError, distribution
        try:
            distribution(top_level)
        except PackageNotFoundError:
            pass  # Not installed — fall through to helpful message
        else:
            # Package IS installed but import failed — show the real error
            raise ImportError(
                f"'{module_name}' is installed but failed to import: {exc}. "
                f"This is likely a missing transitive dependency of '{top_level}'."
            ) from exc
        extra = extra or _EXTRA_MAP.get(top_level, "full")
        raise ImportError(
            f"'{module_name}' is required for this operation but not installed. "
            f'Install it with: pip install "run_anuga[{extra}]"'
        ) from None
