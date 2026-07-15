"""Component tests for raster processing functions.

Requires geopandas (marked requires_geo).

Note: the ``burn_structures_into_raster`` and ``_clip_and_resample`` tests that
formerly lived here were removed during the run_anuga single-main unification
(TASK-2149). The universal DEM burn was retired on the cloud pipeline
(ADR-4 / TASK-1270 — structures now route to Reflective holes / Raised
elevation / Mannings friction) and ``_clip_and_resample`` no longer exists.
Current structure/mesh coverage lives in test_reflective_mesh.py and
test_mesh_qa.py.
"""

import os

import pytest


@pytest.mark.requires_geo
class TestMakeShpFromPolygon:
    def test_creates_shapefile(self, tmp_path):
        from run_anuga.run_utils import make_shp_from_polygon

        gpd = pytest.importorskip("geopandas")

        output_path = str(tmp_path / "boundary.shp")
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]
        make_shp_from_polygon(polygon, 28355, output_path)

        assert os.path.isfile(output_path)
        gdf = gpd.read_file(output_path)
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 28355
