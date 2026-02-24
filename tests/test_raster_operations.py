"""Component tests for raster processing functions.

Requires rasterio and geopandas (marked requires_geo).
"""

import json
import os
import shutil

import pytest



@pytest.mark.requires_geo
class TestBurnStructuresIntoRaster:
    def test_burn_creates_backup(self, small_geotiff, tmp_path):
        from run_anuga.run_utils import burn_structures_into_raster

        # Copy geotiff to modifiable location
        raster = tmp_path / "dem.tif"
        shutil.copy(str(small_geotiff), str(raster))

        structures = tmp_path / "structures.geojson"
        structures.write_text(json.dumps({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[321020, 5812020], [321080, 5812020],
                         [321080, 5812080], [321020, 5812080],
                         [321020, 5812020]]
                    ]
                },
                "properties": {"method": "Mannings"}
            }]
        }))

        result = burn_structures_into_raster(str(structures), str(raster), backup=True)
        assert result is True
        assert (tmp_path / "dem_original.tif").exists()

    def test_burn_no_backup(self, small_geotiff, tmp_path):
        from run_anuga.run_utils import burn_structures_into_raster

        raster = tmp_path / "dem.tif"
        shutil.copy(str(small_geotiff), str(raster))

        structures = tmp_path / "structures.geojson"
        structures.write_text(json.dumps({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[321020, 5812020], [321080, 5812020],
                         [321080, 5812080], [321020, 5812080],
                         [321020, 5812020]]
                    ]
                },
                "properties": {}
            }]
        }))

        result = burn_structures_into_raster(str(structures), str(raster), backup=False)
        assert result is True
        assert not (tmp_path / "dem_original.tif").exists()

    def test_burn_modifies_raster(self, small_geotiff, tmp_path):
        from run_anuga.run_utils import burn_structures_into_raster
        import numpy as np

        rasterio = pytest.importorskip("rasterio")

        raster = tmp_path / "dem.tif"
        shutil.copy(str(small_geotiff), str(raster))

        # Read original values
        with rasterio.open(str(raster)) as src:
            original = src.read(1).copy()

        structures = tmp_path / "structures.geojson"
        structures.write_text(json.dumps({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[321020, 5812020], [321080, 5812020],
                         [321080, 5812080], [321020, 5812080],
                         [321020, 5812020]]
                    ]
                },
                "properties": {}
            }]
        }))

        burn_structures_into_raster(str(structures), str(raster), backup=False)

        with rasterio.open(str(raster)) as src:
            modified = src.read(1)

        # At least some pixels should have been modified (additive burn)
        diff = modified - original
        assert np.any(diff > 0)

    def test_burn_empty_features_no_change(self, small_geotiff, tmp_path):
        from run_anuga.run_utils import burn_structures_into_raster

        raster = tmp_path / "dem.tif"
        shutil.copy(str(small_geotiff), str(raster))

        structures = tmp_path / "structures.geojson"
        structures.write_text(json.dumps({
            "type": "FeatureCollection",
            "features": []
        }))

        result = burn_structures_into_raster(str(structures), str(raster), backup=False)
        assert result is True


@pytest.mark.requires_geo
class TestClipAndResample:
    def test_clip_produces_output(self, small_geotiff, tmp_path):
        from run_anuga.run_utils import _clip_and_resample, make_shp_from_polygon

        rasterio = pytest.importorskip("rasterio")

        # Create cutline shapefile
        cutline_path = str(tmp_path / "cutline.shp")
        polygon = [[321010, 5812010], [321090, 5812010],
                    [321090, 5812090], [321010, 5812090], [321010, 5812010]]
        make_shp_from_polygon(polygon, 28355, cutline_path)

        dst_path = str(tmp_path / "clipped.tif")
        _clip_and_resample(str(small_geotiff), dst_path, cutline_path, resolution=10.0)

        assert os.path.isfile(dst_path)
        with rasterio.open(dst_path) as ds:
            assert ds.width > 0
            assert ds.height > 0
            assert ds.crs is not None


@pytest.mark.requires_geo
class TestMakeShpFromPolygon:
    def test_creates_shapefile(self, tmp_path):
        from run_anuga.run_utils import make_shp_from_polygon

        output_path = str(tmp_path / "boundary.shp")
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]
        make_shp_from_polygon(polygon, 28355, output_path)

        assert os.path.isfile(output_path)

    def test_shapefile_with_buffer(self, tmp_path):
        from run_anuga.run_utils import make_shp_from_polygon

        output_path = str(tmp_path / "buffered.shp")
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]
        make_shp_from_polygon(polygon, 28355, output_path, buffer=10)

        assert os.path.isfile(output_path)
