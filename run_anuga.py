import json
import traceback
import sys
import math

import os
import requests
import pandas as pd
from pathlib import Path
from osgeo import gdal

import anuga
import anuga.utilities.quantity_setting_functions as qs
import anuga.utilities.plot_utils as util
from anuga.operators.rate_operators import Polygonal_rate_operator
from anuga import distribute, finalize, barrier

import logging


def run(username=None, password=None):
    # mpirun -np 8 /opt/venv/hydrata/bin/python /opt/hydrata/gn_anuga/run_code/run_anuga.py "test"
    logging.basicConfig(filename=f'../anuga_tmp.log', level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(f'os.getcwd() {os.getcwd()}')
    scenario_config = json.load(open('../scenario.json'))
    scenario_id = scenario_config.get('id')
    run_id = scenario_config.get('run_id', 0)
    project_id = scenario_config.get('project')
    duration = scenario_config.get('duration')
    maximum_triangle_area = scenario_config.get('maximum_triangle_area')
    constant_rainfall = scenario_config.get('constant_rainfall')
    boundary = json.load(open(f'../inputs/{scenario_config.get("boundary")}'))
    elevation_filename = f'../inputs/{scenario_config.get("elevation")}'
    run_label = f"{project_id}_{scenario_id}_{run_id}"
    output_directory = f'../outputs_{project_id}_{scenario_id}_{run_id}'
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"using run_label: {run_label}")
    logger.info(f"__version__: {anuga.__version__}")
    logger.info(f"__git_sha__: {anuga.__git_sha__}")
    logger.info(f"elevation_filename: {elevation_filename}")
    logger.info(f"boundary: {boundary}")

    client = None
    if anuga.myid == 0:
        if username and password:
            client = requests.Session()
            client.auth = requests.auth.HTTPBasicAuth(username, password)
            response_2 = client.post(
                f'https://hydrata.com/anuga/api/{project_id}/{scenario_id}/run/{run_id}',
                data={
                    "log": "Starting run...",
                    "project": project_id,
                    "scenario": scenario_id,
                    "status": "running"
                }
            )

    # setup rainfall
    def rain(time_in_seconds):
        t_sec = int(math.floor(time_in_seconds))
        return rain_df['rate_m_s'][t_sec]
    date_rng = pd.date_range(start='1/1/2022', periods=duration + 1, freq='s')
    rain_df = pd.DataFrame(date_rng, columns=['datetime'])
    rain_df['rate_m_s'] = constant_rainfall/1000
    logger.info("rainfall")
    logger.info(rain_df.head(10))

    logger.info("Logging is configured.")
    output_stats = dict()
    logger.info(f'run started on processor {anuga.myid}')
    mesh_filepath = f'{output_directory}/run_{scenario_id}_{run_id}.msh'
    bounding_polygon = boundary.get("features")[0].get("geometry").get("coordinates")
    logger.info('bounding_polygon:')
    logger.info(bounding_polygon)
    try:
        if anuga.myid == 0:
            if username and password:
                response_3 = client.patch(f'https://hydrata.com/anuga/api/{project_id}/{scenario_id}/run/{run_id}', data={
                    "status": "creating mesh"
                })
                print(response_3)
            anuga.pmesh.mesh_interface.create_mesh_from_regions(
                bounding_polygon=bounding_polygon,
                boundary_tags={'exterior': range(len(bounding_polygon))},
                maximum_triangle_area=maximum_triangle_area,
                interior_regions=[],
                interior_holes=[],
                filename=mesh_filepath,
                use_cache=False,
                verbose=True
            )
            domain = anuga.shallow_water.shallow_water_domain.Domain(
                mesh_filepath,
                use_cache=False,
                verbose=True,
            )
            domain.set_name(run_label)
            domain.set_datadir(output_directory)
            domain.set_quantity('friction', 0.06, verbose=True)
            domain.set_quantity('stage', 0.0, verbose=True)
            poly_fun_pairs = [['Extent', elevation_filename]]
            elevation_function = qs.composite_quantity_setting_function(
                poly_fun_pairs,
                domain,
                nan_treatment='exception',
            )
            domain.set_quantity('elevation', elevation_function, verbose=True, alpha=0.99)
            domain.set_minimum_storable_height(0.005)
        else:
            domain = None
        if anuga.myid == 0 and username and password:
            response_4 = client.patch(f'https://hydrata.com/anuga/api/{project_id}/{scenario_id}/run/{run_id}', data={
                "status": f"created mesh"
            })
            print(response_4)
        barrier()
        domain = distribute(domain, verbose=True)
        reflective_boundary = anuga.Reflective_boundary(domain)
        transmissive_boundary = anuga.Transmissive_boundary(domain)
        dirichlet_boundary = anuga.Dirichlet_boundary([0, 0, 0])

        domain.set_boundary(
            {
                'exterior': dirichlet_boundary,
                'interior': reflective_boundary
            }
        )
        Polygonal_rate_operator(domain, rate=rain, factor=1.0e-3, polygon=bounding_polygon, default_rate=0.00)
        yieldstep = 60
        logger.info(f'{yieldstep=}')
        logger.info(f'{duration=}')
        for t in domain.evolve(yieldstep=yieldstep, finaltime=duration):
            domain.write_time()
            logger.info(f'domain.evolve {t} on processor {anuga.myid}')
            if anuga.myid == 0:
                if username and password:
                    response_5 = client.patch(f'https://hydrata.com/anuga/api/{project_id}/{scenario_id}/run/{run_id}', data={
                        "status": f"{round(t/duration * 100, 0)}%"
                    })
                    logger.info(f"{round(t/duration * 100, 0)}%")
                    logger.info(response_5)
                logger.info(domain.timestepping_statistics())
        domain.sww_merge(verbose=True, delete_old=True)
        barrier()

        if anuga.myid == 0:
            logger.info('Generating output rasters...')
            raster = gdal.Open(elevation_filename)
            gt = raster.GetGeoTransform()
            resolution = math.floor(gt[1] / 4)
            if resolution == 0:
                resolution = 1
            epsg_integer = int(scenario_config.get("epsg").split(":")[1] if ":" in scenario_config.get("epsg") else scenario_config.get("epsg"))
            logger.info(f'epsg_integer: {epsg_integer}')
            util.Make_Geotif(
                swwFile=f"{output_directory}/{run_label}.sww",
                output_quantities=['depth', 'velocity', 'depthIntegratedVelocity'],
                myTimeStep='max',
                CellSize=resolution,
                lower_left=None,
                upper_right=None,
                EPSG_CODE=epsg_integer,
                proj4string=None,
                velocity_extrapolation=True,
                min_allowed_height=1.0e-05,
                output_dir=output_directory,
                bounding_polygon=bounding_polygon,
                internal_holes=None,
                verbose=False,
                k_nearest_neighbours=3,
                creation_options=[]
            )
            logger.info('Successfully generated depth, velocity, momentum outputs')
            if anuga.myid == 0 and username and password:
                url = f'https://hydrata.com/anuga/api/{project_id}/{scenario_id}/run/{run_id}/'
                response_5 = client.patch(
                    url=url,
                    data={
                        "status": "uploading results",
                    },
                    files={
                        "tif_depth_max": open(f'{output_directory}/{run_label}_depth_max.tif', 'rb'),
                        "tif_depth_integrated_velocity_max": open(f'{output_directory}/{run_label}_depthIntegratedVelocity_max.tif', 'rb'),
                        "tif_velocity_max": open(f'{output_directory}/{run_label}_velocity_max.tif', 'rb'),
                    }
                )
            logger.info('Successfully uploaded outputs')
    except Exception as e:
        logger.error(traceback.format_exc())
    finally:
        barrier()
        finalize()
        logger.info(f'upload log here')
    return f'finished: {run_label}'


if __name__ == '__main__':
    username = None
    password = None
    if len(sys.argv) > 1:
        username = sys.argv[1]
        password = sys.argv[2]
    run(username, password)
