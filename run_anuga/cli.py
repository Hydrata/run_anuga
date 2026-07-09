"""CLI entry point for run-anuga."""
import argparse
import os
import sys


def resolve_package_dir(path):
    """Accept either a directory or a path to scenario.json, return the directory."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        if os.path.basename(path) == "scenario.json":
            return os.path.dirname(path)
        raise argparse.ArgumentTypeError(
            f"Expected scenario.json or a directory containing it, got: {path}"
        )
    if os.path.isdir(path):
        return path
    raise argparse.ArgumentTypeError(f"Path does not exist: {path}")


def cmd_validate(args):
    """Validate a scenario package (core only, no heavy deps)."""
    from run_anuga.config import ScenarioConfig

    try:
        config = ScenarioConfig.from_package(args.package_dir)
        print(f"Valid scenario: {config.run_label}")
        print(f"  Duration: {config.duration}s, EPSG: {config.epsg}")
        if config.resolution:
            print(f"  Resolution: {config.resolution}m")
    except Exception as e:
        print(f"Invalid: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_info(args):
    """Show package summary (core only)."""
    from run_anuga.config import ScenarioConfig

    try:
        config = ScenarioConfig.from_package(args.package_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    inputs_dir = os.path.join(args.package_dir, "inputs")
    print(f"Package: {args.package_dir}")
    print(f"Label:   {config.run_label}")
    print(f"EPSG:    {config.epsg}")
    print(f"Duration: {config.duration}s")
    if os.path.isdir(inputs_dir):
        print("\nInputs:")
        for f in sorted(os.listdir(inputs_dir)):
            size = os.path.getsize(os.path.join(inputs_dir, f))
            print(f"  {f} ({size:,} bytes)")


def cmd_run(args):
    """Run an ANUGA flood simulation."""
    from run_anuga.run import run_sim
    from run_anuga.callbacks import LoggingCallback

    if args.username:
        run_sim(
            args.package_dir,
            args.username,
            args.password,
            args.batch_number,
            args.checkpoint_time,
        )
        return

    # TASK-1160 (F1b): leave the callback unset by default so run_sim's env-based
    # auto-construct kicks in — when HYDRATA_INTERNAL_COMPUTE_TOKEN is present
    # and scenario.json carries a control_server, a HydrataCallback streams
    # /log/ + /progress/ to the control server (matching bare ``python run.py``).
    # When neither is set, run_sim falls back to NullCallback (silent).
    # --log-to-stdout forces LoggingCallback regardless of the env, for
    # standalone debugging or when the token IS set but you want stdout output.
    callback = LoggingCallback() if args.log_to_stdout else None
    run_sim(
        args.package_dir,
        callback=callback,
        batch_number=args.batch_number,
        checkpoint_time=args.checkpoint_time,
    )


def cmd_run_and_report(args):
    """Run a simulation and hand the results back to the Hydrata control server.

    TASK-1159 (F1): one entry point for both Batch (via mpirun) and the F2
    localhost dispatcher. Reads scenario.json + env (HYDRATA_INTERNAL_COMPUTE_TOKEN,
    RESULT_S3_BUCKET), runs the sim, zips outputs, uploads to S3, POSTs
    /process-result/. POSTs /error/ on any failure.
    """
    from run_anuga._handoff import run_and_report

    result = run_and_report(args.package_dir, result_bucket=args.result_bucket)
    if result.get("result_key"):
        print(f"result_key: {result['result_key']}")
        print(f"process_result_status: {result['process_result_status']}")


def cmd_post_process(args):
    """Generate GeoTIFFs from SWW output."""
    from run_anuga.run_utils import post_process_sww

    post_process_sww(args.package_dir, output_raster_resolution=args.resolution)


def cmd_viz(args):
    """Generate video from result TIFFs."""
    from run_anuga.run_utils import make_video, make_comparison_video

    if args.compare:
        make_comparison_video(args.output_dir, args.compare, args.result_type)
    else:
        make_video(args.output_dir, args.result_type)


def cmd_upload(args):
    """Upload results to S3 STAC catalog."""
    from run_anuga.run_utils import generate_stac, _load_package_data

    input_data = _load_package_data(args.output_dir)
    generate_stac(
        output_directory=input_data["output_directory"],
        run_label=input_data["run_label"],
        output_quantities=["depth", "velocity", "depthIntegratedVelocity", "stage"],
        initial_time_iso_string=input_data["scenario_config"].get(
            "model_start", "1970-01-01T00:00:00+00:00"
        ),
        aws_access_key_id=args.aws_key or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=args.aws_secret
        or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        s3_bucket_name=args.bucket,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="run-anuga",
        description="ANUGA flood simulation toolkit",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run an ANUGA flood simulation")
    run_parser.add_argument(
        "package_dir", type=resolve_package_dir,
        help="Path to scenario.json or directory containing it",
    )
    run_parser.add_argument("--username", "-u", help="Hydrata username")
    run_parser.add_argument("--password", "-p", help="Hydrata password")
    run_parser.add_argument(
        "--batch-number", "-bn", type=int, default=1
    )
    run_parser.add_argument(
        "--checkpoint-time", "-ct", type=float, default=None
    )
    run_parser.add_argument(
        "--log-to-stdout",
        action="store_true",
        help="Force LoggingCallback (stdout-only) even when the token env is set; for standalone debugging.",
    )

    # --- run-and-report ---
    rar_parser = subparsers.add_parser(
        "run-and-report",
        help="Run a simulation, zip + upload results, and POST /process-result/ (TASK-1159 F1)",
    )
    rar_parser.add_argument(
        "package_dir", help="Path to scenario package directory"
    )
    rar_parser.add_argument(
        "--result-bucket",
        help="S3 bucket for the result zip (defaults to RESULT_S3_BUCKET env var)",
    )

    # --- validate ---
    val_parser = subparsers.add_parser(
        "validate", help="Validate a scenario package"
    )
    val_parser.add_argument(
        "package_dir", type=resolve_package_dir,
        help="Path to scenario.json or directory containing it",
    )

    # --- info ---
    info_parser = subparsers.add_parser("info", help="Show package summary")
    info_parser.add_argument(
        "package_dir", type=resolve_package_dir,
        help="Path to scenario.json or directory containing it",
    )

    # --- post-process ---
    pp_parser = subparsers.add_parser(
        "post-process", help="Generate GeoTIFFs from SWW"
    )
    pp_parser.add_argument(
        "package_dir", type=resolve_package_dir,
        help="Path to scenario.json or directory containing it",
    )
    pp_parser.add_argument(
        "--resolution", "-r", type=float, default=None
    )

    # --- viz ---
    viz_parser = subparsers.add_parser(
        "viz", help="Generate video from result TIFFs"
    )
    viz_parser.add_argument("output_dir", help="Path to outputs directory")
    viz_parser.add_argument(
        "result_type",
        choices=["depth", "velocity", "depthIntegratedVelocity", "stage"],
    )
    viz_parser.add_argument(
        "--compare", help="Second output dir for comparison video"
    )

    # --- upload ---
    upload_parser = subparsers.add_parser(
        "upload", help="Upload results to S3 STAC catalog"
    )
    upload_parser.add_argument(
        "output_dir", help="Path to scenario package directory"
    )
    upload_parser.add_argument(
        "--bucket", required=True, help="S3 bucket name"
    )
    upload_parser.add_argument(
        "--aws-key", help="AWS access key (or use AWS_ACCESS_KEY_ID env var)"
    )
    upload_parser.add_argument(
        "--aws-secret",
        help="AWS secret key (or use AWS_SECRET_ACCESS_KEY env var)",
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "run": cmd_run,
        "run-and-report": cmd_run_and_report,
        "validate": cmd_validate,
        "info": cmd_info,
        "post-process": cmd_post_process,
        "viz": cmd_viz,
        "upload": cmd_upload,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
