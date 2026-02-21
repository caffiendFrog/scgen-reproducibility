import argparse
import datetime
import json
import os
import subprocess
import sys
import traceback

from scgen.gpu_utils import get_available_gpu_ids, run_command_specs_parallel


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
RECON_SCGEN_DIR = os.path.join(DATA_DIR, "reconstructed", "scGen")
RUN_MANIFEST_DIR = os.path.join(DATA_DIR, "run_manifests")


def _build_env(overwrite=False):
    env = os.environ.copy()
    if overwrite:
        env["SCGEN_OVERWRITE"] = "1"
    else:
        env.pop("SCGEN_OVERWRITE", None)
    seed = env.get("SCGEN_SEED", "4039")
    env.setdefault("SCGEN_PROCESS_SEED", seed)
    env.setdefault("PYTHONHASHSEED", seed)
    env.setdefault("NUMPY_SEED", seed)
    env.setdefault("TF_SEED", seed)
    if env.get("SCGEN_ENABLE_DETERMINISM", "1") != "0":
        env.setdefault("TF_DETERMINISTIC_OPS", "1")
        env.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    return env


def run_command(command, overwrite=False):
    env = _build_env(overwrite=overwrite)
    return subprocess.call(command, shell=True, cwd=SCRIPT_DIR, env=env)


def missing_paths(paths):
    return [path for path in paths if not os.path.exists(path)]


def ensure_inputs_exist():
    required_inputs = [
        os.path.join(DATA_DIR, "train_pbmc.h5ad"),
        os.path.join(DATA_DIR, "valid_pbmc.h5ad"),
        os.path.join(DATA_DIR, "train_hpoly.h5ad"),
        os.path.join(DATA_DIR, "valid_hpoly.h5ad"),
        os.path.join(DATA_DIR, "train_salmonella.h5ad"),
        os.path.join(DATA_DIR, "valid_salmonella.h5ad"),
        os.path.join(DATA_DIR, "train_species.h5ad"),
        os.path.join(DATA_DIR, "valid_species.h5ad"),
        os.path.join(DATA_DIR, "train_study.h5ad"),
        os.path.join(DATA_DIR, "valid_study.h5ad"),
        os.path.join(DATA_DIR, "pancreas.h5ad"),
        os.path.join(DATA_DIR, "MouseAtlas.subset.h5ad"),
    ]
    missing = missing_paths(required_inputs)
    if missing:
        print("Missing input datasets. Running DataDownloader to fetch them...")
        run_command("python ./DataDownloader.py")
        missing_after = missing_paths(required_inputs)
        if missing_after:
            missing_list = "\n".join(f"- {path}" for path in missing_after)
            raise FileNotFoundError(
                "Required datasets are still missing after DataDownloader. "
                "Ensure the data exists in the data directory or provide it manually:\n"
                f"{missing_list}"
            )


def ensure_batch_correction_outputs(overwrite=False):
    required_outputs = [
        os.path.join(RECON_SCGEN_DIR, "pancreas.h5ad"),
        os.path.join(RECON_SCGEN_DIR, "mouse_atlas.h5ad"),
    ]
    missing = missing_paths(required_outputs)
    if not missing:
        return
    if os.path.join(RECON_SCGEN_DIR, "pancreas.h5ad") in missing:
        print("Generating scGen batch-corrected pancreas dataset...")
        run_command("python ./pancreas.py", overwrite=overwrite)
    if os.path.join(RECON_SCGEN_DIR, "mouse_atlas.h5ad") in missing:
        print("Generating scGen batch-corrected mouse atlas dataset...")
        run_command("python ./mouse_atlas.py")
    missing_after = missing_paths(required_outputs)
    if missing_after:
        missing_list = "\n".join(f"- {path}" for path in missing_after)
        raise FileNotFoundError(
            "Batch-corrected datasets are still missing after generation:\n"
            f"{missing_list}"
        )


def _safe_git_output(args):
    try:
        return subprocess.check_output(args, cwd=SCRIPT_DIR, text=True).strip()
    except Exception:
        return "unknown"


def _seed_manifest():
    seed_keys = [
        "PYTHONHASHSEED",
        "SCGEN_SEED",
        "SCGEN_PROCESS_SEED",
        "SCGEN_ENABLE_DETERMINISM",
        "NUMPY_SEED",
        "TF_SEED",
        "TF_DETERMINISTIC_OPS",
        "TF_CUDNN_DETERMINISTIC",
    ]
    return {key: os.environ.get(key) for key in seed_keys}


def _write_manifest(manifest_path, manifest):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _start_run_manifest(model_to_train, overwrite, stage_one_specs, stage_two_specs):
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    manifest_name = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{model_to_train}.json"
    manifest_path = os.path.join(RUN_MANIFEST_DIR, manifest_name)
    manifest = {
        "status": "running",
        "model_to_train": model_to_train,
        "overwrite": overwrite,
        "start_time_utc": timestamp.isoformat(),
        "git": {
            "commit": _safe_git_output(["git", "rev-parse", "HEAD"]),
            "branch": _safe_git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        },
        "env": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": sys.platform,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
        },
        "available_gpu_ids_at_start": get_available_gpu_ids(),
        "seeds": _seed_manifest(),
        "command_plan": {
            "stage_one": stage_one_specs,
            "stage_two": stage_two_specs,
            "post_stage": ["ensure_batch_correction_outputs"],
        },
        "stages": {},
    }
    _write_manifest(manifest_path, manifest)
    return manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description="Train scGen reproducibility models.")
    parser.add_argument(
        "model",
        nargs="?",
        default="all",
        help="Model to train: all, PCA, VecArithm, STGAN, CVAE, scGen",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate reconstructed outputs even if they already exist.",
    )
    args = parser.parse_args()
    model_to_train = args.model
    overwrite = args.overwrite
    if model_to_train == "all":
        ensure_inputs_exist()
        stage_one_specs = [
            {"command": "python ./vec_arith_pca.py", "gpu_count": 2},
            {"command": "python ./vec_arith.py", "gpu_count": 1},
            {"command": "python ./st_gan.py train", "gpu_count": 1},
            {"command": "python ./train_cvae.py", "gpu_count": 1},
        ]
        stage_two_specs = [
            {"command": "python ./train_scGen.py", "gpu_count": 4},
        ]
        manifest_path, manifest = _start_run_manifest(
            model_to_train=model_to_train,
            overwrite=overwrite,
            stage_one_specs=stage_one_specs,
            stage_two_specs=stage_two_specs,
        )
        print(f"Run manifest: {manifest_path}")
        try:
            stage_one_exit_code = run_command_specs_parallel(
                stage_one_specs,
                cwd=SCRIPT_DIR,
                overwrite=overwrite,
            )
            manifest["stages"]["stage_one"] = {"exit_code": stage_one_exit_code}
            _write_manifest(manifest_path, manifest)
            if stage_one_exit_code != 0:
                raise RuntimeError(
                    "Parallel stage 1 failed (vec_arith_pca, vec_arith, st_gan, train_cvae). "
                    "Aborting remaining training steps."
                )

            stage_two_exit_code = run_command_specs_parallel(
                stage_two_specs,
                cwd=SCRIPT_DIR,
                overwrite=overwrite,
            )
            manifest["stages"]["stage_two"] = {"exit_code": stage_two_exit_code}
            _write_manifest(manifest_path, manifest)
            if stage_two_exit_code != 0:
                raise RuntimeError(
                    "Parallel stage 2 failed (train_scGen). "
                    "Aborting batch-correction generation."
                )

            ensure_batch_correction_outputs(overwrite=overwrite)
            manifest["stages"]["post_stage"] = {"exit_code": 0}
            manifest["status"] = "success"
            manifest["end_time_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            _write_manifest(manifest_path, manifest)
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            manifest["traceback"] = traceback.format_exc()
            manifest["end_time_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            _write_manifest(manifest_path, manifest)
            raise

    elif model_to_train == "PCA":
        command = "python ./vec_arith_pca.py"
        run_command(command, overwrite=overwrite)
    elif model_to_train == "VecArithm":
        command = "python ./vec_arith.py"
        run_command(command, overwrite=overwrite)
    elif model_to_train == "STGAN":
        command = "python ./st_gan.py train"
        run_command(command, overwrite=overwrite)
    elif model_to_train == "CVAE":
        command = "python ./train_cvae.py"
        run_command(command, overwrite=overwrite)
    elif model_to_train == "scGen":
        command = "python ./train_scGen.py"
        run_command(command)


if __name__ == '__main__':
    main()
