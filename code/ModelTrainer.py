import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
RECON_SCGEN_DIR = os.path.join(DATA_DIR, "reconstructed", "scGen")


def run_command(command):
    return subprocess.call(command, shell=True, cwd=SCRIPT_DIR)


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
        # os.path.join(DATA_DIR, "MouseAtlas.subset.h5ad"),
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


def ensure_batch_correction_outputs():
    required_outputs = [
        os.path.join(RECON_SCGEN_DIR, "pancreas.h5ad"),
        # os.path.join(RECON_SCGEN_DIR, "mouse_atlas.h5ad"),
    ]
    missing = missing_paths(required_outputs)
    if not missing:
        return
    if os.path.join(RECON_SCGEN_DIR, "pancreas.h5ad") in missing:
        print("Generating scGen batch-corrected pancreas dataset...")
        run_command("python ./pancreas.py")
    # if os.path.join(RECON_SCGEN_DIR, "mouse_atlas.h5ad") in missing:
    #     print("Generating scGen batch-corrected mouse atlas dataset...")
    #     run_command("python ./mouse_atlas.py")
    missing_after = missing_paths(required_outputs)
    if missing_after:
        missing_list = "\n".join(f"- {path}" for path in missing_after)
        raise FileNotFoundError(
            "Batch-corrected datasets are still missing after generation:\n"
            f"{missing_list}"
        )


def main():
    if len(sys.argv) == 1:
        model_to_train = "all"
    else:
        model_to_train = sys.argv[1]
    if model_to_train == "all":
        ensure_inputs_exist()
        command = "python ./vec_arith_pca.py"
        run_command(command)

        command = "python ./vec_arith.py"
        run_command(command)

        command = "python ./st_gan.py train"
        run_command(command)

        command = "python ./train_cvae.py"
        run_command(command)

        command = "python ./train_scGen.py"
        run_command(command)
        ensure_batch_correction_outputs()

    elif model_to_train == "PCA":
        command = "python ./vec_arith_pca.py"
        run_command(command)
    elif model_to_train == "VecArithm":
        command = "python ./vec_arith.py"
        run_command(command)
    elif model_to_train == "STGAN":
        command = "python ./st_gan.py train"
        run_command(command)
    elif model_to_train == "CVAE":
        command = "python ./train_cvae.py"
        run_command(command)
    elif model_to_train == "scGen":
        command = "python ./train_scGen.py"
        run_command(command)


if __name__ == '__main__':
    main()
