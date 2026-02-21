import os
import subprocess
import time


def _base_seed():
    try:
        return int(os.environ.get("SCGEN_SEED", "4039"))
    except ValueError:
        return 4039


def _apply_reproducibility_env(env, seed):
    env["SCGEN_PROCESS_SEED"] = str(seed)
    env["PYTHONHASHSEED"] = str(seed)
    env["NUMPY_SEED"] = str(seed)
    env["TF_SEED"] = str(seed)
    enable_determinism = env.get("SCGEN_ENABLE_DETERMINISM", "1") != "0"
    if enable_determinism:
        env.setdefault("TF_DETERMINISTIC_OPS", "1")
        env.setdefault("TF_CUDNN_DETERMINISTIC", "1")


def get_available_gpu_ids():
    # Respect explicit visibility first so callers can constrain devices externally
    # (e.g., in Slurm, Docker, or a parent launcher script).
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        gpu_ids = [gpu_id.strip() for gpu_id in visible_devices.split(",") if gpu_id.strip() and gpu_id.strip() != "-1"]
        if gpu_ids:
            return gpu_ids
        # Convention: "-1" means "hide all GPUs / force CPU-only execution".
        if visible_devices.strip() == "-1":
            return []
    try:
        # Fall back to host-level GPU discovery when no explicit visibility is set.
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # No NVIDIA runtime available (or query failed): run on CPU.
        return []


def run_on_next_gpu(call_idx, call_name, fn, gpu_ids):
    if gpu_ids:
        # Round-robin assignment keeps placement deterministic and balances sequential calls.
        gpu_id = gpu_ids[call_idx % len(gpu_ids)]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Running {call_name} on GPU {gpu_id}")
    else:
        print(f"Running {call_name} on CPU (no GPU found)")
    # Calls remain synchronous; this utility only controls per-call device affinity.
    fn()


def run_commands_parallel(commands, cwd=None, overwrite=False):
    gpu_ids = get_available_gpu_ids()
    target_cwd = cwd or os.getcwd()
    max_workers = max(1, len(gpu_ids))
    pending = list(enumerate(commands))
    running = []
    exit_codes = []
    available_gpu_ids = list(gpu_ids)
    seed_base = _base_seed()
    while pending or running:
        while pending and len(running) < max_workers:
            command_idx, command = pending.pop(0)
            env = os.environ.copy()
            if overwrite:
                env["SCGEN_OVERWRITE"] = "1"
            else:
                env.pop("SCGEN_OVERWRITE", None)
            _apply_reproducibility_env(env, seed=seed_base + command_idx)
            if gpu_ids:
                gpu_id = available_gpu_ids.pop(0)
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                print(f"Launching `{command}` on GPU {gpu_id}")
            else:
                gpu_id = None
                print(f"Launching `{command}` on CPU (no GPU found)")
            process = subprocess.Popen(command, shell=True, cwd=target_cwd, env=env)
            running.append((command, process, gpu_id))
        next_running = []
        for command, process, gpu_id in running:
            exit_code = process.poll()
            if exit_code is None:
                next_running.append((command, process, gpu_id))
            else:
                print(f"`{command}` finished with exit code {exit_code}")
                exit_codes.append(exit_code)
                if gpu_id is not None:
                    available_gpu_ids.append(gpu_id)
        running = next_running
        if running:
            time.sleep(0.5)
    return max(exit_codes) if exit_codes else 0


def run_command_specs_parallel(command_specs, cwd=None, overwrite=False):
    gpu_ids = get_available_gpu_ids()
    target_cwd = cwd or os.getcwd()
    pending = [(idx, dict(spec)) for idx, spec in enumerate(command_specs)]
    running = []
    exit_codes = []
    total_gpu_count = len(gpu_ids)
    available_gpu_ids = list(gpu_ids)
    seed_base = _base_seed()
    while pending or running:
        launched = True
        while launched and pending:
            launched = False
            for pending_idx, (spec_idx, spec) in enumerate(pending):
                command = spec["command"]
                requested_gpu_count = int(spec.get("gpu_count", 1))
                if total_gpu_count == 0:
                    # CPU-only fallback: run one process at a time to avoid oversubscription.
                    if running:
                        continue
                    allocated_gpu_ids = []
                else:
                    need = max(1, min(requested_gpu_count, total_gpu_count))
                    if len(available_gpu_ids) < need:
                        continue
                    allocated_gpu_ids = available_gpu_ids[:need]
                    available_gpu_ids = available_gpu_ids[need:]
                env = os.environ.copy()
                if overwrite:
                    env["SCGEN_OVERWRITE"] = "1"
                else:
                    env.pop("SCGEN_OVERWRITE", None)
                _apply_reproducibility_env(env, seed=seed_base + spec_idx)
                if allocated_gpu_ids:
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(allocated_gpu_ids)
                    print(f"Launching `{command}` on GPUs {','.join(allocated_gpu_ids)}")
                else:
                    print(f"Launching `{command}` on CPU (no GPU found)")
                process = subprocess.Popen(command, shell=True, cwd=target_cwd, env=env)
                running.append((command, process, allocated_gpu_ids))
                pending.pop(pending_idx)
                launched = True
                break
        if pending and not running and total_gpu_count > 0:
            pending_commands = ", ".join(spec["command"] for _, spec in pending)
            raise RuntimeError(
                "Unable to schedule command specs on available GPUs. "
                f"Pending: {pending_commands}"
            )
        next_running = []
        for command, process, allocated_gpu_ids in running:
            exit_code = process.poll()
            if exit_code is None:
                next_running.append((command, process, allocated_gpu_ids))
            else:
                print(f"`{command}` finished with exit code {exit_code}")
                exit_codes.append(exit_code)
                if allocated_gpu_ids:
                    available_gpu_ids.extend(allocated_gpu_ids)
        running = next_running
        if running:
            time.sleep(0.5)
    return max(exit_codes) if exit_codes else 0
