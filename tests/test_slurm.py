from omegaconf import OmegaConf

from src.app.pipeline.slurm import generate_sbatch_header


def test_slurm_compute_node_defaults():
    site_cfg = OmegaConf.create({
        "name": "ncshare",
        "mail_user": "test@example.com",
        "mail_type": "END,FAIL",
        "resources": {
            "partition": "gpu",
            "cores": 16,
            "memory": "32G",
            "time": "01:00:00",
            "gpus": 1,
        },
        "compute_resources": {
            "partition": "common",
            "cores": 4,
            "standalone_cores": 4,
            "memory": "8G",
            "standalone_memory": "8G",
            "time": "01:00:00",
            "gpus": 0,
        },
    })

    # Test 1: Targeting compute nodes (common partition) without explicit cores/memory in exp
    cfg = OmegaConf.create({
        "site": site_cfg,
        "partition": "common",
        "consolidate": False,
        "resources": {
            "cores": None,
            "memory": None,
        },
    })
    header = generate_sbatch_header("test_job", "logs", cfg, is_consolidated=False)
    assert "#SBATCH --partition=common" in header
    assert "#SBATCH --cpus-per-task=4" in header
    assert "#SBATCH --mem=8G" in header
    assert "--gres" not in header

    # Test 2: Targeting GPU partition retains 16 cores and 32G
    cfg_gpu = OmegaConf.create({
        "site": site_cfg,
        "partition": "gpu",
        "consolidate": False,
        "resources": {
            "cores": None,
            "memory": None,
        },
    })
    header_gpu = generate_sbatch_header("test_gpu_job", "logs", cfg_gpu, is_consolidated=False)
    assert "#SBATCH --partition=gpu" in header_gpu
    assert "#SBATCH --cpus-per-task=16" in header_gpu
    assert "#SBATCH --mem=32G" in header_gpu
    assert "#SBATCH --gres=gpu:1" in header_gpu
