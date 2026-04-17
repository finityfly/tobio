"""Shared training utilities for Tobio notebooks."""
import torch


def setup_device() -> str:
    """Return YOLO device string ('0' for GPU, 'cpu' for CPU) and print info."""
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}  |  PyTorch: {torch.__version__}")
        return "0"
    print(f"No GPU found — training on CPU  |  PyTorch: {torch.__version__}")
    return "cpu"


def print_metrics(results, task: str = "detect") -> None:
    """Print a compact summary of key metrics from model.val() results.

    Args:
        results: The object returned by model.val().
        task:    "detect" (default) or "segment".
    """
    if results is None:
        return
    src = getattr(results, "seg" if task == "segment" else "box", None)
    if src is None:
        return
    rows = [
        ("mAP50",     getattr(src, "map50", None)),
        ("mAP50-95",  getattr(src, "map",   None)),
        ("Precision", getattr(src, "mp",    None)),
        ("Recall",    getattr(src, "mr",    None)),
    ]
    print("\n Metric       Value")
    print(" ──────────────────")
    for name, val in rows:
        if val is not None:
            print(f" {name:<13} {val:.4f}")
    print()
