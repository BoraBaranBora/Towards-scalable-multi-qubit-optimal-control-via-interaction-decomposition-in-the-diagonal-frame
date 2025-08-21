#!/usr/bin/env python3
# ======= Set your path here (file OR directory) =======
TARGET_PATH = "results/pulse_2025-08-21_10-28-42/propagator_projected.pt"
PRETTY = True
WEIGHTS_ONLY = False
# ======================================================

import sys, json
from pathlib import Path
import torch
import numpy as np

def tensor_to_jsonable(t: torch.Tensor) -> dict:
    t = t.detach().cpu()
    obj = {"type": "torch_tensor", "dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape)}
    if t.is_complex():
        a = t.numpy()
        obj["representation"] = "complex_pairs"
        obj["data"] = np.stack([a.real, a.imag], axis=-1).tolist()
    else:
        obj["representation"] = "real"
        obj["data"] = t.tolist()
    return obj

def numpy_to_jsonable(a: np.ndarray) -> dict:
    a = np.asarray(a)
    obj = {"type": "numpy_array", "dtype": str(a.dtype), "shape": list(a.shape)}
    if np.iscomplexobj(a):
        obj["representation"] = "complex_pairs"
        obj["data"] = np.stack([a.real, a.imag], axis=-1).tolist()
    else:
        obj["representation"] = "real"
        obj["data"] = a.tolist()
    return obj

def scalar_to_jsonable(x):
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    if isinstance(x, complex):
        return {"type": "complex", "data": [float(x.real), float(x.imag)]}
    if isinstance(x, np.generic):
        if np.iscomplexobj(x):
            return {"type": "complex", "data": [float(x.real), float(x.imag)]}
        return x.item()
    return repr(x)

def to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return tensor_to_jsonable(obj)
    if isinstance(obj, np.ndarray):
        return numpy_to_jsonable(obj)
    if isinstance(obj, (bool, int, float, str, complex, np.generic)) or obj is None:
        return scalar_to_jsonable(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return repr(obj)

# --- add these config flags near the top if you want ---
DECIMALS = 2        # number of decimals for text/CSV formatting
MAKE_HEATMAPS = True  # set False to skip PNG heatmaps (requires matplotlib)
# -------------------------------------------------------

def _format_complex(z, decimals=6):
    return f"{float(np.real(z)):.{decimals}f}{float(np.imag(z)):+.{decimals}f}i"

def _export_matrix_views(arr: np.ndarray, base: Path, name_hint: str | None = None):
    """
    Save human-friendly views of a 2-D complex/real matrix.
    base: path WITHOUT extension (e.g., /path/to/optimized_propagator)
    name_hint: optional suffix like 'propagator' or dict key name
    """
    stem = f"{base.name}" if name_hint is None else f"{base.name}_{name_hint}"
    out_dir = base.parent

    # 1) Combined "a+bi" CSV
    as_str = np.vectorize(lambda z: _format_complex(z, DECIMALS))(arr)
    np.savetxt(out_dir / f"{stem}.csv", as_str, fmt="%s", delimiter=",")

    # 2) Real/Imag CSVs (useful for Excel/NumPy)
    np.savetxt(out_dir / f"{stem}_real.csv", np.real(arr), delimiter=",", fmt=f"%.{DECIMALS}g")
    if np.iscomplexobj(arr):
        np.savetxt(out_dir / f"{stem}_imag.csv", np.imag(arr), delimiter=",", fmt=f"%.{DECIMALS}g")

    # 3) Pretty TXT
    with (out_dir / f"{stem}_pretty.txt").open("w", encoding="utf-8") as f:
        f.write(f"# shape: {arr.shape}, dtype: {arr.dtype}\n")
        for row in arr:
            row_str = "  ".join(_format_complex(x, DECIMALS) if np.iscomplexobj(arr) else f"{float(x):.{DECIMALS}f}" for x in row)
            f.write("[ " + row_str + " ]\n")

    # 4) Optional heatmaps (|U| and arg(U))
    if MAKE_HEATMAPS:
        try:
            import matplotlib.pyplot as plt
            # Magnitude
            plt.figure()
            plt.imshow(np.abs(arr))
            plt.title(f"|{stem}|")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(out_dir / f"{stem}_magnitude.png", dpi=160)
            plt.close()

            # Phase only if complex
            if np.iscomplexobj(arr):
                plt.figure()
                plt.imshow(np.angle(arr))
                plt.title(f"arg({stem})")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(out_dir / f"{stem}_phase.png", dpi=160)
                plt.close()
        except Exception as e:
            print(f"[warn] Heatmaps not created: {e}")

def _maybe_export_human_readable(obj, p: Path):
    """
    If obj is a 2-D tensor (or a dict of such), export readable matrix files.
    """
    base = p.with_suffix("")  # strip .pt
    if isinstance(obj, torch.Tensor) and obj.ndim == 2:
        _export_matrix_views(obj.detach().cpu().numpy(), base)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                _export_matrix_views(v.detach().cpu().numpy(), base, name_hint=str(k))

def convert_file(p: Path):
    # Load
    try:
        if WEIGHTS_ONLY:
            obj = torch.load(p, map_location="cpu", weights_only=True)  # torch>=2.0
        else:
            obj = torch.load(p, map_location="cpu")
    except TypeError:
        obj = torch.load(p, map_location="cpu")

    # Always write JSON (your original behavior)
    out_json = p.with_suffix(".json")
    with out_json.open("w", encoding="utf-8") as f:
        if PRETTY:
            json.dump(to_jsonable(obj), f, indent=2)
        else:
            json.dump(to_jsonable(obj), f, separators=(",", ":"))
    print(f"[ok] {p} -> {out_json}")

    # Additionally export human-friendly files for matrices
    _maybe_export_human_readable(obj, p)

target = Path(TARGET_PATH).expanduser()
print(f"[*] Looking for: {target.resolve()}")
if not target.exists():
    print(f"[error] Path not found: {target.resolve()}")
    sys.exit(1)

if target.is_dir():
    # Gather .pt/.pth in the directory
    files = sorted([p for p in target.iterdir()
                    if p.is_file() and p.suffix.lower() in {".pt", ".pth"}])
    if not files:
        print(f"[error] No .pt/.pth files found in directory: {target.resolve()}")
        sys.exit(2)
    print(f"[*] Found {len(files)} file(s). Converting...")
    for p in files:
        try:
            convert_file(p)
        except Exception as e:
            print(f"[fail] {p}: {e}")
else:
    # Single file checks
    if target.suffix.lower() not in {".pt", ".pth"}:
        print(f"[error] Not a .pt/.pth file: {target.name}")
        sys.exit(3)
    try:
        convert_file(target)
    except Exception as e:
        print(f"[fail] {target}: {e}")
        sys.exit(4)

