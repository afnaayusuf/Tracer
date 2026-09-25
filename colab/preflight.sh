#!/usr/bin/env bash
# Print what the runtime actually is before anything assumes it. Safe to run anywhere.
py() { python3 - "$@" << 'PY'
import importlib, json, shutil, subprocess, sys
r = {"python": sys.version.split()[0], "ensurepip": importlib.util.find_spec("ensurepip") is not None}
for m in ("torch", "torchaudio", "torchvision", "transformers", "vllm", "rfdetr", "av", "sqlalchemy", "psycopg"):
    try:
        mod = importlib.import_module(m)
        r[m] = getattr(mod, "__version__", "?")
        if m == "torch":
            r["torch_cuda"] = getattr(mod.version, "cuda", None)
            r["cuda_available"] = bool(mod.cuda.is_available())
            if mod.cuda.is_available():
                free, total = mod.cuda.mem_get_info()
                r["gpu"] = f"{mod.cuda.get_device_name(0)} free {free/1e9:.1f}/{total/1e9:.1f} GB"
    except Exception as e:
        r[m] = f"absent ({type(e).__name__})"
r["uv"] = shutil.which("uv") is not None
r["disk_free_gb"] = round(shutil.disk_usage("/").free / 1e9, 1)
print(json.dumps(r))
PY
}
py
