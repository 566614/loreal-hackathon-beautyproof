import json, os, glob
base = os.path.expanduser("~/.cache/huggingface/hub/models--Organika--sdxl-detector")
if not os.path.isdir(base):
    print("CACHE_MISSING")
else:
    snaps = glob.glob(os.path.join(base, "snapshots", "*"))
    print("snapshots:", len(snaps))
    for s in snaps:
        files = os.listdir(s)
        print("snapshot files:", len(files))
        # check model weights + config
        for f in ["config.json"]:
            p = os.path.join(s, f)
            if os.path.exists(p):
                try:
                    json.load(open(p, encoding="utf-8"))
                    print(f"  {f}: VALID")
                except Exception as e:
                    print(f"  {f}: CORRUPT {e}")
        # list any weight file
        for w in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"]:
            p = os.path.join(s, w)
            if os.path.exists(p):
                print(f"  {w}: {os.path.getsize(p)} bytes")
    # incomplete blobs remaining?
    blobs = glob.glob(os.path.join(base, "blobs", "*.incomplete"))
    print("incomplete blobs left:", len(blobs))
