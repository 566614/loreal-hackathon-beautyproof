import shutil, os
p = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--Organika--sdxl-detector")
print("before exists:", os.path.isdir(p))
shutil.rmtree(p, ignore_errors=True)
print("after exists:", os.path.isdir(p))
