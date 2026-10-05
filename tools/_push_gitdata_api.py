"""Push via GitHub Git Data API (fallback when git TLS fails in sandbox).

Usage: python tools/_push_gitdata_api.py <repo> <local_dir> <branch> <msg>
Replicates `git status --porcelain` entries into one commit:
  blobs -> trees -> commits -> PATCH refs/heads/<branch>
"""
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 定位 gh：环境变量 > 标准 PATH > 本机兜底（仅当该路径确实存在时使用）。
# 写死绝对路径会让这个兜底脚本换机即废，故保留兜底但降级为最后一级。
GH = (os.environ.get("BEAUTYPROOF_GH")
      or shutil.which("gh")
      or r"C:/Users/Lenovo/.workbuddy/binaries/gh/bin/gh.exe")


def env():
    e = dict(os.environ)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        e[k] = "http://127.0.0.1:7897"
    return e


def gh(args):
    r = subprocess.run([GH, "api"] + args, capture_output=True, env=env(), cwd=".")
    out = r.stdout.decode("utf-8", "ignore")
    if r.returncode != 0:
        raise SystemExit("GH FAIL: %s\n%s\n%s" % (args[:4], out[:600],
                                                  r.stderr.decode("utf-8", "ignore")[:400]))
    return out.strip()


def gh_json(args):
    return json.loads(gh(args))


def main():
    repo, local, branch, msg = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    br = branch.replace("refs/heads/", "")
    head = gh([f"repos/{repo}/commits/{br}", "--jq", ".sha"])
    print("remote head:", head)

    paths = sys.argv[5:]  # explicit file list; git status quoting breaks on CJK paths
    print("targets:", paths)

    # NOTE: ?recursive=1 is required, otherwise only the top-level tree is listed
    tsv = gh([f"repos/{repo}/git/trees/{head}?recursive=1", "--jq",
              "[.tree[] | .path, .sha, .type] | @tsv"])
    remote = {}
    for line in tsv.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        remote[parts[0]] = (parts[1], parts[2])
    print("remote files:", len(remote))

    entries, created = [], []
    for rel in paths:
        data = (local / rel).read_bytes()
        sha = gh([f"repos/{repo}/git/blobs", "-f",
                  "content=" + base64.b64encode(data).decode(),
                  "-f", "encoding=base64", "--jq", ".sha"])
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": sha})
        remote.pop(rel, None)
        created.append(rel)
        print("  blob", rel, sha[:10])

    keep = [{"path": p, "mode": "100644", "type": "blob", "sha": v[0]}
            for p, v in remote.items() if v[1] == "blob"]
    new_tree = gh_json([f"repos/{repo}/git/trees", "--jq", ".sha",
                        "--input", "-"]) if False else None
    # trees API needs a JSON body; use --input with a temp file
    payload = {"base_tree": head, "tree": keep + entries}
    Path("_t.json").write_text(json.dumps(payload), encoding="utf-8")
    new_tree = json.loads(gh([f"repos/{repo}/git/trees", "-X", "POST",
                              "-H", "Content-Type: application/json",
                              "--input", "_t.json"]))["sha"]
    print("new tree:", new_tree)

    Path("_c.json").write_text(json.dumps(
        {"message": msg, "tree": new_tree, "parents": [head]}), encoding="utf-8")
    commit = json.loads(gh([f"repos/{repo}/git/commits", "-X", "POST",
                            "-H", "Content-Type: application/json",
                            "--input", "_c.json"]))["sha"]
    print("new commit:", commit)

    Path("_r.json").write_text(json.dumps({"sha": commit}), encoding="utf-8")
    gh([f"repos/{repo}/git/refs/heads/{br}", "-X", "PATCH",
        "-H", "Content-Type: application/json", "--input", "_r.json"])
    for f in ("_t.json", "_c.json", "_r.json"):
        try:
            Path(f).unlink()
        except Exception:
            pass
    Path("_push_remote_sha.txt").write_text(commit, encoding="utf-8")
    print("PATCHED. remote head =", commit)


main()
