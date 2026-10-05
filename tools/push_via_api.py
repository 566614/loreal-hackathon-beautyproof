"""git HTTPS 通道被 TLS 重置时的兜底推送：改用 GitHub Git Data API 写远端 main。

用法：
    python tools/push_via_api.py                 # 基线自动取远端 main
    python tools/push_via_api.py <base_sha>      # 手工指定父提交

流程：blobs -> trees(基于远端当前 tree) -> commits -> PATCH refs/heads/main

两个必须知道的坑（都踩过）：
1. 内容必须取自 git blob（`git cat-file blob HEAD:<path>`），不能直接读工作区文件——
   本机 core.autocrlf=true，工作区是 CRLF、提交里是 LF，直接读会把 CRLF 推上去，
   导致远端和本地"内容一致但字节不一致"。
2. 远端 commit SHA 会与本地不同（提交者/时间戳不同），内容一致。
   推完必须善后：git fetch 拿到该对象后 git reset --soft <remote_sha>，
   并 update-ref refs/remotes/origin/main <remote_sha>。
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO = "566614/loreal-hackathon-beautyproof"
def _find_exe(env_var, name, *fallbacks, hint=""):
    """三级定位可执行文件：环境变量 > 标准 PATH > 本机兜底路径（存在才用）。

    写死绝对路径会让脚本换机即废；这里既保证本机行为不变，又让换机的人有明确出路。
    """
    p = os.environ.get(env_var)
    if p and os.path.isfile(p):
        return p
    found = shutil.which(name)
    if found:
        return found
    for fb in fallbacks:
        if os.path.isfile(fb):
            return fb
    raise SystemExit(
        f"找不到 {name} 可执行文件。请任选一种方式配置：\n"
        f"  1) 设置环境变量 {env_var}=<{name} 的完整路径>\n"
        f"  2) 把 {name} 加入 PATH 后重试\n"
        f"{('  ' + hint) if hint else ''}"
    )


GH = _find_exe("BEAUTYPROOF_GH", "gh",
               hint="（GitHub CLI：https://cli.github.com）")
GIT = _find_exe("BEAUTYPROOF_GIT", "git", r"C:/Program Files/Git/cmd/git.exe",
                hint="（Git for Windows 或系统自带 git）")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env = dict(os.environ)
env["HTTP_PROXY"] = "http://127.0.0.1:7897"
env["HTTPS_PROXY"] = "http://127.0.0.1:7897"


def gh(method, endpoint, payload=None, retries=25):
    """调 gh api。网络极不稳（TLS handshake timeout / EOF / 偶发 400），必须重试。"""
    cmd = [GH, "api", "-X", method, endpoint]
    tmp = None
    if payload is not None:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(payload, tmp, ensure_ascii=False)
        tmp.close()
        cmd += ["--input", tmp.name]
    last = ""
    for i in range(retries):
        p = subprocess.run(cmd, capture_output=True, env=env, cwd=ROOT)
        if p.returncode == 0:
            out = p.stdout.decode("utf-8", "replace")
            if tmp:
                os.unlink(tmp.name)
            return json.loads(out) if out.strip() else {}
        last = (p.stderr.decode("utf-8", "replace") or p.stdout.decode("utf-8", "replace")).strip()
        print(f"  retry {i + 1}: {last[:160]}", flush=True)
        time.sleep(4)
    if tmp and os.path.exists(tmp.name):
        os.unlink(tmp.name)
    raise SystemExit(f"GH API FAILED {method} {endpoint}: {last}")


def git_bytes(*args):
    return subprocess.run([GIT, *args], capture_output=True, cwd=ROOT).stdout


def git_text(*args):
    return git_bytes(*args).decode("utf-8", "replace").strip()


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else gh("GET", f"repos/{REPO}/commits/main")["sha"]
    print(f"[1/5] 远端基线 {base}")
    base_tree = gh("GET", f"repos/{REPO}/commits/{base}")["commit"]["tree"]["sha"]
    print(f"      base_tree={base_tree}")

    print("[2/5] 收集变更文件")
    toks = [t for t in git_bytes("diff", "--name-status", "-z", base, "HEAD").split(b"\x00") if t]
    pairs = [(toks[i].decode("utf-8", "replace"),
              toks[i + 1].decode("utf-8", "replace") if i + 1 < len(toks) else "")
             for i in range(0, len(toks), 2)]
    entries = []
    for status, path in pairs:
        if status == "D":
            entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
            continue
        data = git_bytes("cat-file", "blob", f"HEAD:{path}")
        print(f"      {status} {path} ({len(data)} bytes)", flush=True)
        blob = gh("POST", f"repos/{REPO}/git/blobs", {
            "content": base64.b64encode(data).decode("ascii"),
            "encoding": "base64",
        })
        entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})

    print(f"[3/5] 建 tree（{len(entries)} 项）")
    tree = gh("POST", f"repos/{REPO}/git/trees", {"base_tree": base_tree, "tree": entries})
    print(f"      tree={tree['sha']}")

    print("[4/5] 建 commit")
    commit = gh("POST", f"repos/{REPO}/git/commits",
                {"message": git_text("log", "-1", "--pretty=%B"),
                 "tree": tree["sha"], "parents": [base]})
    print(f"      commit={commit['sha']}")

    print("[5/5] 更新 refs/heads/main")
    gh("PATCH", f"repos/{REPO}/git/refs/heads/main", {"sha": commit["sha"], "force": True})
    chk = gh("GET", f"repos/{REPO}/commits/main")
    print(f"      远端 main = {chk['sha']}")
    if chk["sha"] != commit["sha"]:
        raise SystemExit("远端校验不一致")

    print("OK")
    print("善后：git fetch 后执行")
    print(f"  git update-ref refs/remotes/origin/main {commit['sha']}")
    print(f"  git reset --soft {commit['sha']}")


if __name__ == "__main__":
    main()
