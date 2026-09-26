# -*- coding: utf-8 -*-
"""从 Wikimedia Commons 拉取「逐张核验过许可」的欧莱雅相关图，作为赛题2 合规补充参考集。

合规约束（见 docs/标准图片采集与合规清单.md）：
- 只收 CC0 / CC-BY(非SA,非NC) / Public Domain；排除 CC-BY-SA、CC-BY-NC、GFDL、商标/Copyright。
- 每张图配 <name>_license.txt 记录来源 URL + 许可 + 作者；并写入 results/standard_images_manifest.json。
- 绝不碰品牌/电商站图（那些 ToS 禁止复制与商用）。
用法：python tools/collect_cc_images.py
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "cc_verified"
MANIFEST = REPO / "results" / "standard_images_manifest.json"
CATS = ["Category:L'Oréal", "Category:L'Oréal products", "Category:L'Oréal Paris"]
API = "https://commons.wikimedia.org/w/api.php"
MAX_DOWNLOAD = 50
UA = "BeautyProof/1.0 (Loreal hackathon compliance image collection)"


def api(params):
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(10):  # 代理偶发 SSL 超时，重试(退避递增)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(8 * (attempt + 1))
            continue
    raise last


def get_category_files(cat, limit=200):
    files, cont = [], None
    while len(files) < limit:
        p = {"action": "query", "format": "json", "list": "categorymembers",
             "cmtitle": cat, "cmtype": "file", "cmlimit": 50}
        if cont:
            p["cmcontinue"] = cont
        d = api(p)
        for m in d.get("query", {}).get("categorymembers", []):
            files.append(m["title"])
        cont = d.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return files[:limit]


def get_imageinfo(titles):
    p = {"action": "query", "format": "json", "prop": "imageinfo",
         "iiprop": "url|extmetadata|mime",
         "iiextmetadatafilter": "License|LicenseShortName|UsageTerms|Artist|Credit|DescriptionUrl",
         "titles": "|".join(titles)}
    d = api(p)
    out = {}
    for pid, pg in d.get("query", {}).get("pages", {}).items():
        ii = pg.get("imageinfo")
        if ii:
            out[pg["title"]] = ii[0]
    return out


def license_ok(em):
    s = (em.get("LicenseShortName", {}).get("value", "") + " " +
         em.get("UsageTerms", {}).get("value", "")).lower()
    norm = s.replace("-", " ")  # 兼容 "cc-by" 与 "cc by" 两种写法
    if "cc0" in norm:
        return True, s.strip()
    if "public domain" in norm:
        return True, s.strip()
    if "cc by" in norm and "sa" not in norm and "nc" not in norm:
        return True, s.strip()
    return False, s.strip()


def sanitize(name):
    name = re.sub(r"[^A-Za-z0-9 _\-.]", "_", name)
    return name[:120]


def strip_html(x):
    return re.sub("<[^>]+>", "", x or "").strip()[:120]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    collected = manifest.get("collected_cc", [])
    done_urls = {c.get("source", "") for c in collected}  # 已下载的跳过，避免重复

    titles = []
    for cat in CATS:
        titles += get_category_files(cat, 200)
        time.sleep(0.2)
    titles = list(dict.fromkeys(titles))
    print(f"类目文件总数: {len(titles)}", flush=True)

    ok, skipped = [], []
    for i in range(0, len(titles), 20):
        info = get_imageinfo(titles[i:i + 20])
        for t, ii in info.items():
            em = ii.get("extmetadata", {})
            flag, lic = license_ok(em)
            if not flag:
                skipped.append((t, lic))
                continue
            if not ii.get("mime", "").startswith("image"):
                skipped.append((t, lic + " [non-image]"))
                continue
            ok.append((t, ii, em, lic))
        time.sleep(0.2)

    print(f"许可通过(待下载): {len(ok)}；跳过: {len(skipped)}", flush=True)
    for t, ii, em, lic in ok:
        print(f"  候选: {t}  [{lic}]  mime={ii.get('mime','?')}", flush=True)
    if skipped:
        print("跳过样本(前10):", flush=True)
        for t, lic in skipped[:10]:
            print(f"  - {t}  [{lic}]", flush=True)

    n = 0
    for t, ii, em, lic in ok:
        if n >= MAX_DOWNLOAD:
            break
        if ii.get("descriptionurl", "") in done_urls:  # 已下载过，跳过
            print(f"  已存在，跳过: {t}", flush=True)
            continue
        base = sanitize(t.split(":", 1)[-1])
        dst = OUT / base
        if dst.exists():
            base = sanitize(t.split(":", 1)[-1] + "_" + str(abs(hash(t)) & 0xffff))
            dst = OUT / base
        success = False
        for attempt in range(4):  # 429 限流退避
            try:
                req = urllib.request.Request(ii["url"], headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as r:
                    dst.write_bytes(r.read())
                success = True
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                print(f"  下载失败: {base}  HTTP {e.code}", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                # 代理偶发 SSL EOF / 握手超时，退避后重试（与 429 同样处理）
                if attempt < 3:
                    print(f"  下载失败(重试 {attempt+1}/4): {base}  {str(e)[:60]}", flush=True)
                    time.sleep(8 * (attempt + 1))
                    continue
                print(f"  下载失败(放弃): {base}  {str(e)[:80]}", flush=True)
                break
        if not success:
            skipped.append((t, "download-fail"))
            continue
        artist = strip_html(em.get("Artist", {}).get("value", "?"))
        lic_txt = (f"source: {ii.get('descriptionurl', '')}\n"
                   f"license: {lic}\n"
                   f"artist: {artist}\n"
                   f"retrieved: {time.strftime('%Y-%m-%d')}\n"
                   f"note: 仅用于赛题2 合规参考集；CC-BY 需保留署名。\n")
        (OUT / (base + "_license.txt")).write_text(lic_txt, encoding="utf-8")
        collected.append({
            "file": f"data/cc_verified/{base}", "source": ii.get("descriptionurl", ""),
            "license": lic, "is_retouched": "unknown",
            "purpose": "REAL训练补充/产品参考", "in_training": False,
            "risk": "low", "note": "Wikimedia Commons CC 合规图，已逐张核验许可",
        })
        n += 1
        print(f"  + {base}  ({lic})", flush=True)
        time.sleep(2.5)  # 下载间限速，避免 429

    manifest["collected_cc"] = collected
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已下载 {n} 张到 {OUT}；manifest 已更新（collected_cc 共 {len(collected)} 条）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
