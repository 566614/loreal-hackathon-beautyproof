# -*- coding: utf-8 -*-
"""2026-10-05 · 全面复检 · 第四批：异常链可追溯性（B904）

问题：`except` 里 `raise` 新异常时没有 `from`，Python 会打印
「During handling of the above exception, another exception occurred」的两段堆叠，
根因被埋在中间。对本项目尤其不该 —— 评委 / 法务看报告时，报错可追溯性就是可信度。

改动：4 处补 `from exc` / `from e`。**纯附加 `__cause__`，零行为变化**：
  1. tools/export_pdf.py:414  读文件失败 → ExportError
  2. tools/export_pdf.py:450  生成 PDF 失败 → ExportError
  3. tools/pipeline.py:312    图片不可识别 → ValueError
  4. tools/video_tool.py:93   缺 opencv → RuntimeError（需先把 `except ImportError:` 绑定为 `as exc`）

幂等：逐处断言命中 1 次。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATCHES = [
    ("tools/export_pdf.py",
     '''    except OSError as exc:
        raise ExportError("读取失败：{}（{}）".format(md_path.name, exc.strerror or exc))''',
     '''    except OSError as exc:
        raise ExportError("读取失败：{}（{}）".format(md_path.name, exc.strerror or exc)) from exc'''),

    ("tools/export_pdf.py",
     '''    except Exception as exc:  # noqa: BLE001
        raise ExportError(
            "生成 PDF 失败：{}（{}: {}）".format(md_path.name, exc.__class__.__name__, exc)
        )''',
     '''    except Exception as exc:  # noqa: BLE001
        raise ExportError(
            "生成 PDF 失败：{}（{}: {}）".format(md_path.name, exc.__class__.__name__, exc)
        ) from exc'''),

    ("tools/pipeline.py",
     '''    except Exception as e:  # noqa: BLE001
        raise ValueError(f"这不是一张能被识别的图片：{image_path.name}（{type(e).__name__}）")''',
     '''    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"这不是一张能被识别的图片：{image_path.name}（{type(e).__name__}）"
        ) from e'''),

    ("tools/video_tool.py",
     '''    except ImportError:
        raise RuntimeError(
            "未检测到 opencv（cv2）。请用项目隔离环境装：\\n"
            "  ./run.sh -m pip install opencv-python-headless\\n"
            f"（等价于：\\"{sys.executable}\\" -m pip install opencv-python-headless）\\n"
            "（绝不要用系统 pip 或全局安装）"
        )''',
     '''    except ImportError as exc:
        raise RuntimeError(
            "未检测到 opencv（cv2）。请用项目隔离环境装：\\n"
            "  ./run.sh -m pip install opencv-python-headless\\n"
            f"（等价于：\\"{sys.executable}\\" -m pip install opencv-python-headless）\\n"
            "（绝不要用系统 pip 或全局安装）"
        ) from exc'''),
]


def main():
    by_file = {}
    for rel, old, new in PATCHES:
        by_file.setdefault(rel, []).append((old, new))

    for rel, pairs in by_file.items():
        p = ROOT / rel
        text = p.read_text(encoding="utf-8")
        for old, new in pairs:
            if new in text and old not in text:
                print(f"[skip] {rel} 已应用")
                continue
            n = text.count(old)
            assert n == 1, f"{rel} 命中 {n} 次：{old[:56]}"
            text = text.replace(old, new)
            print(f"[ok]   {rel}  ← {old[:52]}")
        p.write_text(text, encoding="utf-8")
    print("[done] B904 异常链补全（4 处，零行为变化）")


if __name__ == "__main__":
    main()
