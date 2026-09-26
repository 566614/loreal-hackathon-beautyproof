# -*- coding: utf-8 -*-
# 来源：原创 —— BeautyProof 团队共享图片工具（pipeline / llm_explainer 共用，消除重复实现）
"""图片 → base64 data URI 的共享工具。

为什么单独放一个模块：
    pipeline.to_data_uri、llm_explainer._encode_image、web/review_app._to_data_uri
    原本是三份几乎一样的「缩略图 + JPEG 编码 + base64」实现。把唯一正确的版本放这里，
    谁要用谁 import，避免改一处忘改另一处导致预览/编码行为不一致（本期先收敛 tools 内两份）。
"""
import base64
import io
from pathlib import Path
from typing import Union

try:
    from PIL import Image
except Exception:  # noqa: BLE001
    Image = None  # 极端环境没装 Pillow 时优雅降级，返回 None

ImageLike = Union[str, "Path", "Image.Image"]


def image_to_base64(image_like: ImageLike, max_side: int = 900, quality: int = 85) -> Union[str, None]:
    """把图片压成缩略图再转成 data URI，失败则返回 None（不抛异常）。

    image_like 可以是：文件路径（str / Path），或已经打开的 PIL.Image.Image。
    后者用于调用方已经打开过图片的场景，避免重复读取文件。
    返回的字符串形如 "data:image/jpeg;base64,....."，可直接塞进 HTML / JSON。
    """
    if Image is None:
        return None
    try:
        if isinstance(image_like, Image.Image):
            img = image_like.convert("RGB")
        else:
            img = Image.open(image_like).convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None
