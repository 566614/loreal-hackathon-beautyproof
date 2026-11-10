# BeautyProof 本地审核运行说明

用项目现有 Flask 入口运行。修改前端后在 web/frontend 执行 npm ci、npm run build，随后回到项目根目录执行 python -B web/app.py。启动环境、依赖下载目录和模型目录设在 D 盘。

## 图像模型

将 BP_MODEL_DIR 指向已准备的本地模型目录。预期结构：

    D:\BeautyProof-models\
      dda\                      官方 Dual-Data-Alignment 仓库
      dinov2\                   官方 facebookresearch/dinov2 仓库
      DDA_ckpt.pth              作者发布的 4390d902 版本
      iml-vit\                  官方 SunnyHaze/IML-ViT 仓库
      iml-vit_checkpoint.pth    作者确认许可的官方权重
      paddlex\                  PP-OCRv5 mobile 下载缓存

本轮使用 DDA 源码 08d6b414a33826403b9096cfdb7758d5eae12dbb 和 IML-ViT 源码 07dd2be0f4ea27a5c97c9fa5ffbe236733833eac。两个官方权重 SHA-256 分别为 b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e 和 7631fe852e139d904ad96fd766a37120d703c966cb80dfad70ffae4ff426cce4。安装匹配显卡与 Python 的 torch/torchvision，以及 timm、fvcore、Pillow、NumPy。OCR 采用 PaddleOCR 3.x 与 PaddlePaddle 3.x。运行每个模型前用一张图验证加载、设备和输出。

- 基线对照用 BP_AIGC_BACKEND=v4；仅在许可范围内用 BP_EDIT_BACKEND=trufor 跑旧定位基线。
- 盲测与试用完成后，BP_ORIGIN_PROFILE 指向审核负责人批准的阈值 JSON。报告核对权重、预处理和模型身份。校准程序写出的初始配置为未批准。
- 配置 BP_CLAIM_MODEL=qwen3:4b-instruct-2507-q4_K_M 和 BP_VISION_MODEL=qwen3-vl:4b-instruct-q4_K_M，指向本地 Ollama。可使用 BP_LLM_URL 更改本地端口。Ollama 使用 D 盘 OLLAMA_MODELS 与 TEMP。页面勾选「本地模型辅助」启用；服务不可用时文字规则继续运行。

## 已核验功效资料

审核负责人先核对原始出处、版本和适用条件，再在项目外准备本地 JSON 清单，用 BP_VERIFIED_REFERENCES 指向该文件。以下是格式示例，内容需要用真实审核记录替换：

    [
      {
        "id": "study-001",
        "source": "报告名称",
        "version": "v1",
        "locator": "第 2 页，表 1",
        "reviewer": "审核人代号",
        "product": "具体产品及规格",
        "excerpt": "从原报告复制的对应原文"
      }
    ]

网页通过目录选择已核验资料。表单粘贴的摘录单独标记为来源待核验。清单配置、审核素材与权重放在本地。
