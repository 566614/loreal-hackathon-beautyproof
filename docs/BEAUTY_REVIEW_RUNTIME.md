# BeautyProof 本地审核运行说明

用项目现有 Flask 入口运行。启动环境、依赖下载目录和模型目录设在 D 盘。修改前端后，可以在 PowerShell 中指定 D 盘 npm 缓存再构建：

    $env:npm_config_cache = "D:\heckson\loreal-hackathon-beautyproof\.submission_work\npm-cache"
    $env:TEMP = "D:\heckson\loreal-hackathon-beautyproof\.submission_work\bp-next\temp"
    $env:TMP = $env:TEMP
    cd web/frontend
    npm ci
    npm run build

本机已准备好 D 盘虚拟环境时，在 PowerShell 运行以下命令检查依赖位置，去掉 -Check 启动工作台：

    powershell -ExecutionPolicy Bypass -File tools/start_local.ps1 -Check

脚本把模型、PyTorch、Paddle、Hugging Face、pip、uv 和临时文件的路径指向项目的 .submission_work/bp-next。虚拟环境复用系统已安装的 Python 基础运行时，不触发 Python 下载。脚本优先选择 D 盘 CUDA 环境，显卡不可用时回退到 CPU 环境。CUDA 环境运行 IML-ViT 前需要至少 5.5 GiB 可用系统内存和 5 GiB 可用显存；DDA 需要至少 4 GiB 可用系统内存，使用 CUDA 时还需要 3 GiB 可用显存。资源不足时对应模型显示不可用，释放资源后可重试。文字审核使用 DeepSeek API。启用文字审核后，正文、OCR 识别文字及产品名称会发送到 DeepSeek，图片留在本地。

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
- DeepSeek API 使用 DEEPSEEK_API_KEY 环境变量，默认模型为 deepseek-flash。可在启动环境中通过 BP_DEEPSEEK_MODEL 指定模型。密钥不可用、请求失败或输出无法解析时，报告显示模型不可用，规则审核继续运行。

## 配置 DeepSeek

在启动工作台的同一个 PowerShell 窗口中输入下列命令。密钥只传给当前进程及其子进程，不写入项目文件或聊天：

    $secret = Read-Host "DeepSeek API key" -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
    try { $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
    powershell -ExecutionPolicy Bypass -File tools/start_local.ps1

DeepSeek API 会产生服务费用，使用前通过 DeepSeek 官方平台配置账号。接口地址和模型名参见 https://api-docs.deepseek.com/。

启用 DeepSeek 时，正文、OCR 识别出的文字以及同产品同市场的资料摘录和条件会发送至该服务；原始图片由本地模型处理，不发送至 DeepSeek。应只纳入获准对外处理的文本资料。

## 已核验功效资料

审核负责人先核对原始出处、版本和适用条件，再通过工作台的「添加已核验资料」保存摘录。默认保存到 D 盘 .submission_work/bp-next/knowledge/verified.json。同一产品、指标和条件的依据会自动参与核验。也可以用 BP_VERIFIED_REFERENCES 指向自行维护的 JSON 清单。以下是格式示例，内容需要用真实审核记录替换：

    [
      {
        "id": "study-001",
        "source": "报告名称",
        "source_type": "efficacy_study",
        "market": "中国大陆",
        "product_version": "2025版",
        "version": "v1",
        "locator": "第 2 页，表 1",
        "reviewer": "审核人代号",
        "product": "具体产品及规格",
        "excerpt": "从原报告复制的对应原文"
      }
    ]

网页通过目录选择已核验资料。表单粘贴的摘录单独标记为来源待核验。清单配置、审核素材与权重放在本地。

选填的包装参考图仅用于本地寻找相似平面图案，结果是人工复核线索，无法确认产品版本、真伪或功效；匹配失败不代表图片不存在该产品。宣传图和参考图的上传总大小不能超过 32 MB。

资料可记录功效试验、产品标签、品牌宣称或技术方法，以及市场、产品版本、发布日期、方法和试验条件。品牌宣称、标签和方法资料只展示为匹配线索，不作为功效支持。自动支持要求资料与送审市场均明确且一致、产品版本明确且可从宣称原句识别一致；资料版本是报告版本，不能代替产品版本。当前可识别“2025版”“第四代”“新版”“旧版”等形式，但笼统的“新版”“旧版”不能确认具体版本，不能自动判为支持；未识别也需人工复核。填写的方法、样本量、人群、周期和条件须逐字出现在宣称原句中才能自动匹配，否则保留为线索、交人工核对。历史目录中缺少类型、市场或产品版本的记录不能自动作为支持依据。
