# BeautyProof 可运行源码包：启动说明

源码 ZIP 不包含 Python 第三方依赖、虚拟环境或 site-packages。依赖版本清单保存在 ZIP 内的 requirements.txt。

## Windows PowerShell

解压后进入 BeautyProof 目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python web/app.py
```

浏览器打开 http://127.0.0.1:5000。命令行单图快检：

```powershell
python tools/pipeline.py "图片路径.jpg" --fast
```

双击 demo/index.html 可以打开不依赖 Python 包的离线交互演示。项目自训 AIGC v4 权重已包含；TruFor 第三方代码和权重需按其许可另行准备。原始训练和评测图片未放进源码包。
