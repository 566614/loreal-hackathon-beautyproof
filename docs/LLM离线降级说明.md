# LLM 解释层 · 离线降级说明

> 适用代码：`tools/llm_explainer.py`、`tools/pipeline.py`（--llm 开关）、`tools/report_generator.py`
> 适用赛题：欧莱雅黑客松赛题2「信任守护师」· BeautyProof 美妆图取证
> 核心目标：**断网、无 Ollama、甚至任何意外异常，都能产出完整、可读、四段式的人话报告，绝不报错、绝不卡死。**

---

## 1. 离线模式是什么，怎么触发

「离线模式」不是一个新的开关，而是 LLM 解释层的**默认降级行为**。

只要满足下面任意一种情况，`--llm` 路径就会自动走离线降级：

| 触发条件 | 代码判断 | 行为 |
| --- | --- | --- |
| Ollama 服务没启动 / 端口不通 | `ollama_available()` 返回 `False` | 直接返回 `None` |
| 模型没注册进 Ollama（GGUF 没下/没 `ollama create`） | `ensure_model_created()` 返回 `False` | 返回 `None` |
| 模型调用超时 / 连接被拒 / 返回空 | `_chat()` 内部 `try-except` 兜底 | 返回 `None` |
| 任何意外异常（JSON 解析错误、网络抖动、未知 bug） | `explain()` 外层 `try-except` 兜底 | 返回 `None`，打印日志 |

也就是说：**只要有任何一环不 OK，就降级，不阻塞主流程。**
所谓「断网答辩现场」——没装 Ollama、也没网拉模型——正好命中第一条，自动降级。

> 显式触发方式：直接 `python tools/pipeline.py <图> --llm`，在没 Ollama 的环境里它就是离线模式；
> 不需要任何额外参数。想强制纯规则报告，也可以干脆不传 `--llm`。

---

## 2. 降级时产出什么

降级后，**流水线照常跑完**，产出与在线模式完全一致的结构，只是少了「第五段：大模型辅助解读」：

- `outputs/analysis_<图>.json`：完整结果（结论、证据、工具清单、Agent 决策日志……）
- `reports/report_<图>.md`：**四段式人话鉴定报告**
  - 一、结论
  - 二、检测依据（逐工具）
  - 三、这些工具说不清什么（必读的能力边界）
  - 四、建议
- （在线时多一段）五、大模型辅助解读（可选，经规则护栏校验）

离线模式下，报告里**不会有「五、大模型辅助解读」**，也不会出现 `result["llm_explanation"]` 字段。
但前四段 100% 由 `report_generator.py` 的模板从真实工具证据拼出，**结论可追溯到 `outputs/` 下的原始证据 JSON**，可审计、可复现。

降级发生时会在终端打一行明确日志，方便现场一眼确认走了离线分支：

```
  [llm] offline/degraded, fallback to rule report (Ollama unreachable)
```

（括号内会说明具体降级原因：`Ollama unreachable` / `model not ready` / `empty response` / 或 `<异常类型>: <信息>`）

---

## 3. 与在线模式的差异

| 维度 | 离线模式（无 LLM） | 在线模式（本地 Ollama + Qwen3-VL-4B） |
| --- | --- | --- |
| 触发 | 默认（无 Ollama 即触发） | 装好 Ollama、拉好模型、`--llm` |
| 报告第五段 | 无 | 有，模型把证据翻译成大白话 |
| 结论定性 | 规则引擎（唯一权威） | 规则引擎（唯一权威，模型只翻译不判定） |
| 跑通所需依赖 | 仅 Python + 普通工具依赖 | 额外 ~2.5GB GGUF + Ollama 服务 |
| 速度 | 秒~分钟级（取决于其它工具） | 多等模型加载/推理时间 |
| 可审计性 | 完全可审计 | 可审计，第五段过禁用措辞护栏 |

**重要**：无论在线离线，**最终定性结论永远来自规则引擎**（`rule_engine.judge`）。
LLM 只负责「翻译/解释」，且输出会过 `validator` 的禁用措辞关（"确认造假"等越界表述会被替换成"**疑似**"）。
离线只是少了这一段锦上添花的解释，不影响信任判定的权威性。

---

## 4. 护栏 / 不变量（为什么不会崩）

- `explain()` 与 `generate_for_pipeline()` 都被最外层 `try-except Exception` 包住：
  **任何异常都返回 `None`，绝不冒泡到 `pipeline.analyze`。**
- `pipeline.analyze` 在调用 LLM 层时又额外包了一层 `try-except`（见 `tools/pipeline.py` 的 `--llm` 分支），
  双重保险。
- 连接探测 `_ollama_tags()` 用 `timeout=5` 秒快速失败，不会卡住主流程。
- 所有外部 I/O（网络请求、读图、写文件）各自有 `try-except`，出错即降级。

---

## 5. 答辩现场怎么用

1. **断网直接跑**（最稳）：
   ```bash
   python tools/pipeline.py <图片路径> --fast --llm
   ```
   - `--fast` 跳过两个深度学习模型（AIGC / TruFor），秒出结论，避免现场模型加载慢/缺依赖。
   - `--llm` 开着也没关系：没 Ollama 会自动降级，报告照出，且日志会打印 `offline/degraded` 自证。
2. **看产出**：打开 `reports/report_<图>.md`，四段式结论直接能讲；`outputs/analysis_<图>.json` 备查。
3. **想展示「在线也能跑」**（现场有网+装好 Ollama 时）：
   ```bash
   python tools/download_vlm.py
   ollama create qwen3vl-4b -f models/qwen3vl-4b/Modelfile
   ollama serve
   python tools/pipeline.py <图片路径> --llm
   ```
   此时报告会多出「五、大模型辅助解读」段。
4. **自证降级有效**：`tests/test_offline_report.py` 用 monkeypatch 把 Ollama 调用模拟成「不可达 / 抛异常」，
   断言进程不抛异常、报告含四段式。现场可当众跑：
   ```bash
   python -m pytest tests/test_offline_report.py -v
   ```

---

## 6. 相关文件

- `tools/llm_explainer.py`：`explain()` / `generate_for_pipeline()` —— 降级逻辑与日志在此。
- `tools/pipeline.py`：`analyze()` 的 `--llm` 分支（双重 try-except 兜底）。
- `tools/report_generator.py`：`build_report()` —— 四段式纯规则报告，离线模式的产出主体。
- `tests/test_offline_report.py`：离线降级自动化测试。
