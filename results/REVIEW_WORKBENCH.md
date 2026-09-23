# 人工复核工作台（Review Workbench）

> 把「高风险必人工复核」从口号变成可操作的功能。

## 1. 解决什么

品牌方法务在审核美妆营销图时，面对算法给出的「高风险 / 可疑」结论，需要在一个界面里：

1. **看到证据**：原图、TruFor 定位叠加图与热力图、TruFor 分数 / 可疑占比 / 方位 / AIGC 概率等关键指标、规则引擎的判定依据与边界提醒；
2. **做出判定**：对每张图给出 **确认篡改 / 排除误报 / 需补充材料** 三种结论之一；
3. **留痕**：记录「谁、什么时间、什么结论、备注」，可追溯、可审计。

本工作台只处理 `outputs/analysis_*.json` 中 `risk_level` 为 `high_risk` / `suspicious` 的图（即「必人工复核」范围），不覆盖可信 / 无法判定的图。

## 2. 怎么起服务

独立 Flask 应用，**新增文件 `web/review_app.py`**，用项目隔离 venv 运行（Flask 已在该 venv 中）：

```bash
# 默认端口 5077（本地避免与现有 web/app.py:5000 冲突）
C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe web/review_app.py

# 或指定端口并对外（便于部署），端口读环境变量 PORT，bind 0.0.0.0
PORT=8080 C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe web/review_app.py
```

浏览器打开 `http://127.0.0.1:5077`（或指定端口）。页面为内嵌 HTML，**不依赖任何外部 CDN**，离线可用。

## 3. 界面能做什么

- **待复核队列**：首页网格卡片列出所有 high_risk / suspicious 图，高风险在前、同档按 TruFor 篡改分数降序。每张卡片显示图名、风险徽章、TruFor 分数、可疑占比、方位、AIGC 概率、定位图缩略，以及若已复核则显示结论。
- **复核详情（点击卡片「复核 →」）**：弹出面板展示原图、定位叠加图、TruFor 热力图、关键指标、一句话结论、判定依据逐条、建议下一步、边界提醒；底部表单填写「复核人 / 备注」并点击三种结论之一提交。
- **报告导出**：详情内「导出/预览鉴定报告」按钮调用 `tools/export_pdf.py` 生成该图 PDF 并返回下载；若 PDF 生成失败（如缺字体），自动回退为该图 `reports/report_<stem>.md` 的 Markdown 文本预览，保证总有可存档产物。

## 4. 复核结论怎么留痕

- `POST /api/review` 接收 `{stem, decision, reviewer, note}`，写入 **新建文件 `outputs/review_queue.json`**（只新增，不改动任何已有文件）。
- 存储结构：按 `stem` 为键存 `decision / decision_zh / reviewer / note / timestamp`；同一图再次提交即**更新**该条目。
- 队列卡片与详情面板都会读取该文件并显示「已复核」状态，形成闭环留痕。

## 5. 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 复核工作台主页（内嵌 HTML） |
| GET | `/api/queue` | 待复核队列：high_risk / suspicious 条目，按风险排序 |
| GET | `/api/review/<stem>` | 单图完整 analysis JSON + 已存复核结论 |
| POST | `/api/review` | 写入/更新一条复核结论 `{stem, decision, reviewer, note}` |
| GET | `/api/report/<stem>` | 调用 `tools/export_pdf.py` 导出 PDF 下载（失败回退 Markdown） |

`decision` 取值：`confirm_tampered`（确认篡改）/ `exclude_false_positive`（排除误报）/ `need_more_material`（需补充材料）。

## 6. 与「高风险必人工复核」口号的关系

- 算法只给风险分级，**不替人做最终定性**——这正是 `rule_engine.explain()` 里反复强调的「算法不是法律结论」。
- 本工作台把口号落地为流程：高风险图**必须**进入人工队列 → 法务/品牌方看证据、做判定、留痕 → 结论写回 `review_queue.json` 可审计。
- 三种结论覆盖真实业务动作：确认篡改（下线/追责）、排除误报（算法误报，正常投放）、需补充材料（暂不下结论，等待原图/C2PA 凭证）。

## 7. 硬约束遵守

- 仅**新增** `web/review_app.py` 与 `results/REVIEW_WORKBENCH.md`；运行期新建 `outputs/review_queue.json`。
- 未修改 `web/app.py`、`tools/*`、`demo/`、`README.md`、`data/` 或 `outputs/` 下任何已有文件。
- 未做 git commit/push，未重训模型。
