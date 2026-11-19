# 新人守护师 · RAG 知识库索引

> 用于接入 BeautyProof 后端的检索增强生成知识库

---

## 知识库结构

```
beauty_proof_kb/
├── ad_detection_rules.md       ← 广告鉴别规则（核心规则引擎）
├── beauty_categories.md        ← 美妆品类知识（功效边界）
├── rag_corpus.jsonl            ← 爬取的真实小红书笔记（RAG 检索语料）
├── rag_corpus_comments.jsonl   ← 爬取的真实评论（社区黑话/真实反馈）
├── curated_phrase_patterns.jsonl ← 200 条整理的话术示例（非真实笔记）
├── data_summary.json           ← 数据集元信息
└── KB_README.md                ← 本文件
```

## 使用方式

### 1. RAG 检索
将 `rag_corpus.jsonl` 和 `rag_corpus_comments.jsonl` 向量化后存入向量数据库，用于用户提问时的语义检索。

### 2. 规则引擎
将 `ad_detection_rules.md` 中的规则提取为结构化数据，作为 Prompt 注入 LLM。

### 3. 品类知识
`beauty_categories.md` 作为参考知识，帮助 AI 理解产品合理功效边界。

---

## 数据说明

| 数据集 | 来源 | 内容量 | 采集时间 |
|--------|------|--------|---------|
| rag_corpus.jsonl | 小红书关键词搜索 | 60 条笔记 | 2026-11-19 |
| rag_corpus_comments.jsonl | 对应笔记的评论 | 695 条评论 | 2026-11-19 |
| curated_phrase_patterns.jsonl | 公开美妆内容趋势、监管指南的编辑归纳 | 200 条表达模式 | 2026-11-20 |

话术库覆盖情景开场、效果宣称、数字卖点、成分技术、适用人群、购买引导、个人体验、评论互动、图片证据和内容趋势，每类 20 条。条目是为检索与展示整理的改写示例，不是新增爬取内容、真实用户反馈、产品功效依据或广告判定标签。广告识别和功效核验仍需结合原文、合作信息和对应产品资料。

整理时参考了[国家药监部门《化妆品功效宣称评价规范》](https://yjj.scjgj.fujian.gov.cn/hzp/flfg/202107/t20210708_5642667.htm)、[市场监管总局《互联网广告管理办法》](https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/fgs/art/2023/art_d93a579afd45413e8576e4623fab348f.html)、[小红书化妆品虚假宣传规则](https://school.xiaohongshu.com/rule/detail/119727)及[美妆内容趋势报道](https://www.donews.com/article/detail/5099/55964.html)。法规只用于说明哪些说法需要对应依据；流行表达只用于补充语境。

### 关键词覆盖
- 美妆推荐 / 化妆品测评 / 口红推荐
- 护肤心得 / 粉底液测评
- 国货彩妆 / 国货之光 / 平价好物
- 花知晓 / 毛戈平等国货品牌
- 送礼好物 / 平价生日礼物

### 每条笔记包含字段
```json
{
  "note_id": "唯一ID",
  "type": "video 或 image",
  "title": "笔记标题",
  "desc": "正文内容",
  "nickname": "博主昵称(脱敏)",
  "liked_count": "点赞数",
  "collected_count": "收藏数",
  "comment_count": "评论数",
  "share_count": "分享数",
  "tag_list": "话题标签",
  "note_url": "原文链接",
  "source_keyword": "来源搜索词"
}
```

### 每条评论包含字段
```json
{
  "comment_id": "唯一ID",
  "note_id": "所属笔记ID",
  "content": "评论内容",
  "nickname": "评论者昵称",
  "like_count": "评论点赞",
  "sub_comment_count": "子评论数",
  "create_time": "评论时间戳"
}
```
