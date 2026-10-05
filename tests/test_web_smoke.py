# -*- coding: utf-8 -*-
"""Web 三服务冒烟测试（pytest + Flask test_client，不需要起服务器）

为什么必须有这组测试：
  ① web/app.py、web/review_app.py、web/content_app.py 是评审实际打开的演示入口，
     但此前**自动化覆盖率为 0** —— 139 项测试全在 tools/ 侧，Web 层裸奔。
  ② 2026-10-05 安全加固新增了「stem 白名单」「job 404」「JOBS 上限」三处行为，
     没有测试就会被下一次改动悄悄改回去。这组测试就是把它们钉死。

设计原则（沿用本项目既有纪律）：
  · 只测 HTTP 契约（状态码 / 关键字段），不碰内部实现，避免测试变脆；
  · 依赖外部产物的用 pytest.skip 跳过并说明原因，不让「没跑」伪装成「通过」；
  · 纯加文件，不修改任何生产代码。
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "web"))


@pytest.fixture(scope="module")
def client_app():
    """web/app.py 的测试客户端。导入失败要显式报错，不能静默跳过。"""
    import app as A
    A.app.config.update(TESTING=True)
    return A.app.test_client()


@pytest.fixture(scope="module")
def client_review():
    """web/review_app.py 的测试客户端。"""
    import review_app as R
    R.app.config.update(TESTING=True)
    return R.app.test_client()


@pytest.fixture(scope="module")
def client_content():
    """web/content_app.py 的测试客户端。"""
    import content_app as C
    C.app.config.update(TESTING=True)
    return C.app.test_client()


# ── 正常路径 ────────────────────────────────────────────────────────────────
def test_样本列表返回200且为数组(client_app):
    r = client_app.get("/api/samples")
    assert r.status_code == 200
    body = r.get_json()
    assert isinstance(body.get("samples"), list)


def test_合法标识能取到完整结果(client_app):
    """至少有一个已生成的分析结果时，取回它并核对关键字段。"""
    samples = client_app.get("/api/samples").get_json()["samples"]
    if not samples:
        pytest.skip("outputs/ 下暂无 analysis_*.json（先跑 tools/batch_run.py）")
    stem = samples[0]["stem"]
    r = client_app.get(f"/api/sample/{stem}")
    assert r.status_code == 200
    data = r.get_json()
    assert "verdict" in data and "tool_meta" in data
    assert data["verdict"].get("risk_level")


def test_空文件上传返回400(client_app):
    assert client_app.post("/api/analyze", data={}).status_code == 400


def test_复核队列可读(client_review):
    assert client_review.get("/api/queue").status_code == 200


def test_内容核验首页可渲染(client_content):
    assert client_content.get("/").status_code == 200


# ── 安全回归：以下每条都对应 2026-10-05 加固时修掉的一个真实问题 ──────────────
@pytest.mark.parametrize("bad", [
    "..%2f..%2f..%2fetc%2fpasswd",
    "..",
    "a/b",
    "x" * 200,
])
def test_非法标识一律404而非读文件(client_app, bad):
    """stem 白名单：越权读文件必须 404，不能返回任何文件内容。"""
    assert client_app.get(f"/api/sample/{bad}").status_code == 404


def test_非法标识在复核服务同样被拒(client_review):
    assert client_review.get("/api/review/..%2f..%2f..%2fetc%2fpasswd").status_code == 404
    assert client_review.get("/api/report/..%2f..%2f..%2fetc%2fpasswd").status_code == 404


def test_查询不存在的任务返回404而非500(client_app):
    """回归测试：修复前这里是 JOBS[job_id] 抛 KeyError → 500。"""
    r = client_app.get("/api/job/deadbeefdeadbeef")
    assert r.status_code == 404
    assert r.get_json()["status"] == "not_found"


def test_任务表有上限配置(client_app):
    """JOBS 必须有容量上限，否则长时间演示会把内存吃满。"""
    import app as A
    assert 0 < A.MAX_JOBS <= 1024
