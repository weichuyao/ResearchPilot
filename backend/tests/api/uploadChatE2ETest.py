"""改造 #5 的端到端验收：上传 → 索引 → **用聊天问它** → 删除。

设计文档第七节说的验收点是这一条：

    上传一篇新的 PDF → 拿到 202
    轮询状态 → 已索引
    **用聊天问它里面的内容 → 能答出来并带引用**   ← 前面全是手段，这一条才是目的

和 uploadApiTest.py 的区别：那个测的是接口本身（校验、状态、检索层、删除），
这个测的是**整条链走通之后 agent 能不能用上它** —— 中间隔着 agent 的查询改写、
检索管线、提示词约束，任何一环没接上都会在这里露馅。

    python tests/api/uploadChatE2ETest.py
"""

from __future__ import annotations

import os
import sys
import time

# 一个语料里绝对不存在的术语 + 一个编造的数值。
# 这样"答对了"只可能来自本次上传，不可能来自模型的知识或已有语料。
MARKER = "zephyr-quantum-flux"
FACT_VALUE = "42.7"

BODY = """# Sable-9 标定手册（ResearchPilot 上传验收用）

## 标定常数

Sable-9 传感器的标定常数是 %s。这个数值由 %s 方法在 2024 年的标定流程中确定。

## 使用说明

标定之前需要预热 15 分钟。%s 方法对温度漂移不敏感，所以不需要恒温环境。
""" % (FACT_VALUE, MARKER, MARKER)


def _bootstrap() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(here))
    os.chdir(backend_root)
    sys.path.insert(0, os.path.join(backend_root, "app"))
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    return backend_root


def main() -> int:
    _bootstrap()
    import httpx

    base = "http://127.0.0.1:8001"
    failures: list[str] = []
    paper_id = None
    source_file = None

    with httpx.Client(base_url=base, timeout=180.0) as client:
        before = client.get("/health").json()["index"]
        print("[基线] papers=%s chunks=%s" % (before["papers"], before["chunks"]))

        # 1) 上传
        files = {"file": ("sable9-manual.md", BODY.encode("utf-8"), "text/markdown")}
        data = {"title": "Sable-9 标定手册", "collection": "reid-papers"}
        r = client.post("/documents", files=files, data=data)
        print("[上传] HTTP %s  %s" % (r.status_code, r.json()))
        if r.status_code != 202:
            return report(["上传没有返回 202：%s" % r.text[:200]])
        paper_id = r.json()["id"]

        # 2) 轮询到索引完成
        deadline = time.time() + 120
        state = None
        while time.time() < deadline:
            state = client.get("/documents/%s" % paper_id).json()
            if state["status_text"] in ("indexed", "failed"):
                break
            time.sleep(0.5)
        print("[状态] %s" % state)
        if not state or state["status_text"] != "indexed":
            cleanup(client, paper_id, None)
            return report(["索引没有完成：%s" % state])
        source_file = state["source_file"]

        # 3) 真正的验收：用聊天问它
        question = "Sable-9 传感器的标定常数是多少？用的是哪种标定方法？"
        print("[提问] %s" % question)
        t0 = time.time()
        r = client.post("/chat/invoke", json={"message": question, "agent_id": "research-workflow"})
        dt = time.time() - t0
        if r.status_code != 200:
            cleanup(client, paper_id, source_file)
            return report(["聊天接口返回 %s：%s" % (r.status_code, r.text[:200])])
        answer = r.json()["content"]
        print("[回答] %.1fs" % dt)
        print("       %s" % " ".join(answer.split())[:500])
        print()

        if FACT_VALUE not in answer:
            failures.append("回答里没有出现上传文档里的数值 %s" % FACT_VALUE)
        if MARKER.replace("-", " ") not in answer and MARKER not in answer:
            failures.append("回答里没有提到上传文档里的独特术语")
        if "Sable-9" not in answer and "sable9-manual" not in answer:
            failures.append("回答里没有引用上传的文档")

        # 4) 清理
        client.delete("/documents/%s" % paper_id)
        paper_id = None

        after = client.get("/health").json()["index"]
        print("[清理后] papers=%s chunks=%s（基线 %s / %s）"
              % (after["papers"], after["chunks"], before["papers"], before["chunks"]))
        if after["papers"] != before["papers"] or after["chunks"] != before["chunks"]:
            failures.append("清理后没回到基线")

    if paper_id:
        pass
    return report(failures)


def cleanup(client, paper_id, source_file) -> None:
    try:
        client.delete("/documents/%s" % paper_id)
        print("   清理：已删除 id=%s" % paper_id)
    except Exception as exc:
        print("   清理失败：%s" % exc)


def report(failures: list[str]) -> int:
    print()
    if failures:
        print("FAILED (%d):" % len(failures))
        for item in failures:
            print("   - %s" % item)
        return 1
    print("OK 端到端验收通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
