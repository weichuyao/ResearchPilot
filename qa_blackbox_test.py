# -*- coding: utf-8 -*-
"""ResearchPilot 后端黑盒 QA 脚本。

用法:
    python qa_blackbox_test.py phase1    # 只读 + 会话错误路径
    python qa_blackbox_test.py phase2    # 文档上传全链路 + 错误路径
    python qa_blackbox_test.py phase3    # chat invoke / stream
    python qa_blackbox_test.py cleanup   # 清理本脚本创建的所有测试产物
    python qa_blackbox_test.py verify    # 核实数字与基线一致

只在 127.0.0.1:8002 上跑；绝不改动/删除基线已存在的资源。
状态记录在 .qa_state.json。
"""
import io
import json
import os
import sys
import time
import uuid

import httpx

BASE = "http://127.0.0.1:8002"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".qa_state.json")
NOEXIST = "qa-noexist-%s" % uuid.uuid4().hex[:12]

results = []  # (name, ok, detail)


def rec(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("PASS  " if ok else "FAIL  ") + name + ("  | " + detail if detail else ""))


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_state(st):
    with open(STATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(st, fh, ensure_ascii=False, indent=2)


def client():
    # trust_env=False: 完全无视系统代理，直连本机
    return httpx.Client(base_url=BASE, timeout=180, trust_env=False)


def make_pdf(text):
    """手工构造一个最小合法单页 PDF（正确 xref 偏移）。"""
    buf = io.BytesIO()

    def w(b):
        buf.write(b if isinstance(b, bytes) else b.encode("latin-1"))

    def obj(n, body):
        offsets[n] = buf.tell()
        w("%d 0 obj\n%s\nendobj\n" % (n, body))

    offsets = {}
    w("%PDF-1.4\n")
    stream = "BT /F1 14 Tf 72 720 Td (%s) Tj ET" % text
    obj(1, "<< /Type /Catalog /Pages 2 0 R >>")
    obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    obj(3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
           "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")
    obj(4, "<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    obj(5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    xref_pos = buf.tell()
    w("xref\n0 6\n0000000000 65535 f \n")
    for i in range(1, 6):
        w("%010d 00000 n \n" % offsets[i])
    w("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref_pos)
    return buf.getvalue()


def status_line(r):
    try:
        return "%s %s" % (r.status_code, json.dumps(r.json(), ensure_ascii=False)[:300])
    except Exception:
        return "%s %s" % (r.status_code, r.text[:200].replace("\n", " "))


def expect(name, r, code, cond=None):
    ok = r.status_code == code and (cond is None or cond(r))
    rec(name, ok, "expect %s got %s" % (code, status_line(r)))


# ---------------------------------------------------------------- phase 1
def phase1():
    st = load_state()
    with client() as c:
        # 基线
        r = c.get("/health")
        expect("GET /health", r, 200,
               lambda x: x.json()["app"] and "papers" in x.json()["index"])
        st["baseline"] = {
            "papers": r.json()["index"]["papers"],
            "chunks": r.json()["index"]["chunks"],
            "by_status": r.json()["index"]["by_status"],
        }
        save_state(st)

        r = c.get("/agents")
        expect("GET /agents", r, 200,
               lambda x: isinstance(x.json(), list) and len(x.json()) > 0
               and all(set(a) >= {"key", "description"} for a in x.json()))
        st["agents"] = [a["key"] for a in r.json()]

        r = c.get("/conversations")
        expect("GET /conversations", r, 200,
               lambda x: isinstance(x.json(), list)
               and all(set(it) >= {"thread_id", "title", "agent_id"} for it in x.json()))
        st["baseline_convs"] = [it["thread_id"] for it in r.json()]
        st["baseline_conv_count"] = len(r.json())
        save_state(st)

        # 对真实会话只读
        if st["baseline_convs"]:
            tid = st["baseline_convs"][0]
            r = c.get("/conversations/%s/messages" % tid)
            ok = r.status_code == 200 and isinstance(r.json(), list)
            rec("GET 真实会话 messages（只读）", ok,
                "%s, %d 条消息" % (r.status_code, len(r.json()) if ok else -1))

        r = c.get("/documents", params={"limit": 5})
        expect("GET /documents?limit=5", r, 200,
               lambda x: x.json()["total"] >= len(x.json()["items"])
               and len(x.json()["items"]) <= 5
               and all(set(d) >= {"id", "title", "status", "status_text", "chunk_count"}
                       for d in x.json()["items"]))
        st["baseline_doc_total"] = r.json()["total"]
        save_state(st)

        r = c.get("/documents/formats")
        expect("GET /documents/formats", r, 200,
               lambda x: ".pdf" in x.json()["formats"] and "note" in x.json())

        # 错误路径
        expect("GET 不存在会话 messages → 404",
               c.get("/conversations/%s/messages" % NOEXIST), 404)
        expect("PUT 不存在会话 → 404",
               c.put("/conversations/%s" % NOEXIST, json={"title": "x"}), 404)
        expect("PUT title 空串 → 422",
               c.put("/conversations/%s" % NOEXIST, json={"title": ""}), 422)
        expect("PUT title 101 字 → 422",
               c.put("/conversations/%s" % NOEXIST, json={"title": "a" * 101}), 422)
        expect("DELETE 不存在会话 → 404",
               c.delete("/conversations/%s" % NOEXIST), 404)


# ---------------------------------------------------------------- phase 2
def poll_indexed(c, doc_id, timeout=300):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        r = c.get("/documents/%s" % doc_id)
        if r.status_code != 200:
            return None, r
        last = r.json()
        if last["status"] in (1, 2):  # 1=indexed 2=failed
            return last, r
        time.sleep(2)
    return None, None


def phase2():
    st = load_state()
    made = st.setdefault("doc_ids", [])
    with client() as c:
        base_total = c.get("/documents", params={"limit": 1}).json()["total"]

        # -- 错误路径先做（不产生资源）
        r = c.post("/documents",
                   files={"file": ("fake.pdf", b"this is plain text, not a pdf", "application/pdf")},
                   data={"title": ""})
        expect("POST /documents 假 .pdf（txt 内容）→ 415", r, 415)

        r = c.post("/documents",
                   files={"file": ("evil.exe", b"MZ\x90\x00binary", "application/octet-stream")})
        expect("POST /documents 不支持的扩展名 .exe → 415", r, 415)

        expect("GET /documents/999999 → 404", c.get("/documents/999999"), 404)
        expect("PUT /documents/1 → 405", c.put("/documents/1"), 405)

        # -- 全链路: 两个不同内容的 PDF
        ids = []
        for i, (name, text) in enumerate([
            ("qa-blackbox-alpha.pdf", "QA blackbox alpha document about retrieval testing."),
            ("qa-blackbox-beta.pdf", "QA blackbox beta document about embedding pipelines."),
        ]):
            pdf = make_pdf(text)
            # 本地先自检 PDF 可被解析（pypdf 与后端同一依赖）
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf))
            assert reader.pages[0].extract_text().strip(), "生成的 PDF 无法解析"

            r = c.post("/documents",
                       files={"file": (name, pdf, "application/pdf")},
                       data={"title": "QA Blackbox %s" % ["Alpha", "Beta"][i], "collection": "reid-papers"})
            expect("POST /documents %s → 202" % name, r, 202,
                   lambda x: "poll" in x.json() and x.json()["status_text"] == "pending")
            if r.status_code != 202:
                return
            doc = r.json()
            ids.append(doc["id"])
            made.append(doc["id"])
            save_state(st)

            final, _ = poll_indexed(c, doc["id"])
            rec("轮询 %s → indexed" % name, final is not None and final["status"] == 1,
                json.dumps(final, ensure_ascii=False)[:200] if final else "超时/丢失")
            if final and final["status"] == 2 and final["chunk_count"] <= 0:
                rec("chunk_count > 0", False, str(final["chunk_count"]))

        r = c.get("/documents", params={"limit": 50})
        got_ids = {d["id"] for d in r.json()["items"]}
        rec("GET /documents 包含新上传且 total=base+2",
            all(i in got_ids for i in ids) and r.json()["total"] == base_total + 2,
            "total=%s (base=%s)" % (r.json()["total"], base_total))

        # -- reindex（对第一篇）
        r = c.post("/documents/%s/reindex" % ids[0])
        expect("POST /documents/%s/reindex → 202" % ids[0], r, 202)
        if r.status_code == 202:
            final, _ = poll_indexed(c, ids[0])
            rec("reindex 轮询 → indexed", final is not None and final["status"] == 1,
                json.dumps(final, ensure_ascii=False)[:200] if final else "超时")

        # -- reindex 不存在的文档
        expect("POST /documents/999999/reindex → 404",
               c.post("/documents/999999/reindex"), 404)

        # -- 删除
        for i in ids:
            r = c.delete("/documents/%s" % i)
            expect("DELETE /documents/%s → 204" % i, r, 204)
        for i in ids:
            expect("GET 已删文档 %s → 404" % i, c.get("/documents/%s" % i), 404)

        r = c.get("/documents", params={"limit": 1})
        rec("删除后 total 恢复基线", r.json()["total"] == base_total,
            "total=%s base=%s" % (r.json()["total"], base_total))

        st["doc_ids"] = [d for d in made if d not in ids]
        save_state(st)


# ---------------------------------------------------------------- phase 3
def phase3():
    st = load_state()
    convs = st.setdefault("conv_ids", [])
    with client() as c:
        # agent 不存在
        r = c.post("/chat/invoke", json={"message": "hi", "agent_id": "nonexistent"})
        ok = r.status_code == 404
        detail_ok = False
        try:
            detail_ok = "detail" in r.json()
        except Exception:
            pass
        rec("POST /chat/invoke agent_id=nonexistent → 404+JSON detail",
            ok and detail_ok, status_line(r))

        # stream: agent 不存在 —— 期望流开始前 404（非 SSE）
        try:
            with c.stream("POST", "/chat/stream",
                          json={"message": "hi", "agent_id": "nonexistent"}) as r:
                body = ""
                for chunk in r.iter_text():
                    body += chunk
                    if len(body) > 500:
                        break
            rec("POST /chat/stream agent_id=nonexistent → 流前 404",
                r.status_code == 404, "实际 status=%s body=%r" % (r.status_code, body[:120]))
        except httpx.HTTPError as e:
            rec("POST /chat/stream agent_id=nonexistent → 流前 404", False,
                "连接异常: %s" % type(e).__name__)

        # stream 正常路径（新 UUID thread）
        tid_stream = str(uuid.uuid4())
        events = []
        err = None
        try:
            with c.stream("POST", "/chat/stream",
                          json={"message": "say hi briefly", "thread_id": tid_stream,
                                "agent_id": "research-workflow"}) as r:
                rec("POST /chat/stream → 200 text/event-stream", r.status_code == 200,
                    "%s %s" % (r.status_code, r.headers.get("content-type", "")))
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        events.append(line[5:].strip())
                        try:
                            if json.loads(line[5:].strip()).get("type") == "end":
                                break
                        except Exception:
                            pass
                    if len(events) > 500:
                        break
        except httpx.HTTPError as e:
            err = e
        types = []
        for ev in events:
            try:
                types.append(json.loads(ev).get("type"))
            except Exception:
                types.append("?")
        rec("chat/stream 有 data: 事件且以 type=end 结束",
            err is None and len(events) > 0 and types[-1] == "end",
            "events=%d types(前5)=%s" % (len(events), types[:5]))
        convs.append(tid_stream)
        save_state(st)

        # invoke 默认 agent（省略 agent_id）
        r = c.post("/chat/invoke", json={"message": "reply with exactly: ok",
                                         "thread_id": tid_stream})
        ok = r.status_code == 200
        body = r.json() if ok else {}
        rec("POST /chat/invoke 复用 thread → 200", ok, status_line(r))
        if ok:
            rec("invoke 返回 ChatMessage 结构",
                body.get("type") == "ai" and isinstance(body.get("content"), str)
                and body.get("run_id"),
                "content=%r" % body.get("content", "")[:60])

        # invoke 默认 agent，全新 thread，省略 agent_id
        tid_default = str(uuid.uuid4())
        r = c.post("/chat/invoke", json={"message": "say hi", "thread_id": tid_default})
        expect("POST /chat/invoke 省略 agent_id（默认 research-workflow）→ 200",
               r, 200, lambda x: x.json().get("type") == "ai")
        convs.append(tid_default)
        save_state(st)

        # 会话索引应出现这两个 thread
        r = c.get("/conversations")
        tids = {it["thread_id"] for it in r.json()}
        rec("新会话出现在 /conversations",
            tid_stream in tids and tid_default in tids, "")

        # 重命名（只对自己的会话）
        r = c.put("/conversations/%s" % tid_default, json={"title": "QA rename test 01"})
        expect("PUT 重命名自己的会话 → 200", r, 200,
               lambda x: x.json().get("title") == "QA rename test 01")
        r = c.get("/conversations")
        titles = {it["thread_id"]: it["title"] for it in r.json()}
        rec("重命名后列表 title 生效", titles.get(tid_default) == "QA rename test 01", "")

        # 读自己的会话消息
        r = c.get("/conversations/%s/messages" % tid_default)
        rec("GET 自己会话 messages → 200 非空", r.status_code == 200 and len(r.json()) >= 2,
            "%s, %d 条" % (r.status_code, len(r.json()) if r.status_code == 200 else -1))


# ---------------------------------------------------------------- cleanup
def cleanup():
    st = load_state()
    with client() as c:
        for doc_id in st.get("doc_ids", []):
            r = c.get("/documents/%s" % doc_id)
            if r.status_code == 200 and r.json()["status"] in (0, 1):
                print("doc %s 仍在处理中，等待..." % doc_id)
                poll_indexed(c, doc_id, timeout=300)
            r = c.delete("/documents/%s" % doc_id)
            print("DELETE /documents/%s -> %s" % (doc_id, r.status_code))
            r2 = c.get("/documents/%s" % doc_id)
            print("  verify GET -> %s %s" % (r2.status_code, "OK" if r2.status_code == 404 else "STILL EXISTS"))
        for tid in st.get("conv_ids", []):
            r = c.delete("/conversations/%s" % tid)
            print("DELETE /conversations/%s -> %s" % (tid, r.status_code))
            r2 = c.get("/conversations/%s/messages" % tid)
            print("  verify GET messages -> %s %s" % (r2.status_code, "OK" if r2.status_code == 404 else "STILL EXISTS"))
        st["doc_ids"] = []
        st["conv_ids"] = []
        save_state(st)


def verify():
    st = load_state()
    with client() as c:
        h = c.get("/health").json()
        d = c.get("/documents", params={"limit": 1}).json()
        convs = c.get("/conversations").json()
        b = st.get("baseline", {})
        bconv = st.get("baseline_convs", [])
        print("health.index.papers  = %s (基线 %s) %s" % (
            h["index"]["papers"], b.get("papers"), "OK" if h["index"]["papers"] == b.get("papers") else "MISMATCH"))
        print("health.index.chunks = %s (基线 %s) %s" % (
            h["index"]["chunks"], b.get("chunks"), "OK" if h["index"]["chunks"] == b.get("chunks") else "MISMATCH"))
        print("documents.total     = %s (基线 %s) %s" % (
            d["total"], st.get("baseline_doc_total"), "OK" if d["total"] == st.get("baseline_doc_total") else "MISMATCH"))
        now_tids = {it["thread_id"] for it in convs}
        leftover = now_tids - set(bconv)
        print("conversations       = %d (基线 %d)  新增残留: %s %s" % (
            len(convs), st.get("baseline_conv_count", -1), sorted(leftover),
            "OK" if not leftover and len(convs) == st.get("baseline_conv_count") else "MISMATCH"))
        print("by_status           = %s (基线 %s)" % (h["index"]["by_status"], b.get("by_status")))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "phase1"
    {"phase1": phase1, "phase2": phase2, "phase3": phase3,
     "cleanup": cleanup, "verify": verify}[cmd]()
    if results:
        fails = [r for r in results if not r[1]]
        print("\n=== 小计: %d 项, %d 失败 ===" % (len(results), len(fails)))
