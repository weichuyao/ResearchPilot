"""批量导入：把一个文件夹里的文献逐个 POST 给上传接口。

为什么走接口而不是直接写 uploads/ 目录：paper 记录、后台索引、状态机
都长在 POST /documents 的链路上 —— 手动拷文件进来只会得到无人索引的孤儿。
本脚本就是「替你点上传按钮」。

用法（后端需在跑）：
    python scripts/batch_upload.py D:\\文献文件夹
    python scripts/batch_upload.py D:\\文献文件夹 --wait   # 传完等全部索引结束
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx

# 环境变量可能存在但为空 —— or 兜底，不能用 get 的默认值参数
BASE = os.environ.get("NEXT_PUBLIC_API_BASE_URL") or "http://127.0.0.1:8001"
SUPPORTED = {".pdf", ".docx", ".md", ".markdown", ".txt", ".text"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="存放文献的文件夹")
    parser.add_argument("--wait", action="store_true", help="上传后等待全部索引结束")
    args = parser.parse_args()

    files = sorted(
        f for f in os.listdir(args.folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED
        and os.path.isfile(os.path.join(args.folder, f))
    )
    if not files:
        raise SystemExit(f"{args.folder} 里没有可上传的文件（支持：{' '.join(sorted(SUPPORTED))}）")
    print(f"共 {len(files)} 个文件，逐个上传到 {BASE}/documents")

    ids: list[tuple[str, int]] = []
    failed = 0
    with httpx.Client(timeout=300, base_url=BASE) as c:
        for name in files:
            path = os.path.join(args.folder, name)
            with open(path, "rb") as fh:
                resp = c.post("/documents", files={"file": (name, fh)})
            if resp.status_code == 202:
                doc = resp.json()
                ids.append((name, doc["id"]))
                print(f"  [202] {name} -> id={doc['id']}")
            else:
                failed += 1
                detail = resp.json().get("detail", resp.text[:80]) if resp.headers.get("content-type", "").startswith("application/json") else resp.text[:80]
                print(f"  [{resp.status_code}] {name} —— {detail}")
            time.sleep(0.3)  # 给后台索引线程喘息，别把 embedding 排成饥荒

        if args.wait and ids:
            pending = dict(ids)
            print(f"\n等待 {len(pending)} 篇索引完成（每 5 秒轮询一次）...")
            while pending:
                for name, doc_id in list(pending.items()):
                    st = c.get(f"/documents/{doc_id}").json()
                    if st["status_text"] == "indexed":
                        print(f"  [完成] {name}（{st['chunk_count']} 块）")
                        del pending[name]
                    elif st["status_text"] == "failed":
                        print(f"  [失败] {name} —— {st.get('error')}")
                        del pending[name]
                if pending:
                    time.sleep(5)

    print(f"\n上传 {len(ids)} 成功 / {failed} 失败")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
