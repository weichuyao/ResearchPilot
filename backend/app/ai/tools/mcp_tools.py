"""MCP 工具接入（改造 #4，第一刀：arXiv）。

## 为什么只有这一个模块知道 MCP

进程外的工具提供方（arxiv-mcp-server，stdio 子进程）在这里被转成 LangChain
工具，再 append 进 react-assistant 的工具列表。设计取舍见
reference/transformation-04-mcp-arxiv-design.md：
research-workflow **不接**（它的检索是图上的确定性节点，接外部工具要重设
三态语义，不是顺手活）。

## 生命周期

server 子进程由 langchain-mcp-adapters 在 get_tools() 时经 stdio 拉起。
uvx 首次运行会下载包（几秒~几十秒），之后走缓存。

## 降级口径（看得见的降级）

ENABLED=false → 空列表，不打日志（那是刻意关闭，不是故障）。
server 拉不起来 / 包缺失 / 超时 → **WARNING 日志 + 空列表**，agent 照常工作。
本地检索永远不受 MCP 存亡影响 —— 这条是底线。
"""

from __future__ import annotations

import logging

from core.config import settings

logger = logging.getLogger(__name__)

# 只挂检索/元数据类工具。server 实际暴露 19 个（还有 download / read / watch /
# latex 等），不全要：
# 1. `list_papers` 与本地 research_tools.list_papers **重名** —— ToolNode 按
#    名字分发，重名直接撞车（实测抓到）；
# 2. download/read/latex 系列管理的是 MCP server 自己的存储 —— 本项目有
#    自己的入库链路（ingest.py），两套语料并存只会制造混乱；
# 3. 19 个工具 schema 全绑给模型，会稀释工具选择。
MCP_ARXIV_ALLOWLIST = (
    "search_papers", "get_abstract", "citation_graph",
    # 研究主题订阅：server 原生能力（自带存储与检查），挂上即可用
    "watch_topic", "check_alerts", "list_watches", "unwatch_topic",
)


async def _mount(command: list[str]) -> list | None:
    """尝试挂载；任何失败返回 None（不抛），由调用方决定降级路径。"""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {"arxiv": {"transport": "stdio", "command": command[0], "args": command[1:]}}
    )
    try:
        return await client.get_tools()
    except Exception as exc:
        logger.info("MCP: 挂载命令 %s 失败 —— %s: %s", command, type(exc).__name__, str(exc)[:120])
        return None


async def get_mcp_tools() -> list:
    """返回经 MCP 挂载的外部工具；失败时返回空列表（不抛异常）。"""
    if not settings.MCP_ARXIV_ENABLED:
        return []
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        command = settings.MCP_ARXIV_COMMAND.split()
        all_tools = await _mount(command)
        if all_tools is None and command[0] == "uvx":
            # uvx 默认联网校验包版本，pypi 不可达（代理没开/被墙）就整体失败。
            # 包已在 uv 缓存时 --offline 可以完全离线挂载 —— 自动降级重试。
            all_tools = await _mount(["uvx", "--offline"] + command[1:])
            if all_tools is not None:
                logger.info("MCP: 在线挂载失败，已用 uvx --offline 从缓存挂载")
        tools = [t for t in all_tools if t.name in MCP_ARXIV_ALLOWLIST]
        dropped = sorted(set(t.name for t in all_tools) - set(t.name for t in tools))
        logger.info(
            "MCP: arxiv server 挂载 %d/%d 个工具（%s）｜未挂：%s",
            len(tools), len(all_tools), ", ".join(t.name for t in tools), ", ".join(dropped),
        )
        return tools
    except Exception as exc:
        logger.warning(
            "MCP: arxiv 工具挂载失败，本 agent 将只有本地检索工具 —— %s: %s",
            type(exc).__name__, str(exc)[:200],
        )
        return []
