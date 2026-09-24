"""Browser-level smoke test for the lightweight Research Workbench.

This test creates one question and one approved hypothesis. Run it only against
an isolated/test deployment and clean the fixture afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from pyppeteer import launch


async def click_text(page, text: str) -> None:
    predicate = "(text) => [...document.querySelectorAll('button')].some(x => x.innerText.replace(/\\s/g, '').includes(text.replace(/\\s/g, '')))"
    await page.waitForFunction(predicate, {"timeout": 30000}, text)
    clicked = await page.evaluate(
        "(text) => { const wanted = text.replace(/\\s/g, ''); const button = [...document.querySelectorAll('button')].find(x => x.innerText.replace(/\\s/g, '').includes(wanted)); if (!button) return false; button.click(); return true; }",
        text,
    )
    if not clicked:
        raise RuntimeError(f"button not found: {text}")


async def visible_modal(page):
    await page.waitForSelector(".ant-modal-content")
    for modal in reversed(await page.querySelectorAll(".ant-modal-content")):
        if await modal.boundingBox() is not None:
            return modal
    raise RuntimeError("no visible modal found")


async def submit_modal(page) -> None:
    modal = await visible_modal(page)
    button = await modal.querySelector(".ant-modal-footer button.ant-btn-primary")
    if button is None:
        raise RuntimeError("visible modal has no primary action")
    await button.click()


async def run(url: str, executable: str, screenshot: Path | None) -> dict:
    browser = await launch(
        executablePath=executable,
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    errors: list[str] = []
    try:
        page = await browser.newPage()
        await page.setViewport({"width": 1440, "height": 1000, "deviceScaleFactor": 1})
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(url, {"waitUntil": "networkidle2", "timeout": 60000})
        await page.waitForFunction("document.body.innerText.includes('科研状态工作台')")

        title = f"Browser Smoke Research Question {uuid4().hex[:8]}"
        await click_text(page, "新建研究问题")
        modal = await visible_modal(page)
        title_input = await modal.querySelector("input")
        await title_input.type(title)
        textareas = await modal.querySelectorAll("textarea")
        await textareas[0].type("Verify the Workbench against a real browser and API.")
        await submit_modal(page)
        await page.waitForFunction(
            "(title) => document.body.innerText.includes(title)",
            {"timeout": 30000},
            title,
        )

        await click_text(page, "提出假设")
        modal = await visible_modal(page)
        textareas = await modal.querySelectorAll("textarea")
        await textareas[0].type("Stable provenance improves traceability.")
        await textareas[1].type("Every accepted claim resolves to a source locator.")
        await submit_modal(page)
        await page.waitForFunction(
            "document.body.innerText.includes('REGISTER_HYPOTHESIS')",
            {"timeout": 30000},
        )
        await click_text(page, "批准")
        await page.waitForFunction(
            "document.body.innerText.includes('TESTABLE')",
            {"timeout": 30000},
        )
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot({"path": str(screenshot), "fullPage": True})
        if errors:
            raise RuntimeError(f"browser page errors: {errors}")
        return {
            "ok": True,
            "url": url,
            "question_created": True,
            "question_title": title,
            "hypothesis_registered": True,
            "screenshot": str(screenshot) if screenshot else None,
        }
    finally:
        await browser.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3000/research")
    parser.add_argument("--browser-executable", required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(asyncio.run(run(
            args.url, args.browser_executable, args.screenshot
        )), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
