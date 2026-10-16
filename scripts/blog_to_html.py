"""Convert blog/*.md to HTML with inline styles, for blog editors that drop
Markdown and <style> blocks. Open the output in a browser and copy, or paste
the file's body into the editor's HTML mode.

Usage: python scripts/blog_to_html.py
"""
import re
import subprocess
from pathlib import Path

STYLES = {
    "h1": "font-size:28px;font-weight:700;line-height:1.4;margin:32px 0 16px;",
    "h2": "font-size:22px;font-weight:700;line-height:1.4;margin:36px 0 12px;padding-bottom:6px;border-bottom:1px solid #ddd;",
    "h3": "font-size:18px;font-weight:700;line-height:1.4;margin:28px 0 10px;",
    "p": "font-size:16px;line-height:1.8;margin:12px 0;",
    "ul": "font-size:16px;line-height:1.8;margin:12px 0;padding-left:24px;",
    "ol": "font-size:16px;line-height:1.8;margin:12px 0;padding-left:24px;",
    "li": "margin:4px 0;",
    "blockquote": "margin:16px 0;padding:12px 16px;border-left:4px solid #4a7bd0;background:#f3f6fb;color:#333;",
    "table": "border-collapse:collapse;width:100%;margin:16px 0;font-size:14px;line-height:1.6;",
    "th": "border:1px solid #ccc;background:#f0f0f0;padding:8px 10px;text-align:left;font-weight:700;",
    "td": "border:1px solid #ccc;padding:8px 10px;vertical-align:top;",
    "pre": "background:#f6f8fa;border:1px solid #e1e4e8;border-radius:6px;padding:14px 16px;overflow-x:auto;font-size:13px;line-height:1.5;margin:16px 0;",
    "hr": "border:none;border-top:1px solid #ddd;margin:32px 0;",
    "strong": "font-weight:700;",
}
INLINE_CODE = "background:#f0f2f4;border-radius:4px;padding:1px 5px;font-family:Consolas,Menlo,monospace;font-size:0.9em;"
PRE_CODE = "font-family:Consolas,Menlo,monospace;background:none;padding:0;"


def style(html: str) -> str:
    html = re.sub(r' id="[^"]*"', "", html)
    html = re.sub(r"<(pre)[^>]*>\s*<code[^>]*>", lambda m: f'<pre style="{STYLES["pre"]}"><code style="{PRE_CODE}">', html)
    html = re.sub(r"<code>", f'<code style="{INLINE_CODE}">', html)
    for tag, css in STYLES.items():
        if tag == "pre":
            continue
        html = re.sub(rf"<{tag}(\s[^>]*)?>", lambda m, t=tag, c=css: f'<{t} style="{c}">', html)
        if tag == "hr":
            html = html.replace(f'<hr style="{css}">', f'<hr style="{css}" />')
    return html


def main():
    root = Path(__file__).resolve().parents[1] / "blog"
    out = root / "html"
    out.mkdir(exist_ok=True)
    for md in sorted(root.glob("*.md")):
        body = subprocess.run(
            ["pandoc", str(md), "-f", "gfm", "-t", "html", "--no-highlight", "--wrap=none"],
            check=True, capture_output=True, text=True,
        ).stdout
        body = style(body).replace('href="week', 'href="week').replace('.md"', '.html"')
        title = re.search(r"^# (.+)$", md.read_text(), re.M).group(1)
        doc = (
            "<!doctype html>\n<html lang=\"ko\"><head><meta charset=\"utf-8\">"
            f"<title>{title}</title></head>\n"
            '<body style="max-width:820px;margin:40px auto;padding:0 16px;'
            'font-family:-apple-system,\'Apple SD Gothic Neo\',\'Malgun Gothic\',sans-serif;color:#222;">\n'
            f"{body}</body></html>\n"
        )
        (out / f"{md.stem}.html").write_text(doc)
        print(out / f"{md.stem}.html")


if __name__ == "__main__":
    main()
