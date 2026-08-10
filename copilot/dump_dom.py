"""Dump the my-likes page STRUCTURE (no personal content) to fix selectors.

    python -m copilot.dump_dom            # writes dom_dump.html

Attaches to your running Chrome (same as run_calibration), navigates to my-likes,
and writes an attribute-only skeleton of the page: tag names, classes, ids, data-*
attributes, roles, and button labels. All text, names, bios, image URLs, alt/title
text, and profile aria-labels are stripped/redacted, so the file is page structure,
not people. It's what's needed to build correct CSS selectors.
"""

from __future__ import annotations

import argparse
import os

# Walk the DOM and emit an attribute-only skeleton. Personal content is dropped:
# text nodes removed; src/href/srcset redacted; background-image URLs redacted;
# alt/title redacted; aria-label redacted EXCEPT on buttons/links (control labels
# like "Like"/"Nope", which we need and which aren't personal).
_JS = r"""() => {
  const KEEP_ARIA = new Set(['button', 'a']);
  const testids = new Set();
  const buttonLabels = new Set();
  function walk(node) {
    if (node.nodeType !== 1) return '';
    const tag = node.tagName.toLowerCase();
    if (['script','style','svg','path','noscript','iframe'].includes(tag)) return '';
    const attrs = [];
    for (const a of node.attributes) {
      const n = a.name; let v = a.value;
      if (n === 'data-testid') testids.add(v);
      if (n === 'aria-label' && KEEP_ARIA.has(tag)) buttonLabels.add(v);
      if (n === 'src' || n === 'href' || n === 'srcset') v = 'REDACTED';
      else if (n === 'style') {
        if (v.includes('background-image')) v = 'background-image:REDACTED'; else continue;
      } else if (n === 'aria-label' && !KEEP_ARIA.has(tag)) v = 'REDACTED';
      else if (n === 'alt' || n === 'title') v = 'REDACTED';
      attrs.push(n + '="' + v + '"');
    }
    const kids = Array.from(node.childNodes).map(walk).join('');
    return '<' + tag + (attrs.length ? ' ' + attrs.join(' ') : '') + '>' + kids + '</' + tag + '>';
  }
  const main = document.querySelector('main') || document.body;
  const skeleton = walk(main);
  return { skeleton, testids: [...testids], buttonLabels: [...buttonLabels] };
}"""


def run(cdp_url: str, out_path: str, url: str, wait_ms: int = 3500) -> None:
    from .drivers.tinder_cdp import TinderDriver

    driver = TinderDriver(cdp_url=cdp_url, source="likes")
    driver.start()
    try:
        page = driver._page
        page.goto(url)
        page.wait_for_timeout(wait_ms)
        result = page.evaluate(_JS)
    finally:
        driver.stop()

    with open(out_path, "w") as f:
        f.write(f"<!-- url: {url} -->\n")
        f.write(f"<!-- data-testid values: {result['testids']} -->\n")
        f.write(f"<!-- button/link aria-labels: {result['buttonLabels']} -->\n")
        f.write(result["skeleton"])

    print(f"wrote {out_path} ({len(result['skeleton'])} chars)")
    print(f"data-testid values on page: {result['testids']}")
    print(f"button/link labels on page: {result['buttonLabels']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump my-likes page structure.")
    parser.add_argument("--cdp-url", default=os.environ.get("COPILOT_CDP_URL") or "http://127.0.0.1:9222")
    parser.add_argument("--url", default="https://tinder.com/app/my-likes")
    parser.add_argument("--out", default="dom_dump.html")
    args = parser.parse_args()
    run(args.cdp_url, args.out, args.url)


if __name__ == "__main__":
    main()
