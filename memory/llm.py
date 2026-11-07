"""任何 OpenAI 兼容接口都行。环境变量：LLM_BASE_URL / LLM_API_KEY / LLM_MODEL，EMBED_ 同理。"""
import json
import math
import os
import time
import urllib.error
import urllib.request


class Truncated(RuntimeError):
    """输出被 max_tokens 截断了，半截摘要不能存。"""


def _post(prefix, path, body):
    req = urllib.request.Request(
        os.environ[f"{prefix}_BASE_URL"].rstrip("/") + path,
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {os.environ[f'{prefix}_API_KEY']}", "Content-Type": "application/json"},
    )
    for wait in (2, 8, None):  # 429 / 5xx / 断网：等 2 秒、8 秒各重试一次
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if wait is None or (e.code != 429 and e.code < 500):
                raise
        except urllib.error.URLError:
            if wait is None:
                raise
        time.sleep(wait)


def chat(messages, **kw):
    out = _post("LLM", "/chat/completions", {"model": os.environ["LLM_MODEL"], "messages": messages, **kw})
    choice = out["choices"][0]
    if choice.get("finish_reason") == "length":
        raise Truncated
    return choice["message"]["content"] or ""


def embed(texts):
    out = _post("EMBED", "/embeddings", {"model": os.environ["EMBED_MODEL"], "input": texts})
    vecs = [d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"])]
    return [[x / (math.sqrt(sum(y * y for y in v)) or 1) for x in v] for v in vecs]
