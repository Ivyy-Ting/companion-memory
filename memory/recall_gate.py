"""召回门槛：检索后端捞上来的候选，哪些真该给模型看。

检索只负责"找得到"，"该不该想起来"交给一个便宜模型来挑。
相似度阈值、字面重合、重要度门槛我们都试过，在真实数据上都分不开相关和不相关（数据见 README）。
"""
import re

from . import llm

MAX_CANDIDATES, BODY_CHARS, MAX_ACCEPT = 12, 200, 5

PROMPT = """你在帮{assistant}决定：回{user}这句话之前，要从候选里想起哪几条旧记忆。标准是：一个用心的伴侣此刻会不会自然想到这件事。
该选：
- 偏好/习惯/基础信息（吃什么、过敏、在读的书、{user}的计划）：正在聊相关的日常，就选。
- 这句话（结合{assistant}上一句）在说同一个人、同一件事、同一条还在继续的线，或者正在重演那条记忆里的模式。
不该选：
- 只是撞了一个词、情绪相近、泛泛的同类话题。
- 某一次具体的旧场景：除非这句话明确提到那一次。
- 沉重的旧事（家庭、创伤、吵架）：除非现在正在经历或谈论同一件事。
宁缺毋滥，经常一条都不选。只输出编号数组，例如 [] 或 [2,5]。"""

RULES = """以下是可能相关的旧记忆，不是这一轮说的话。
1. 只用和当前话题真相关的，不相关就当没看见。
2. 记忆里没写的细节不要补；要细节就用 read_day 翻出处的原话，翻不到就说记不清。
3. 记忆和原话对不上时，以原话为准。
4. 不要提"记忆库""召回"，像自然想起来那样说。"""


def pool(candidates, recent_ids=()):
    """送去挑的候选：按检索排名取前 12 条，去掉最近给过的、归档的、重复的。每条都写上 fate。"""
    seen, out = set(), []
    for c in sorted(candidates, key=lambda c: c["rank"]):
        if c["id"] in recent_ids:
            c["fate"] = "RECENTLY_SHOWN"
        elif c.get("archived"):
            c["fate"] = "ARCHIVED"  # 归档的记忆在日常聊天里不复活
        elif c["id"] in seen or not c["text"].strip():
            c["fate"] = "DUPLICATE"
        elif len(out) >= MAX_CANDIDATES:
            c["fate"] = "OVER_LIMIT"
        else:
            seen.add(c["id"])
            out.append(c)
    return out


def parse_picks(text, n):
    """'[2, 5]' → [2, 5]（1 起）。没有数组就返回 None，和"一条都不选"的 [] 分开。"""
    m = re.search(r"\[[\d,\s]*\]", text or "")
    if not m:
        return None
    picks = []
    for x in map(int, re.findall(r"\d+", m.group())):
        if 1 <= x <= n and x not in picks:
            picks.append(x)
    return picks


def pick(query, candidates, names, prev="", recent_ids=(), chat=llm.chat):
    """candidates: [{id, text, rank, archived?, day?, seg?}]，rank 是检索排名（1 起）。
    prev 是助手上一句：她说"那个呢"时，光看这一句不知道指什么。
    返回 (挑中的, ok)。挑的模型没回话时 ok=False：一条都不给，告诉模型"查不了"。
    不退回阈值规则——阈值规则放进来的大多是错的，宁可这一轮不想起。"""
    cands = pool(candidates, recent_ids)
    if not cands:
        return [], True
    listing = "\n".join(f'{i}. {c["text"][:BODY_CHARS]}' for i, c in enumerate(cands, 1))
    ask = (f'{names["assistant"]}上一句（结尾）：{prev[-120:]}\n' if prev else "") + (
        f'{names["user"]}现在说：{query[:400]}\n\n候选记忆：\n{listing}'
    )
    try:
        reply = chat(
            [{"role": "system", "content": PROMPT.format(**names)}, {"role": "user", "content": ask}],
            max_tokens=60,
            temperature=0,
        )
        picks = parse_picks(reply, len(cands))
    except Exception:  # 挑的模型挂了不能拖垮这一轮回复
        picks = None
    if picks is None:
        for c in cands:
            c["fate"] = "PICKER_UNAVAILABLE"
        return [], False
    for i, c in enumerate(cands, 1):
        c["fate"] = "PICKED" if i in picks else "NOT_PICKED"
    return [cands[i - 1] for i in picks[:MAX_ACCEPT]], True


def render(accepted, unavailable=False):
    """"没找到"和"查不了"要分开告诉模型，不然它会把服务挂了当成没这回事。"""
    if unavailable:
        return '<recall status="unavailable">记忆暂时查不到。别假装记得，也别说没发生过。</recall>'
    if not accepted:
        return '<recall status="empty">没有找到相关的旧记忆。</recall>'
    items = "\n".join(f'<memory id="{c["id"]}"{_source(c)}>{c["text"]}</memory>' for c in accepted)
    return f'<recall status="ok">\n{RULES}\n{items}\n</recall>'


def _source(c):
    """记忆指回原话：有出处才能比对。"""
    return f' from="{c["day"]} 第 {c["seg"]} 段"' if "day" in c else ""
