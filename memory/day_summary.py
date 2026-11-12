"""日总结：每天按段总结，一件事一个向量。总结只是索引，原话才是证据，要细节回去翻原话。"""
import struct
from datetime import datetime, timedelta

from . import llm
from .store import local

GAP = timedelta(hours=3)  # 隔 3 小时以上算新的一段
SEG_CHARS = 24000
TOP_K, MIN_SIM = 3, 0.62
READ_CAP = 12000

PROMPT = """下面是{user}和{assistant}某一天里的一段原始对话。写一份总结：
- 一件事一段，段与段之间空一行；
- 写清楚时间、发生了什么、说过的要紧的话和情绪；
- 只根据原文写，不要补充；
- 如果没有值得记的内容，只回一个字：无。
只写正文，不要 JSON，不要标题。"""


def segments(msgs):
    segs, size, prev = [], 0, None
    for m in msgs:
        t = datetime.fromisoformat(m["created_at"])
        if not segs or t - prev >= GAP or size + len(m["content"]) > SEG_CHARS:
            segs.append([])
            size = 0
        segs[-1].append(m)
        size += len(m["content"])
        prev = t
    return segs


def day_messages(db, conv, day):
    # ponytail: 全表扫一遍按本地日期筛，消息多了给 created_at 加本地日期列
    rows = db.execute("SELECT * FROM messages WHERE conv=? ORDER BY id", (conv,))
    return [m for m in rows if local(m["created_at"]).date().isoformat() == day]


def transcript(seg, names):
    return "\n".join(f'[{local(m["created_at"]):%H:%M}] {names[m["role"]]}：{m["content"]}' for m in seg)


def split_chunks(text, short=40, cap=500):
    """一段一个向量，召回时拿到的就是一件事。太短的并进下一段，太长的硬切。"""
    out, carry = [], ""
    for p in filter(None, map(str.strip, text.split("\n"))):
        p = f"{carry}\n{p}" if carry else p
        if len(p) < short:
            carry = p
            continue
        carry = ""
        out += [p[i : i + cap] for i in range(0, len(p), cap)]
    if carry:
        out.append(carry)
    return out


def pack(vec):
    return struct.pack(f"{len(vec)}f", *vec)


def unpack(blob):
    return struct.unpack(f"{len(blob) // 4}f", blob)


def summarize_day(db, conv, day, names, chat=llm.chat, embed=llm.embed):
    for i, seg in enumerate(segments(day_messages(db, conv, day))):
        text = chat(
            [
                {"role": "system", "content": PROMPT.format(**names)},
                # 不围起来的话，角色扮演的对话会被模型接着演下去，而不是总结
                {"role": "user", "content": f"<原始对话>\n{transcript(seg, names)}\n</原始对话>\n\n请总结以上原始对话，不要接着往下写。"},
            ],
            temperature=0.3,
        ).strip()
        if not text:
            continue  # 失败不覆盖旧的好总结
        # 记下这段原话的 id 范围：以后当天补进消息、切段规则变了，总结也还指着它当初读的那几句
        db.execute("INSERT OR REPLACE INTO day_summaries VALUES (?,?,?,?,?,?)",
                   (conv, day, i, text, seg[0]["id"], seg[-1]["id"]))
        db.execute("DELETE FROM chunks WHERE conv=? AND day=? AND seg=?", (conv, day, i))
        if text != "无":
            parts = split_chunks(text)
            for part, vec in zip(parts, embed(parts)):
                db.execute(
                    "INSERT INTO chunks (conv, day, seg, text, vec) VALUES (?,?,?,?,?)",
                    (conv, day, i, part, pack(vec)),
                )
        db.commit()


def search(db, query, embed=llm.embed, k=TOP_K, min_sim=MIN_SIM, exclude=()):
    """exclude：最近几轮已经给过的 chunk id，别反复塞同一件事。"""
    q = embed([query])[0]
    # ponytail: 纯 Python 全量点积，3000 条约 40ms；过两万条换 numpy
    scored = [
        {"id": r["id"], "day": r["day"], "seg": r["seg"], "text": r["text"],
         "sim": sum(a * b for a, b in zip(q, unpack(r["vec"])))}
        for r in db.execute("SELECT * FROM chunks")
        if r["id"] not in exclude
    ]
    scored.sort(key=lambda c: -c["sim"])
    for rank, c in enumerate(scored, 1):
        c["rank"] = rank
    return [c for c in scored[:k] if c["sim"] >= min_sim]


def read_day(db, conv, day, seg, names):
    """给模型的工具：按日期和段号翻回这份总结当初读的原话。"""
    r = db.execute("SELECT first_id, last_id FROM day_summaries WHERE conv=? AND day=? AND seg=?",
                   (conv, day, seg)).fetchone()
    if not r:
        return "这一段不存在。"
    msgs = db.execute("SELECT * FROM messages WHERE conv=? AND id BETWEEN ? AND ? ORDER BY id", (conv, *r))
    return transcript(msgs, names)[:READ_CAP]


def render(hits):
    if not hits:
        return ""
    lines = "\n".join(f'- {c["day"]} 第 {c["seg"]} 段：{c["text"]}' for c in hits)
    return f"<day_summary_recall>\n{lines}\n（这些是日总结，只是索引；要细节就用 read_day 翻原话，对不上以原话为准。）\n</day_summary_recall>"
