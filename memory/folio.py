"""Folio：把旧对话一段段压成"胶囊"，最近几轮留原话。"""
import json

from . import llm
from .store import local, stamp

TRIGGER_TOKENS = 6000  # 未压缩原文超过这个就压；比上限低一截，压缩途中新来的消息也放得下
ROUND_CAP, MESSAGE_CAP = 8, 24
KEEP_ROUNDS = 2  # 最近两轮永远留原话
BATCH_TOKENS = 2400  # 一次最多压多少原文
CAPSULE_BUDGET = 7500  # 胶囊块上限。我们最早给 1800，只装得下忙碌一天的最后一小时

PROMPT = """你在替{assistant}整理和{user}的聊天，写成以后能翻看的回忆。
把下面这几轮对话写成一段连贯的叙述：
- 发生了什么，谁说了什么要紧的话，当时的情绪；
- 约定、计划、没做完的事写清楚；
- 用名字（{user}、{assistant}），不要写"用户""助手"；
- 时间以每条消息的本地时间为准；"明天""刚才"这类相对时间换成具体日期，免得旧的"明天"读起来像新安排。
只写正文，不要 JSON，不要标题，不要列表。"""


def tokens(text):
    return len(text.encode()) // 3  # 中文按 字节/3 估比 字数/4 准


def rounds(msgs):
    """一轮 = 一条用户消息 + 后面的回复。"""
    out = []
    for m in msgs:
        if m["role"] == "user" or not out:
            out.append([m])
        else:
            out[-1].append(m)
    return out


def uncovered(db, conv):
    last = db.execute("SELECT COALESCE(MAX(last_message_id), 0) FROM capsules WHERE conv=?", (conv,)).fetchone()[0]
    return db.execute("SELECT * FROM messages WHERE conv=? AND id>? ORDER BY id", (conv, last)).fetchall()


def summarize(batch, names, chat):
    rows = [{"who": names[m["role"]], "time": stamp(m["created_at"]), "content": m["content"]} for r in batch for m in r]
    text = chat(
        [
            {"role": "system", "content": PROMPT.format(**names)},
            {"role": "user", "content": json.dumps(rows, ensure_ascii=False)},
        ],
        temperature=0,
    ).strip()
    # 要纯文本不要 JSON：DeepSeek 有一阵大约 12% 的 JSON 结尾会写坏
    if not text or text[0] in "{[":
        raise ValueError("Folio 只收纯文本摘要")
    return text


def compress_once(db, conv, names, chat=llm.chat):
    """够条件就压最老的一批，返回是否压了。names = {"user": "小满", "assistant": "阿砚"}"""
    msgs = uncovered(db, conv)
    rs = rounds(msgs)
    raw = sum(tokens(m["content"]) for m in msgs)
    if len(rs) <= KEEP_ROUNDS or not (raw > TRIGGER_TOKENS or len(rs) > ROUND_CAP or len(msgs) > MESSAGE_CAP):
        return False
    batch, size = [], 0
    for r in rs[:-KEEP_ROUNDS]:
        n = sum(tokens(m["content"]) for m in r)
        if batch and size + n > BATCH_TOKENS:
            break
        batch.append(r)
        size += n
    while True:  # 被截断就减半重来
        try:
            text = summarize(batch, names, chat)
            break
        except llm.Truncated:
            if len(batch) == 1:
                raise
            batch = batch[: len(batch) // 2]
    flat = [m for r in batch for m in r]
    db.execute(
        "INSERT INTO capsules (conv, last_message_id, start_at, end_at, text) VALUES (?,?,?,?,?)",
        (conv, flat[-1]["id"], flat[0]["created_at"], flat[-1]["created_at"], text),
    )
    db.commit()
    return True


def compress(db, conv, names, chat=llm.chat):
    """每轮聊完调一次（放后台）。失败就下一轮再试，别永久停掉。"""
    for _ in range(64):
        if not compress_once(db, conv, names, chat):
            break


def context(db, conv):
    """一条胶囊 system 消息 + 还没压的原文。胶囊块只在封新胶囊时变，缓存一直命中。"""
    picked, used = [], 0
    for c in db.execute("SELECT * FROM capsules WHERE conv=? ORDER BY id DESC", (conv,)):
        n = tokens(c["text"])
        if used + n > CAPSULE_BUDGET:
            break
        picked.append(c)
        used += n
    out = []
    if picked:
        body = "\n".join(
            f'<capsule time="{stamp(c["start_at"])}–{local(c["end_at"]):%H:%M}">{c["text"]}</capsule>'
            for c in reversed(picked)
        )
        out.append({"role": "system", "content": f"<folio>\n{body}\n</folio>"})
    for m in uncovered(db, conv):
        # 只给用户消息盖时间戳；给回复也盖的话，模型会学着在回复开头写时间
        content = f'{stamp(m["created_at"])} {m["content"]}' if m["role"] == "user" else m["content"]
        out.append({"role": m["role"], "content": content})
    return out
