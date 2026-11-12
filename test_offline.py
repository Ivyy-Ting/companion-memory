"""不联网的自检：python3 test_offline.py"""
from datetime import datetime, timedelta

from memory import day_summary, folio, layout, llm, recall_gate, store

NAMES = {"user": "小满", "assistant": "阿砚"}
T0 = datetime(2026, 11, 1, 9, 0, tzinfo=store.TZ)


def seed(db, n_rounds, start=T0, step=timedelta(minutes=5)):
    t = start
    for i in range(n_rounds):
        store.add_message(db, "c", "user", f"第{i}句", t)
        store.add_message(db, "c", "assistant", f"回{i}", t + timedelta(minutes=1))
        t += step
    return t


def test_folio():
    db = store.connect()
    seed(db, 10)
    calls = []

    def fake(messages, **kw):
        calls.append(messages)
        return "两个人聊了几句。"

    folio.compress(db, "c", NAMES, fake)
    assert len(calls) == 1
    left = folio.rounds(folio.uncovered(db, "c"))
    assert [r[0]["content"] for r in left] == ["第8句", "第9句"]  # 留最近两轮
    ctx = folio.context(db, "c")
    assert ctx[0]["role"] == "system" and "两个人聊了几句" in ctx[0]["content"]
    assert ctx[1]["content"] == "[11-01 周日 09:40] 第8句" and ctx[2]["content"] == "回8"
    assert ctx == folio.context(db, "c")  # 同样的数据拼出来一字不差

    db2 = store.connect()
    seed(db2, 10)
    try:
        folio.compress_once(db2, "c", NAMES, lambda m, **kw: '{"summary": "x"}')
        assert False, "JSON 摘要应该被拒"
    except ValueError:
        pass

    def truncates(messages, **kw):
        if messages[1]["content"].count('"who"') > 4:
            raise llm.Truncated
        return "短的"

    folio.compress_once(db2, "c", NAMES, truncates)
    assert db2.execute("SELECT last_message_id FROM capsules").fetchone()[0] == 4  # 8 轮 → 4 → 2


def test_day_summary():
    db = store.connect()
    t = seed(db, 2)
    seed(db, 2, start=t + timedelta(hours=4))  # 隔 4 小时，第二段
    segs = day_summary.segments(day_summary.day_messages(db, "c", "2026-11-01"))
    assert [len(s) for s in segs] == [4, 4]

    assert day_summary.split_chunks("短\n\n" + "长" * 50 + "\n\n" + "x" * 1001) == [
        "短\n" + "长" * 50, "x" * 500, "x" * 500, "x"]

    vocab = ["薄荷", "海边", "芝麻"]
    fake_embed = lambda texts: [[1.0 if w in s else 0.0 for w in vocab] for s in texts]
    p1, p2 = "早上种了薄荷。" * 6, "晚上约好周末去海边。" * 5
    replies = iter([f"{p1}\n\n{p2}", "无"])  # 第二段没什么可记
    day_summary.summarize_day(db, "c", "2026-11-01", NAMES,
                              chat=lambda m, **kw: next(replies), embed=fake_embed)
    assert db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    hits = day_summary.search(db, "薄荷长得怎么样", embed=fake_embed)
    assert [h["text"] for h in hits] == [p1]
    assert day_summary.search(db, "薄荷", embed=fake_embed, exclude={hits[0]["id"]}) == []
    assert "[09:00] 小满：第0句" in day_summary.read_day(db, "c", "2026-11-01", 0, NAMES)
    # 事后改了切段规则、又补进一句：出处还指着这份总结当初读的那几句
    gap, day_summary.GAP = day_summary.GAP, timedelta(days=1)
    store.add_message(db, "c", "user", "后来补的一句", T0 + timedelta(hours=6))
    seg1 = day_summary.read_day(db, "c", "2026-11-01", 1, NAMES)
    day_summary.GAP = gap
    assert seg1.startswith("[13:10] 小满：第0句") and "后来补的" not in seg1
    assert day_summary.read_day(db, "c", "2026-11-01", 2, NAMES) == "这一段不存在。"


def test_pick():
    cands = [
        {"id": "a", "text": "小满在阳台种了一盆薄荷", "rank": 1, "day": "2026-11-01", "seg": 0},
        {"id": "b", "text": "周末去海边", "rank": 2},
        {"id": "c", "text": "芝麻打翻了水杯", "rank": 3},
        {"id": "d", "text": "阳台的薄荷又长高了", "rank": 4, "archived": True},
        {"id": "a", "text": "小满在阳台种了一盆薄荷", "rank": 5},
    ]
    seen = []

    def judge(messages, **kw):
        seen.append(messages[1]["content"])
        return "[1, 1, 9]"  # 重复的、越界的编号都丢掉

    got, ok = recall_gate.pick("阳台那盆薄荷还好吗", cands, NAMES, prev="今天天气不错", recent_ids={"b"}, chat=judge)
    assert ok and [c["id"] for c in got] == ["a"]
    assert [c["fate"] for c in cands] == ["PICKED", "RECENTLY_SHOWN", "NOT_PICKED", "ARCHIVED", "DUPLICATE"]
    assert seen[0].startswith("阿砚上一句（结尾）：今天天气不错\n小满现在说：") and "2. 芝麻" in seen[0]
    assert 'from="2026-11-01 第 0 段"' in recall_gate.render(got)

    assert recall_gate.pick("在吗", cands[:2], NAMES, chat=lambda m, **kw: "[]") == ([], True)
    for broken in (lambda m, **kw: "我觉得都不太相关", lambda m, **kw: (_ for _ in ()).throw(llm.Truncated)):
        assert recall_gate.pick("在吗", cands[:2], NAMES, chat=broken) == ([], False)
    assert "unavailable" in recall_gate.render([], unavailable=True)


def test_layout():
    history = [{"role": "system", "content": "<folio>…</folio>"},
               {"role": "user", "content": "早"}, {"role": "assistant", "content": "早呀"}]
    msgs, bps = layout.build("你是阿砚。", history, ["<recall/>", ""], "在吗",
                             now=datetime(2026, 11, 7, 21, 3, tzinfo=store.TZ))
    assert [m["role"] for m in msgs] == ["system", "system", "user", "assistant", "user", "user"]
    assert bps == [0, 1, 3]
    assert msgs[4]["content"] == "<volatile_context>\n现在是 2026-11-07 周六 21:03\n\n<recall/>\n</volatile_context>"
    assert msgs[-1]["content"] == "在吗"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
