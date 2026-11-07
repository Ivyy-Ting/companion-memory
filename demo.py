"""用一段虚构对话把整套流程跑一遍（需要真实接口，见 README）。"""
from datetime import datetime, timedelta
from pathlib import Path

from memory import day_summary, folio, layout, llm, recall_gate, store

NAMES = {"user": "小满", "assistant": "阿砚"}
DAY1 = datetime(2026, 11, 1, 9, 0, tzinfo=store.TZ)

MORNING = [
    ("早呀，我今天去花市买了一盆薄荷！", "薄荷好呀，放阳台吧，它喜欢太阳。"),
    ("放阳台了，芝麻一直在闻它", "芝麻是想尝尝吗哈哈，别让它啃了。"),
    ("薄荷要多久浇一次水", "土表干了就浇，夏天勤一点，现在两三天一次差不多。"),
    ("好，我记下了", "嗯，过两周可以掐尖，长得更密。"),
]
EVENING = [
    ("今天加班好累", "辛苦啦，先吃点东西，别空着肚子。"),
    ("周末想去海边", "好啊，周六去？天气预报说周六晴。"),
    ("那就周六！我想看日落", "那下午出发，带件外套，海边晚上冷。"),
    ("芝麻怎么办", "让楼下阿姨帮忙喂一顿，你上次说她很喜欢芝麻。"),
    ("对哦", "那就这么定了，周六看日落。"),
]
DAY2 = [
    ("早", "早，昨晚睡得好吗？"),
    ("还行，梦见海了", "看来是真的很想去海边了。"),
]


def seed(db):
    for start, rounds in ((DAY1, MORNING), (DAY1.replace(hour=21), EVENING), (DAY1 + timedelta(days=1), DAY2)):
        for i, (u, a) in enumerate(rounds):
            t = start + timedelta(minutes=3 * i)
            store.add_message(db, "demo", "user", u, t)
            store.add_message(db, "demo", "assistant", a, t + timedelta(seconds=30))


def main():
    db = store.connect()
    seed(db)

    day_summary.summarize_day(db, "demo", "2026-11-01", NAMES)
    folio.compress(db, "demo", NAMES)

    question = "阳台那盆薄荷要多久浇一次水来着？"
    # demo 里直接拿日总结的向量当检索后端；真实环境可以换成任何能给出 相似度+排名 的记忆库
    candidates = day_summary.search(db, question, k=5, min_sim=0)
    accepted = recall_gate.gate(question, candidates, noise=NAMES.values())
    for c in candidates:
        print(f'{c["fate"]:>18}  {c["sim"]:.2f}  {c["text"][:40]}')

    msgs, breakpoints = layout.build(
        Path(__file__).with_name("prompt.md").read_text(),
        folio.context(db, "demo"),
        [recall_gate.render(accepted)],
        question,
    )
    print("\n—— 发给模型的请求 ——")
    for i, m in enumerate(msgs):
        print(f'\n[{i}] {m["role"]}{"  ← 缓存断点" if i in breakpoints else ""}\n{m["content"]}')
    print("\n—— 阿砚 ——\n" + llm.chat(msgs))


if __name__ == "__main__":
    main()
