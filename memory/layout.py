"""把所有东西排成一次请求：不变的在前，每轮变的在最后。"""
from datetime import datetime

from .store import TZ, WEEK


def build(prompt, history, volatile, current, now=None):
    """
    prompt    固定 prompt（人设 + 规则），永远不变
    history   folio.context() 的结果：胶囊 system 消息 + 最近原文
    volatile  这一轮才有的东西（召回、日总结召回…），拼成一条 user 消息放在当前这句前面，不存进历史
    current   这一句

    返回 (messages, breakpoints)。breakpoints 是建议打缓存的位置：
    OpenAI / DeepSeek 自动缓存前缀，不用管；Anthropic 在这几条上加 cache_control。
    """
    now = now or datetime.now(TZ)
    snapshot = [m for m in history if m["role"] == "system"]
    rest = [m for m in history if m["role"] != "system"]

    msgs, breakpoints = [{"role": "system", "content": prompt}], [0]
    for block in (snapshot, rest):
        if block:
            msgs += block
            breakpoints.append(len(msgs) - 1)

    # "现在几点"只放这里。放进 system 的话每轮都变，整个缓存就废了
    parts = [f"现在是 {now:%Y-%m-%d} 周{WEEK[now.weekday()]} {now:%H:%M}", *filter(None, volatile)]
    msgs.append({"role": "user", "content": "<volatile_context>\n" + "\n\n".join(parts) + "\n</volatile_context>"})
    msgs.append({"role": "user", "content": current})
    return msgs, breakpoints
