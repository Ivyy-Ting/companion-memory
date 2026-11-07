"""召回门槛：检索后端捞上来的候选，哪些真能给模型看。

核心规则：必须有语义证据。新近、重要、常被翻——这些都不能单独让一条记忆"相关"。
阈值是我们在自己的数据上用离线评测（留出集）校准的，换了 embedding 模型请重新校准。
"""
import re

LEXICAL_SIM, LEXICAL_RANK = 0.60, 5
VECTOR_SIM, VECTOR_RANK = 0.62, 2
MARGIN, MARGIN_SIM = 0.08, 0.52
MAX_ACCEPT = 5

CJK = re.compile(r"[一-鿿]+")
LATIN = re.compile(r"[a-z0-9]{4,}")
NOISE = {"我们", "你们", "今天", "现在", "什么", "一下", "这个", "那个", "就是", "还是", "没有", "记得"}

RULES = """以下是可能相关的旧记忆，不是这一轮说的话。
1. 只用和当前话题真相关的，不相关就当没看见。
2. 记忆里没写的细节不要补；不确定就说记不清。
3. 不要提"记忆库""召回"，像自然想起来那样说。
4. 记忆之间矛盾时，以时间更近的为准。"""


def _bigrams(s, noise):
    return {run[i : i + 2] for run in CJK.findall(s) for i in range(len(run) - 1)} - noise


def lexical_anchor(query, text, noise=()):
    """字面上有没有像样的重合：4 位以上的英文/数字词，或 2 个以上中文双字。noise 里放名字、口头禅。"""
    q, t = query.lower(), text.lower()
    if set(LATIN.findall(q)) & set(LATIN.findall(t)):
        return True
    noise = NOISE | set(noise)
    return len(_bigrams(q, noise) & _bigrams(t, noise)) >= 2


def decide(c, margin_gap, lexical, allow_margin=False):
    vector = c["rank"] <= VECTOR_RANK and c["sim"] >= VECTOR_SIM
    margin = c["rank"] == 1 and margin_gap >= MARGIN and c["sim"] >= MARGIN_SIM
    if lexical and (vector or margin):
        return "HYBRID"
    if lexical and c["sim"] >= LEXICAL_SIM and c["rank"] <= LEXICAL_RANK:
        return "LEXICAL"
    if vector:
        return "VECTOR"
    if margin and allow_margin:  # 只靠"第一名甩开第二名"最容易误召，默认关
        return "MARGIN"
    return None


def gate(query, candidates, recent_ids=(), noise=(), allow_margin=False):
    """candidates: [{id, text, sim, rank, archived?}]，rank 是纯语义排名（1 起）。
    每条候选都会写上 fate（为什么收 / 为什么拒），方便事后查。返回被收下的。"""
    sims = sorted((c["sim"] for c in candidates), reverse=True) + [0.0, 0.0]
    accepted = []
    for c in sorted(candidates, key=lambda c: c["rank"]):
        if c["id"] in recent_ids:
            c["fate"] = "RECENTLY_SHOWN"
        elif c.get("archived"):
            c["fate"] = "ARCHIVED"  # 归档的记忆在日常聊天里不复活
        else:
            reason = decide(c, sims[0] - sims[1], lexical_anchor(query, c["text"], noise), allow_margin)
            if not reason:
                c["fate"] = "NO_SEMANTIC_SIGNAL"
            elif len(accepted) >= MAX_ACCEPT:
                c["fate"] = "OVER_LIMIT"
            else:
                c["fate"] = reason
                accepted.append(c)
    return accepted


def render(accepted, unavailable=False):
    """"没找到"和"查不了"要分开告诉模型，不然它会把服务挂了当成没这回事。"""
    if unavailable:
        return '<recall status="unavailable">记忆暂时查不到。别假装记得，也别说没发生过。</recall>'
    if not accepted:
        return '<recall status="empty">没有找到相关的旧记忆。</recall>'
    items = "\n".join(f'<memory id="{c["id"]}" reason="{c["fate"]}">{c["text"]}</memory>' for c in accepted)
    return f'<recall status="ok">\n{RULES}\n{items}\n</recall>'
