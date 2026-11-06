import json

import pytest

from gr2mini.m05_semantic_ids.sid_index import SIDIndex
from gr2mini.m09_prompts.prompts import TARGET_HINT, build_rerank_messages
from gr2mini.m12_traces.teacher import OpenAICompatClient
from gr2mini.m12_traces.traces import check_trace, generate_traces, sample_trace

ITEMS = [{"item_id": i, "title": f"Product {i}", "categories": ["Beauty", "Hair"]} for i in range(30)]
INDEX = SIDIndex([(i, 0, 0, 1) for i in range(30)], codebook_size=32)
PCFG = {"max_history": 5, "max_title_words": 10, "max_category_levels": 2, "knowledge_priming": False}
TCFG = {"temperature": 0.7, "max_tokens": 100, "max_attempts": 3}


def ex(user=0, target_rank=3):
    cands = list(range(10, 20))
    return {"user": user, "history": [1, 2, 3], "candidates": cands, "target": cands[target_rank - 1],
            "target_rank": target_rank}


def ans(top, expl="The user bought hair products, so candidate fits."):
    order = [top] + [k for k in range(1, 11) if k != top]
    return json.dumps({"explanation": expl, "recommendations": [str(k) for k in order]})


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, n=1, temperature=0.7, max_tokens=1500):
        self.calls.append({"messages": messages, "n": n, "temperature": temperature, "max_tokens": max_tokens})
        return [self.replies.pop(0) for _ in range(n)]


# ======================================================================= check_trace
def test_check_trace_reasons():
    e = ex(target_rank=3)
    assert check_trace(ans(3), e, INDEX)[:2] == (True, "ok")
    assert check_trace("no idea", e, INDEX)[:2] == (False, "unparsed")
    assert check_trace(ans(2), e, INDEX)[:2] == (False, "wrong_top1")
    assert check_trace(ans(3, expl="  "), e, INDEX)[:2] == (False, "no_reasoning")
    ok_sid = INDEX.sid_str(12)
    assert check_trace(ans(3, expl=f"Like {ok_sid} and {INDEX.sid_str(2)}"), e, INDEX)[0] is True
    bad = ans(3, expl=f"Similar to {INDEX.sid_str(25)}")  # item 25 is not in the prompt
    assert check_trace(bad, e, INDEX)[:2] == (False, "hallucinated_sid")
    assert check_trace("<think>hmm</think>" + ans(3), e, INDEX)[0] is True


# ======================================================================= sample_trace
def test_rejection_stops_at_first_accepted_and_saves_policy_view():
    client = FakeClient([ans(1), "garbage", ans(3), ans(3)])
    rec = sample_trace(client, ex(target_rank=3), ITEMS, INDEX, "rejection", PCFG, TCFG)
    assert rec["accepted"] and rec["attempts"] == 3 and rec["reason"] == "ok" and rec["mode"] == "rejection"
    assert len(client.calls) == 3 and client.replies == [ans(3)]
    assert client.calls[0]["temperature"] == 0.7 and client.calls[0]["max_tokens"] == 100
    policy = build_rerank_messages(ex(target_rank=3), ITEMS, INDEX, "policy", PCFG)
    assert client.calls[0]["messages"] == policy, "rejection prompt == policy prompt (no hint)"
    assert rec["messages"][:2] == policy and rec["messages"][2]["role"] == "assistant"
    assert json.loads(rec["messages"][2]["content"])["recommendations"][0] == "3"
    assert rec["raw"] == ans(3) and rec["user"] == 0


def test_rejection_gives_up_after_max_attempts():
    client = FakeClient([ans(1)] * 5)
    rec = sample_trace(client, ex(target_rank=3), ITEMS, INDEX, "rejection", PCFG, TCFG)
    assert not rec["accepted"] and rec["attempts"] == 3 and rec["messages"] is None and rec["reason"] == "wrong_top1"


def test_targeted_uses_hint_once_but_saves_policy_prompt():
    client = FakeClient([ans(3)])
    rec = sample_trace(client, ex(target_rank=3), ITEMS, INDEX, "targeted", PCFG, TCFG)
    assert len(client.calls) == 1 and TARGET_HINT.format(rank=3) in client.calls[0]["messages"][1]["content"]
    assert rec["accepted"] and rec["attempts"] == 1
    assert "Candidate 3 next" not in rec["messages"][1]["content"], "the saved prompt must NOT contain the hint"
    client2 = FakeClient([ans(5), ans(3)])
    rec2 = sample_trace(client2, ex(target_rank=3), ITEMS, INDEX, "targeted", PCFG, TCFG)
    assert not rec2["accepted"] and len(client2.calls) == 1
    with pytest.raises(ValueError):
        sample_trace(client, ex(), ITEMS, INDEX, "magic", PCFG, TCFG)


# ======================================================================= generate_traces
def test_generate_traces_stats_and_resume(tmp_path):
    out = tmp_path / "t" / "traces.jsonl"
    exs = [ex(user=u, target_rank=3) for u in range(4)]
    client = FakeClient([ans(3), ans(1), ans(1), ans(1), ans(3), ans(3)] + [ans(3)] * 10)
    stats = generate_traces(client, exs[:3], ITEMS, INDEX, "rejection", out, PCFG, TCFG, log=None)
    # user 0: ok on call 1; user 1: 3 failures; user 2: ok on call 1
    assert stats == {"tried": 3, "accepted": 2, "acceptance_rate": pytest.approx(2 / 3), "calls": 5, "skipped": 0}
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [r["user"] for r in lines] == [0, 1, 2]
    stats2 = generate_traces(client, exs, ITEMS, INDEX, "rejection", out, PCFG, TCFG, log=None)
    assert stats2["skipped"] == 3 and stats2["tried"] == 1
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4
    stats3 = generate_traces(client, exs[:1], ITEMS, INDEX, "targeted", out, PCFG, TCFG, log=None)
    assert stats3["skipped"] == 0 and stats3["tried"] == 1, "resume is per mode"


# ======================================================================= HTTP client
class FakeResponse:
    def __init__(self, content, status=200):
        self.content, self.status = content, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return {"choices": [{"message": {"role": "assistant", "content": self.content}}]}


class FakeSession:
    def __init__(self, status=200):
        self.posts, self.status = [], status

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse(f"reply {len(self.posts)}", self.status)


def test_openai_compat_client_requests():
    s = FakeSession()
    c = OpenAICompatClient("http://localhost:11434/v1/", "qwen3:4b", api_key="sk-x", timeout=12, session=s)
    out = c.chat([{"role": "user", "content": "hi"}], n=2, temperature=0.3, max_tokens=50)
    assert out == ["reply 1", "reply 2"] and len(s.posts) == 2
    p = s.posts[0]
    assert p["url"] == "http://localhost:11434/v1/chat/completions"
    assert p["json"] == {"model": "qwen3:4b", "messages": [{"role": "user", "content": "hi"}],
                         "temperature": 0.3, "max_tokens": 50, "n": 1}
    assert p["headers"] == {"Authorization": "Bearer sk-x"} and p["timeout"] == 12


def test_openai_compat_client_no_key_and_errors():
    s = FakeSession()
    OpenAICompatClient("http://x/v1", "m", session=s).chat([{"role": "user", "content": "hi"}])
    assert s.posts[0]["headers"] == {}
    with pytest.raises(RuntimeError):
        OpenAICompatClient("http://x/v1", "m", session=FakeSession(status=500)).chat([{"role": "user", "content": "hi"}])
