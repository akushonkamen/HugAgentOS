"""Replayed history must not carry duplicate tool-call identities to the provider.

Plan-execute used to re-collect the replayed history's tool calls into every
step's persisted log (issue #140), so sessions written by those builds can
hold the same tool-use id — and its result — several times. Bedrock-backed
upstreams reject the whole request when an id repeats or when a result's call
is missing, so replay dedupes defensively: the first occurrence of a call
wins, each result follows the call it answers, and a row left without any
block disappears entirely.
"""

from core.llm.message_compat import session_to_msgs


def _blocks(msgs, kind):
    return [
        block
        for msg in msgs
        if msg.has_content_blocks(kind)
        for block in msg.get_content_blocks(kind)
    ]


def _poisoned_rows():
    """What a session poisoned by the old collector looks like on disk.

    One real call (``c1``) plus a duplicate copy of the call/result pair, an
    orphan result whose call never existed, and untouched text around them.
    """
    return [
        {"role": "user", "content": "查一下"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "我来查"},
                {"type": "tool_call", "id": "c1", "name": "search", "input": '{"q": "x"}'},
            ],
        },
        {
            "role": "tool",
            "content": [{"type": "tool_result", "id": "c1", "name": "search", "output": "结果A"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "补充"},
                {"type": "tool_call", "id": "c1", "name": "search", "input": '{"q": "x"}'},
            ],
        },
        {
            "role": "tool",
            "content": [
                {"type": "tool_result", "id": "c1", "name": "search", "output": "结果A"},
                {"type": "tool_result", "id": "cX", "name": "search", "output": "孤儿结果"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "答案"}]},
        {"role": "user", "content": "下一个问题"},
    ]


def test_duplicate_tool_ids_collapse_to_the_first_occurrence():
    msgs = session_to_msgs(_poisoned_rows())

    calls = _blocks(msgs, "tool_call")
    results = _blocks(msgs, "tool_result")
    assert [block.id for block in calls] == ["c1"]
    assert [block.id for block in results] == ["c1"]
    # Everything that was not a duplicate survives untouched, in order.
    texts = [
        block.text
        for msg in msgs
        if msg.role == "assistant"
        for block in msg.get_content_blocks("text")
    ]
    assert texts == ["我来查", "补充", "答案"]
    # The row that only carried dropped blocks disappears; the others stay.
    assert msgs[-1].get_text_content() == "下一个问题"


def test_dedupe_never_mutates_the_caller_rows():
    rows = _poisoned_rows()

    def _snapshot(items):
        return [
            (
                dict(
                    row,
                    content=[
                        dict(block) if isinstance(block, dict) else block
                        for block in row["content"]
                    ],
                )
                if isinstance(row.get("content"), list)
                else dict(row)
            )
            for row in items
        ]

    snapshot = _snapshot(rows)

    session_to_msgs(rows)

    assert rows == snapshot


def test_clean_history_passes_through_unchanged():
    rows = [
        {"role": "user", "content": "问"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_call", "id": "a", "name": "s", "input": "{}"},
                {"type": "tool_call", "id": "b", "name": "s", "input": "{}"},
            ],
        },
        {
            "role": "tool",
            "content": [
                {"type": "tool_result", "id": "a", "name": "s", "output": "1"},
                {"type": "tool_result", "id": "b", "name": "s", "output": "2"},
            ],
        },
    ]

    msgs = session_to_msgs(rows)

    assert [block.id for block in _blocks(msgs, "tool_call")] == ["a", "b"]
    assert [block.id for block in _blocks(msgs, "tool_result")] == ["a", "b"]
