from k2do.utils.helpers import sanitize_model_output


def test_sanitize_strips_reasoning_tail_tokens() -> None:
    raw = "Internal notes...\n</think_fast>\nOK<|im_end|>"
    assert sanitize_model_output(raw) == "OK"


def test_sanitize_keeps_plain_text() -> None:
    assert sanitize_model_output("READY") == "READY"
