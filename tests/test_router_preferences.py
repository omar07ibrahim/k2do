from k2do.agent.router import classify_query, get_route_label


def test_music_generation_prefers_deepthink() -> None:
    route = classify_query("Сгенерируй трек в арабском стиле")
    assert route == "deepthink"


def test_simple_route_label_uses_k2_think_name() -> None:
    assert get_route_label("simple") == "K2 Think (Single-Agent)"


def test_greeting_stays_simple() -> None:
    assert classify_query("hello") == "simple"

