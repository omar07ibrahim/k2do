from __future__ import annotations

from k2do.agent.tools.web import _strip_tags


def test_visible_text_parser_suppresses_script_and_style_contents() -> None:
    markup = (
        "<p>Visible &amp; safe.</p> "
        '<script type="text/javascript">hidden-script()</script > '
        '<style media="all">.hidden-style { display: block; }</style > '
        "<span>Tail.</span>"
    )
    visible = _strip_tags(markup)

    assert " ".join(visible.split()) == "Visible & safe. Tail."
    assert "hidden-script" not in visible
    assert "hidden-style" not in visible


def test_visible_text_parser_handles_case_and_spaced_closing_tag() -> None:
    markup = '<SCRIPT data-fixture=">">hidden</SCRIPT   >kept'

    assert _strip_tags(markup) == "kept"
