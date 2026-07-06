from pact.llm import _clean_base_url


def test_clean_base_url_strips_surrounding_whitespace():
    assert _clean_base_url("https://idealab.alibaba-inc.com/api/openai/v1\t") == (
        "https://idealab.alibaba-inc.com/api/openai/v1"
    )


def test_clean_base_url_preserves_none():
    assert _clean_base_url(None) is None
