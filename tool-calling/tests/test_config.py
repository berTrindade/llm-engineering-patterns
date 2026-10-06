from tool_calling_poc.config import load_config


def test_load_config_defaults(monkeypatch):
    for var in ["LLM_BASE_URL", "LLM_API_KEY", "CHEAP_MODEL", "STRONG_MODEL", "EMBED_MODEL"]:
        monkeypatch.delenv(var, raising=False)

    config = load_config()

    assert config.llm_base_url == "http://localhost:11434/v1"
    assert config.llm_api_key == ""
    assert config.cheap_model == "qwen3:8b"
    assert config.strong_model == "qwen3.5:35b"
    assert config.embed_model == "nomic-embed-text"


def test_load_config_reads_env(monkeypatch):
    monkeypatch.setenv("CHEAP_MODEL", "custom-model")

    config = load_config()

    assert config.cheap_model == "custom-model"
