import asyncio
import time

from mlx_audio import server


def test_model_provider_retains_warm_model(monkeypatch):
    provider = server.ModelProvider()
    release_calls = []
    monkeypatch.setattr(server, "load_model", lambda model_name: {"id": model_name})
    monkeypatch.setattr(server, "_release_mlx_memory", lambda: release_calls.append(True))

    model = provider.load_model("model-a")
    provider.last_used_at["model-a"] = time.monotonic() - 10

    assert provider.unload_idle_models(30) == []
    assert provider.load_model("model-a") is model
    assert release_calls == []


def test_model_provider_unloads_idle_model(monkeypatch):
    provider = server.ModelProvider()
    release_calls = []
    monkeypatch.setattr(server, "load_model", lambda model_name: {"id": model_name})
    monkeypatch.setattr(server, "_release_mlx_memory", lambda: release_calls.append(True))

    provider.load_model("model-a")
    provider.last_used_at["model-a"] = time.monotonic() - 31

    assert provider.unload_idle_models(30) == ["model-a"]
    assert asyncio.run(provider.get_available_models()) == []
    assert release_calls == [True]


def test_model_provider_idle_unload_can_be_disabled(monkeypatch):
    provider = server.ModelProvider()
    release_calls = []
    monkeypatch.setattr(server, "load_model", lambda model_name: {"id": model_name})
    monkeypatch.setattr(server, "_release_mlx_memory", lambda: release_calls.append(True))

    provider.load_model("model-a")
    provider.last_used_at["model-a"] = time.monotonic() - 3600

    assert provider.unload_idle_models(0) == []
    assert asyncio.run(provider.get_available_models()) == ["model-a"]
    assert release_calls == []


def test_model_provider_reloads_after_idle_unload(monkeypatch):
    provider = server.ModelProvider()
    loads = []
    monkeypatch.setattr(server, "_release_mlx_memory", lambda: None)

    def load(model_name):
        loads.append(model_name)
        return {"id": model_name, "load": len(loads)}

    monkeypatch.setattr(server, "load_model", load)

    first = provider.load_model("model-a")
    provider.last_used_at["model-a"] = time.monotonic() - 31
    assert provider.unload_idle_models(30) == ["model-a"]

    second = provider.load_model("model-a")
    assert first != second
    assert loads == ["model-a", "model-a"]
