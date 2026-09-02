import src.app as app_module
from src.data.data_loader import create_sample_data


def test_create_app_with_sample_data(monkeypatch):
    monkeypatch.setattr(app_module, "load_data", lambda **kwargs: create_sample_data())

    app = app_module.create_app()

    assert app.layout is not None
    assert app.server is not None
