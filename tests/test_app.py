import src.app as app_module


def test_create_app_with_explicit_sample_data():
    app = app_module.create_app(use_sample_data=True)

    assert app.layout is not None
    assert app.server is not None
    assert app.title == "Video Game Sales Dashboard"
