import lib_python_comfy


def test_version_is_exported():
    assert lib_python_comfy.__version__


def test_httpx_is_importable():
    import httpx  # noqa: F401


def test_pydantic_is_importable():
    import pydantic  # noqa: F401


def test_pillow_is_importable():
    from PIL import Image  # noqa: F401
