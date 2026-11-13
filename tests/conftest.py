import pytest


@pytest.fixture(scope="session")
def native_root():
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def native_window(native_root):
    import tkinter as tk

    window = tk.Toplevel(native_root)
    window.withdraw()
    yield window
    if window.winfo_exists():
        window.destroy()
