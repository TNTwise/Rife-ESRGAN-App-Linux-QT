import os
IS_FLATPAK = "FLATPAK_ID" in os.environ


if IS_FLATPAK:
    CWD = os.path.join(
        os.path.expanduser("~"), ".var", "app", "io.github.tntwise.REAL-Video-Enhancer"
    )
    if not os.path.exists(CWD):
        CWD = os.path.join(
            os.path.expanduser("~"),
            ".var",
            "app",
            "io.github.tntwise.REAL-Video-EnhancerV2",
        )
else:
    CWD = os.getcwd()