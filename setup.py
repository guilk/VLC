from setuptools import setup, find_packages

setup(
    name="vlc",
    packages=find_packages(
        exclude=[".dfc", ".vscode", "dataset", "notebooks", "result", "scripts"]
    ),
    version="1.0.0",
    license="MIT",
    description="Training Vision-Language Transformers from Captions Alone",
    author="Liangke Gui",
    author_email="liangkeg@cs.cmu.edu",
    url="https://github.com/guilk/VLC",
    keywords=["vision and language pretraining"],
    install_requires=["torch", "pytorch_lightning"],
)
