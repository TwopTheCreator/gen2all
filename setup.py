from setuptools import setup, find_packages

setup(
    name="gen2all",
    version="2.1.4",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.12.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "requests>=2.25.0",
        "cryptography>=3.4.0",
        "psutil>=5.8.0",
        "redis>=4.0.0",
        "sqlite3-fts4",
        "msgpack>=1.0.0",
        "lz4>=3.1.0",
        "xxhash>=2.0.0",
        "multiprocessing-logging>=0.3.0"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "gen2all-server=gen2all.api.server:main",
            "gen2all-train=gen2all.core.trainer:main",
        ],
    },
)