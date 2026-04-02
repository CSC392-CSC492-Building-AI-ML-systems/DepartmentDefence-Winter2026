import importlib
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import rag.corpus
import rag.embedding_client
import rag.pipeline


def import_backend_app_module(temp_db_path: Path):
    real_connect = sqlite3.connect
    fake_chunk = SimpleNamespace(chunk_id="fixture-chunk", text="Fixture text")

    with patch.object(rag.corpus, "list_docs", return_value=["fixture-doc.txt"]), patch.object(
        rag.pipeline, "load_chunks_from_docs", return_value=[fake_chunk]
    ), patch.object(rag.pipeline, "embed_chunks", return_value=np.ones((1, 1), dtype=np.float32)), patch.object(
        rag.embedding_client, "create_client", return_value=object()
    ), patch(
        "sqlite3.connect",
        side_effect=lambda *args, **kwargs: real_connect(str(temp_db_path), **kwargs),
    ):
        sys.modules.pop("app", None)
        module = importlib.import_module("app")

    module.DB_PATH = temp_db_path
    return module
