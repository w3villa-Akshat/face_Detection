import sqlite3
import os
import zlib
import numpy as np

DB_PATH = "face_db.sqlite"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            vector BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def add_embedding(name: str, vector: np.ndarray):
    conn = get_conn()
    c = conn.cursor()
    blob = zlib.compress(vector.astype(np.float16).tobytes())
    c.execute('INSERT INTO embeddings (name, vector) VALUES (?, ?)', (name, sqlite3.Binary(blob)))
    conn.commit()
    last_id = c.lastrowid
    conn.close()
    return last_id


def get_all_embeddings():
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT id, name, vector FROM embeddings')
    rows = c.fetchall()

    conn.close()
    result = []
    for _id, name, blob in rows:
        vec = vec = np.frombuffer(zlib.decompress(blob), dtype=np.float16).astype(np.float32)
        result.append((_id, name, vec))
    return result

get_all_embeddings()