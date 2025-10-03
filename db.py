import sqlite3
import os
import pickle
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
    # blob = pickle.dumps(vector.astype(np.float32), protocol=pickle.HIGHEST_PROTOCOL)
    blob = vector.tobytes()
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
        vec = pickle.loads(blob)
        result.append((_id, name, np.array(vec, dtype=np.float32)))
    return result

