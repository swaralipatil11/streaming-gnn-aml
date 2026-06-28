import sqlite3
import json
import os
import time

DB_PATH = "aml_platform.db"

def get_connection():
    # check_same_thread=False is required for multi-threaded FastAPI access
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Create transactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_bank TEXT,
            from_account TEXT,
            to_bank TEXT,
            to_account TEXT,
            amount REAL,
            payment_format TEXT,
            timestamp REAL
        )
    """)
    
    # 2. Create accounts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            account_id TEXT PRIMARY KEY,
            in_degree INTEGER DEFAULT 0,
            out_degree INTEGER DEFAULT 0,
            amount_sent REAL DEFAULT 0.0,
            amount_received REAL DEFAULT 0.0,
            risk_score REAL DEFAULT 0.0,
            is_illicit INTEGER DEFAULT 0
        )
    """)
    
    # 3. Create inference_tasks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inference_tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT,
            result_json TEXT,
            created_at REAL
        )
    """)
    
    conn.commit()
    conn.close()
    print("SQLite Database initialized successfully.")

def save_transaction(from_bank, from_account, to_bank, to_account, amount, payment_format):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transactions (from_bank, from_account, to_bank, to_account, amount, payment_format, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (from_bank, from_account, to_bank, to_account, amount, payment_format, time.time()))
    conn.commit()
    conn.close()

def get_account_stats(account_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def save_account_stats(account_id, in_degree, out_degree, amount_sent, amount_received, risk_score, is_illicit):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO accounts (account_id, in_degree, out_degree, amount_sent, amount_received, risk_score, is_illicit)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id) DO UPDATE SET
            in_degree = excluded.in_degree,
            out_degree = excluded.out_degree,
            amount_sent = excluded.amount_sent,
            amount_received = excluded.amount_received,
            risk_score = excluded.risk_score,
            is_illicit = excluded.is_illicit
    """, (account_id, in_degree, out_degree, amount_sent, amount_received, risk_score, int(is_illicit)))
    conn.commit()
    conn.close()

def increment_account_stats(account_id, in_deg_inc, out_deg_inc, sent_inc, recv_inc, risk_score, is_illicit):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO accounts (account_id, in_degree, out_degree, amount_sent, amount_received, risk_score, is_illicit)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id) DO UPDATE SET
            in_degree = in_degree + excluded.in_degree,
            out_degree = out_degree + excluded.out_degree,
            amount_sent = amount_sent + excluded.amount_sent,
            amount_received = amount_received + excluded.amount_received,
            risk_score = excluded.risk_score,
            is_illicit = excluded.is_illicit
    """, (account_id, in_deg_inc, out_deg_inc, sent_inc, recv_inc, risk_score, int(is_illicit)))
    conn.commit()
    conn.close()

def register_task(task_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO inference_tasks (task_id, status, created_at)
        VALUES (?, ?, ?)
    """, (task_id, "PENDING", time.time()))
    conn.commit()
    conn.close()

def update_task_status(task_id, status, result_json=None):
    conn = get_connection()
    cursor = conn.cursor()
    if result_json is not None:
        cursor.execute("""
            UPDATE inference_tasks
            SET status = ?, result_json = ?
            WHERE task_id = ?
        """, (status, json.dumps(result_json), task_id))
    else:
        cursor.execute("""
            UPDATE inference_tasks
            SET status = ?
            WHERE task_id = ?
        """, (status, task_id))
    conn.commit()
    conn.close()

def get_task_status(task_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT status, result_json FROM inference_tasks WHERE task_id = ?", (task_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "status": row["status"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None
        }
    return None
