import sqlite3

import pandas as pd

from .config import DATABASE


def initialize_database():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS network_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cell_id TEXT NOT NULL,
                cpu REAL NOT NULL,
                traffic REAL NOT NULL,
                packet_loss REAL NOT NULL,
                latency REAL NOT NULL,
                signal_strength REAL NOT NULL,
                throughput REAL NOT NULL,
                predicted_fault TEXT NOT NULL,
                confidence REAL NOT NULL,
                explanation TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (
                    cell_id, cpu, traffic, packet_loss,
                    latency, signal_strength, throughput
                )
            )
            """
        )


def get_cached_analysis(metrics):
    with sqlite3.connect(DATABASE) as conn:
        return conn.execute(
            """
            SELECT predicted_fault, confidence, explanation
            FROM network_history
            WHERE cell_id = ?
              AND cpu = ?
              AND traffic = ?
              AND packet_loss = ?
              AND latency = ?
              AND signal_strength = ?
              AND throughput = ?
            """,
            (
                metrics["cell_id"],
                metrics["cpu"],
                metrics["traffic"],
                metrics["packet_loss"],
                metrics["latency"],
                metrics["signal_strength"],
                metrics["throughput"],
            ),
        ).fetchone()


def save_analysis(metrics, prediction, explanation):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute(
            """
            INSERT INTO network_history (
                cell_id, cpu, traffic, packet_loss,
                latency, signal_strength, throughput,
                predicted_fault, confidence, explanation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (
                cell_id, cpu, traffic, packet_loss,
                latency, signal_strength, throughput
            )
            DO UPDATE SET
                predicted_fault = excluded.predicted_fault,
                confidence = excluded.confidence,
                explanation = excluded.explanation
            """,
            (
                metrics["cell_id"],
                metrics["cpu"],
                metrics["traffic"],
                metrics["packet_loss"],
                metrics["latency"],
                metrics["signal_strength"],
                metrics["throughput"],
                prediction["fault"],
                prediction["confidence"],
                explanation,
            ),
        )


def get_all_history():
    with sqlite3.connect(DATABASE) as conn:
        return pd.read_sql_query("SELECT * FROM network_history ORDER BY id DESC", conn)
