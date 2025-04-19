"""
Feedback database module for the SKU prediction system.

This module provides functions for storing and retrieving user feedback
on SKU predictions, as well as tracking model retraining history.
"""

import os
import sqlite3
import json
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Database path
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, 'feedback.db')

def get_db_connection():
    """Create a connection to the SQLite database.
    
    Returns:
        sqlite3.Connection: A connection to the SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

def init_db():
    """Initialize the database with the required tables.
    
    Creates the feedback and retraining_history tables if they don't exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create feedback table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT NOT NULL,
        maker TEXT NOT NULL,
        series TEXT,
        model_year TEXT,
        predicted_sku TEXT NOT NULL,
        is_correct BOOLEAN NOT NULL,
        correct_sku TEXT,
        confidence REAL,
        timestamp TEXT NOT NULL,
        processed BOOLEAN DEFAULT FALSE
    )
    ''')
    
    # Create a table to store retraining history
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS retraining_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        feedback_count INTEGER NOT NULL,
        previous_accuracy REAL,
        new_accuracy REAL,
        model_path TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database initialized at {DB_PATH}")

def save_feedback(feedback_data):
    """Save user feedback to the database.
    
    Args:
        feedback_data (dict): Dictionary containing feedback information
            - description: Product description
            - maker: Vehicle maker
            - series: Vehicle series
            - model_year: Vehicle model year
            - predicted_sku: SKU predicted by the model
            - is_correct: Boolean indicating if prediction was correct
            - correct_sku: Correct SKU if prediction was wrong
            - confidence: Confidence of the prediction
    
    Returns:
        int: ID of the inserted feedback
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Ensure timestamp is present
    if 'timestamp' not in feedback_data:
        feedback_data['timestamp'] = datetime.now().isoformat()
    
    # Insert feedback into database
    cursor.execute('''
    INSERT INTO feedback (
        description, maker, series, model_year, 
        predicted_sku, is_correct, correct_sku, confidence, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        feedback_data.get('description', ''),
        feedback_data.get('maker', ''),
        feedback_data.get('series', ''),
        feedback_data.get('model_year', ''),
        feedback_data.get('predicted_sku', ''),
        feedback_data.get('is_correct', False),
        feedback_data.get('correct_sku', None),
        feedback_data.get('confidence', 0.0),
        feedback_data.get('timestamp')
    ))
    
    # Get the ID of the inserted feedback
    feedback_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    logger.info(f"Saved feedback with ID {feedback_id}")
    return feedback_id

def get_unprocessed_feedback():
    """Get all unprocessed feedback from the database.
    
    Returns:
        list: List of dictionaries containing unprocessed feedback
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT * FROM feedback WHERE processed = FALSE
    ''')
    
    # Convert rows to dictionaries
    feedback_list = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    logger.info(f"Retrieved {len(feedback_list)} unprocessed feedback items")
    return feedback_list

def mark_feedback_as_processed(feedback_ids):
    """Mark feedback as processed.
    
    Args:
        feedback_ids (list): List of feedback IDs to mark as processed
    """
    if not feedback_ids:
        return
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Convert list to comma-separated string for SQL IN clause
    id_string = ','.join(['?'] * len(feedback_ids))
    
    cursor.execute(f'''
    UPDATE feedback SET processed = TRUE 
    WHERE id IN ({id_string})
    ''', feedback_ids)
    
    conn.commit()
    conn.close()
    
    logger.info(f"Marked {len(feedback_ids)} feedback items as processed")

def save_retraining_record(record):
    """Save a record of model retraining.
    
    Args:
        record (dict): Dictionary containing retraining information
            - timestamp: When retraining occurred
            - feedback_count: Number of feedback items used
            - previous_accuracy: Accuracy before retraining
            - new_accuracy: Accuracy after retraining
            - model_path: Path to the saved model
    
    Returns:
        int: ID of the inserted record
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Ensure timestamp is present
    if 'timestamp' not in record:
        record['timestamp'] = datetime.now().isoformat()
    
    cursor.execute('''
    INSERT INTO retraining_history (
        timestamp, feedback_count, previous_accuracy, new_accuracy, model_path
    ) VALUES (?, ?, ?, ?, ?)
    ''', (
        record.get('timestamp'),
        record.get('feedback_count', 0),
        record.get('previous_accuracy', 0.0),
        record.get('new_accuracy', 0.0),
        record.get('model_path', '')
    ))
    
    record_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    logger.info(f"Saved retraining record with ID {record_id}")
    return record_id

def get_feedback_stats():
    """Get statistics about collected feedback.
    
    Returns:
        dict: Dictionary containing feedback statistics
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute('SELECT COUNT(*) FROM feedback')
    total_count = cursor.fetchone()[0]
    
    # Get correct predictions count
    cursor.execute('SELECT COUNT(*) FROM feedback WHERE is_correct = TRUE')
    correct_count = cursor.fetchone()[0]
    
    # Get incorrect predictions count
    cursor.execute('SELECT COUNT(*) FROM feedback WHERE is_correct = FALSE')
    incorrect_count = cursor.fetchone()[0]
    
    # Get unprocessed count
    cursor.execute('SELECT COUNT(*) FROM feedback WHERE processed = FALSE')
    unprocessed_count = cursor.fetchone()[0]
    
    # Get most recent feedback
    cursor.execute('''
    SELECT * FROM feedback ORDER BY timestamp DESC LIMIT 10
    ''')
    recent_feedback = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    stats = {
        'total_count': total_count,
        'correct_count': correct_count,
        'incorrect_count': incorrect_count,
        'unprocessed_count': unprocessed_count,
        'accuracy_rate': correct_count / total_count if total_count > 0 else 0,
        'recent_feedback': recent_feedback
    }
    
    logger.info(f"Retrieved feedback stats: {total_count} total, {correct_count} correct, {incorrect_count} incorrect")
    return stats

# Initialize the database when this module is imported
init_db()
