# src/train_model.py

import os
import logging
import argparse
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd # Required for loading joblib files which might contain pandas structures if saved differently

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def top_k_accuracy(y_true, y_pred_proba, k=5):
    """
    Calculates the top-k accuracy score.

    Args:
        y_true (array-like): True labels.
        y_pred_proba (array-like): Predicted probabilities for each class (n_samples, n_classes).
        k (int): The number of top predictions to consider.

    Returns:
        float: The top-k accuracy score.
    """
    # Ensure y_true is a numpy array for efficient indexing
    y_true = np.array(y_true)
    # Get the indices of the top k predictions for each sample (sorted descending)
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    # Check if the true label is in the top k predictions for each sample
    # Need to handle potential issues if y_true values aren't directly comparable to indices
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return np.mean(correct)


def train_model(input_data_dir: str, output_model_dir: str, model_filename: str = "lgbm_sku_predictor.joblib"):
    """
    Loads preprocessed data, trains a LightGBM model, evaluates it, and saves the model.
    """
    logging.info("--- Starting Model Training ---")

    # --- Load Data ---
    train_path = os.path.join(input_data_dir, 'train_processed.joblib')
    val_path = os.path.join(input_data_dir, 'val_processed.joblib')
    encoder_path = os.path.join(input_data_dir, 'label_encoder.joblib')

    if not all(os.path.exists(p) for p in [train_path, val_path, encoder_path]):
        logging.error("Processed data files or label encoder not found. Run preprocessing first.")
        return

    try:
        logging.info(f"Loading training data from {train_path}")
        train_data = joblib.load(train_path)
        X_train, y_train = train_data['X'], train_data['y']

        logging.info(f"Loading validation data from {val_path}")
        val_data = joblib.load(val_path)
        X_val, y_val = val_data['X'], val_data['y']

        logging.info(f"Loading label encoder from {encoder_path}")
        label_encoder = joblib.load(encoder_path)
        num_classes = len(label_encoder.classes_)
        logging.info(f"Loaded data. Training shape: {X_train.shape}, Validation shape: {X_val.shape}, Num classes: {num_classes}")
        # Log memory usage
        def get_mem_usage(arr):
            try:
                if hasattr(arr, 'memory_usage'):
                    return arr.memory_usage(deep=True).sum()
                elif hasattr(arr, 'nbytes'):
                    return arr.nbytes
                else:
                    return 'Unknown'
            except Exception as e:
                return f'Error: {e}'
        train_mem = get_mem_usage(X_train)
        val_mem = get_mem_usage(X_val)
        logging.info(f"X_train memory usage: {train_mem} bytes")
        logging.info(f"X_val memory usage: {val_mem} bytes")
        # Optionally limit number of rows for debugging resource issues
        row_limit = os.environ.get('TRAIN_ROW_LIMIT')
        if row_limit is not None:
            try:
                import numpy as np
                row_limit = int(row_limit)
                # Randomly sample rows for train and val
                rng = np.random.default_rng(42)
                # Use .shape[0] for row count (works for both DataFrame and sparse matrix)
                n_train = X_train.shape[0]
                n_val = X_val.shape[0]
                train_indices = rng.choice(n_train, min(row_limit, n_train), replace=False)
                val_indices = rng.choice(n_val, min(row_limit, n_val), replace=False)
                if hasattr(X_train, 'iloc'):
                    X_train_small = X_train.iloc[train_indices]
                    y_train_small = y_train.iloc[train_indices]
                    X_val_small = X_val.iloc[val_indices]
                    y_val_small = y_val.iloc[val_indices]
                else:
                    X_train_small = X_train[train_indices]
                    y_train_small = y_train[train_indices]
                    X_val_small = X_val[val_indices]
                    y_val_small = y_val[val_indices]
                # Filter val set to only classes present in train set
                train_classes = set(np.unique(y_train_small))
                val_mask = np.isin(y_val_small, list(train_classes))
                X_val_small = X_val_small[val_mask]
                y_val_small = y_val_small[val_mask]
                # Assign back
                X_train, y_train = X_train_small, y_train_small
                X_val, y_val = X_val_small, y_val_small
                logging.warning(f"Limiting train/val data to {row_limit} random rows for debugging. Filtered val to only classes in train.")
                # Warn if any classes in val are not in train
                unseen_val_classes = set(np.unique(y_val)) - train_classes
                if unseen_val_classes:
                    logging.warning(f"After filtering, val set still contains unseen classes: {unseen_val_classes}")
            except Exception as e:
                logging.error(f"Failed to apply TRAIN_ROW_LIMIT: {e}")

    except Exception as e:
        logging.error(f"Failed to load data or encoder: {e}")
        return

    # --- Model Definition ---
    logging.info("Defining LightGBM model...")
    # Parameters can be tuned later using techniques like GridSearchCV or Optuna
    lgbm = lgb.LGBMClassifier(
        objective='multiclass',
        metric='multi_logloss', # Logloss is good for probability calibration
        n_estimators=1000,      # Max number of trees, will stop early
        learning_rate=0.05,
        num_leaves=31,          # Default, balance between speed and accuracy
        n_jobs=-1,              # Use all available CPU cores
        random_state=42,
        num_class=num_classes   # Explicitly set number of classes
    )

    # --- Model Training ---
    logging.info("Starting model training with early stopping...")
    # Use validation set for early stopping
    early_stopping_callback = lgb.early_stopping(stopping_rounds=50, verbose=True) # Stop if no improvement after 50 rounds

    try:
        lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss', # Monitor logloss on validation set
            callbacks=[early_stopping_callback]
        )
        logging.info("Model training finished.")
        logging.info(f"Best iteration: {lgbm.best_iteration_}")

    except Exception as e:
        logging.error(f"Model training failed: {e}")
        return

    # --- Evaluation ---
    logging.info("Evaluating model on validation set...")
    y_pred_proba_val = lgbm.predict_proba(X_val)
    y_pred_val = np.argmax(y_pred_proba_val, axis=1) # Get class with highest probability

    accuracy_val = accuracy_score(y_val, y_pred_val)
    top5_accuracy_val = top_k_accuracy(y_val, y_pred_proba_val, k=5)

    logging.info(f"Validation Accuracy: {accuracy_val:.4f}")
    logging.info(f"Validation Top-5 Accuracy: {top5_accuracy_val:.4f}")

    # --- Save Model ---
    os.makedirs(output_model_dir, exist_ok=True)
    model_save_path = os.path.join(output_model_dir, model_filename)
    logging.info(f"Saving trained model to {model_save_path}")
    try:
        joblib.dump(lgbm, model_save_path)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")

    logging.info("--- Model Training Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a LightGBM model for SKU prediction.")
    parser.add_argument(
        "--input-dir",
        default="data/processed",
        help="Directory containing preprocessed data splits and label encoder."
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save the trained model."
    )
    parser.add_argument(
        "--model-name",
        default="lgbm_sku_predictor.joblib",
        help="Filename for the saved model."
    )

    args = parser.parse_args()

    # Determine project root assuming script is in 'src' subdir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Construct absolute paths relative to project root if needed
    input_dir_abs = args.input_dir
    if not os.path.isabs(input_dir_abs):
        input_dir_abs = os.path.join(project_root, input_dir_abs)

    output_dir_abs = args.output_dir
    if not os.path.isabs(output_dir_abs):
        output_dir_abs = os.path.join(project_root, output_dir_abs)

    train_model(input_dir_abs, output_dir_abs, args.model_name)

