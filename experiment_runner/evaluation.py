from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np


class BaseEvaluator(ABC):
    """Template base class for query evaluation."""
    
    def __init__(self, query_id: str, class_name: str):
        self.query_id = query_id
        self.class_name = class_name

    @abstractmethod
    def evaluate(self, predicted_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate predicted results against ground truth.
        
        Args:
            predicted_df: DataFrame with predictions
            ground_truth_df: DataFrame with ground truth values
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    # def _align_dataframes(self, predicted_df: pd.DataFrame, ground_truth_df: pd.DataFrame, 
    #                       key_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     """
    #     Align two dataframes by key columns to ensure row-by-row comparison.
        
    #     Args:
    #         predicted_df: DataFrame with predictions
    #         ground_truth_df: DataFrame with ground truth
    #         key_cols: Columns to use as keys for alignment
            
    #     Returns:
    #         Tuple of aligned dataframes
    #     """
    #     # Merge on key columns to ensure alignment
    #     if key_cols:
    #         merged = predicted_df.merge(ground_truth_df, on=key_cols, how='inner', suffixes=('_pred', '_truth'))
    #         return merged
    #     return None

class DerivationEvaluator(BaseEvaluator):
    """Evaluator for derivation queries (column generation tasks)."""
    
    def evaluate(self, predicted_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
                 new_col_name: str, ground_truth_col_name: str, 
                 key_cols: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate derivation query results by comparing predicted and ground truth columns.
        
        Args:
            predicted_df: DataFrame with predicted new column
            ground_truth_df: DataFrame with ground truth values
            new_col_name: Name of the predicted column (suffix: _pred if merged)
            ground_truth_col_name: Name of the ground truth column
            key_cols: Columns to use for aligning rows (if None, uses index alignment)
            
        Returns:
            Dictionary containing:
                - accuracy: Fraction of exact matches
                - partial_matches: Fraction of partial/substring matches
                - total_predictions: Total number of predictions
                - total_correct: Number of correct predictions
                - total_partial: Number of partial matches
                - incorrect_values: List of incorrect predictions with their ground truth
        """
        
        metrics = {
            'query_id': self.query_id,
            'class_name': self.class_name,
            'total_predictions': 0,
            'total_correct': 0,
            'total_partial': 0,
            'accuracy': 0.0,
            'partial_match_rate': 0.0,
            'correct_values': [],
            'incorrect_predictions': []
        }
        
        try:
            # Align dataframes if key columns provided
            if key_cols:
                merged_df = predicted_df.merge(
                    ground_truth_df, 
                    on=key_cols, 
                    how='inner', 
                    suffixes=('_pred', '_truth')
                )
                pred_col = f"{new_col_name}_pred" if f"{new_col_name}_pred" in merged_df.columns else new_col_name
                truth_col = f"{ground_truth_col_name}_truth" if f"{ground_truth_col_name}_truth" in merged_df.columns else ground_truth_col_name
            else:
                # Use index alignment
                merged_df = predicted_df.copy()
                merged_df[ground_truth_col_name] = ground_truth_df[ground_truth_col_name]
                pred_col = new_col_name
                truth_col = ground_truth_col_name
            
            # Ensure columns exist
            if pred_col not in merged_df.columns or truth_col not in merged_df.columns:
                raise ValueError(f"Columns {pred_col} or {truth_col} not found in merged dataframe")
            
            metrics['total_predictions'] = len(merged_df)
            
            # Calculate exact matches
            exact_matches = merged_df[pred_col] == merged_df[truth_col]
            metrics['total_correct'] = exact_matches.sum()
            metrics['accuracy'] = metrics['total_correct'] / metrics['total_predictions'] if metrics['total_predictions'] > 0 else 0.0
            
            # Calculate partial matches (substring matching) for string columns
            if merged_df[pred_col].dtype == 'object' and merged_df[truth_col].dtype == 'object':
                partial_matches = merged_df.apply(
                    lambda row: self._is_substring_match(str(row[pred_col]), str(row[truth_col])),
                    axis=1
                )
                metrics['total_partial'] = partial_matches.sum()
                metrics['partial_match_rate'] = metrics['total_partial'] / metrics['total_predictions'] if metrics['total_predictions'] > 0 else 0.0
            
            # Collect correct and incorrect predictions
            for idx, row in merged_df.iterrows():
                pred_val = row[pred_col]
                truth_val = row[truth_col]
                
                if pred_val == truth_val:
                    metrics['correct_values'].append({
                        'predicted': pred_val,
                        'ground_truth': truth_val,
                        'row_index': idx
                    })
                else:
                    metrics['incorrect_predictions'].append({
                        'predicted': pred_val,
                        'ground_truth': truth_val,
                        'row_index': idx
                    })
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    @staticmethod
    def _is_substring_match(pred_val: str, truth_val: str, case_sensitive: bool = False) -> bool:
        if not case_sensitive:
            pred_val = pred_val.lower()
            truth_val = truth_val.lower()
        
        return pred_val in truth_val or truth_val in pred_val


def evaluate_derivation_query(query_id: str, predicted_df: pd.DataFrame, 
                               ground_truth_df: pd.DataFrame, new_col_name: str,
                               ground_truth_col_name: str, key_cols: List[str] = None) -> Dict[str, Any]:
    """
    Generic method for evaluating derivation queries.
    
    Args:
        query_id: Identifier for the query
        predicted_df: DataFrame with predicted new column
        ground_truth_df: DataFrame with ground truth values
        new_col_name: Name of the predicted column
        ground_truth_col_name: Name of the ground truth column in ground_truth_df
        key_cols: Columns to use for row alignment
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = DerivationEvaluator(query_id, 'derivation')
    return evaluator.evaluate(
        predicted_df, 
        ground_truth_df, 
        new_col_name, 
        ground_truth_col_name,
        key_cols
    )
