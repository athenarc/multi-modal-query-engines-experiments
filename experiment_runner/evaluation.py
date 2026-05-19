from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

#TODO: Replace embedding models similarity with LLM as Judge

class BaseEvaluator(ABC):
    """Template base class for query evaluation."""
    
    def __init__(self, query_id: str, class_name: str):
        self.query_id = query_id
        self.class_name = class_name

    @abstractmethod
    def evaluate(self, predicted_df: pd.DataFrame, ground_truth_table_name: str, input_size: int) -> Dict[str, Any]:
        """
        Evaluate predicted results against ground truth.
        
        Args:
            predicted_df: DataFrame with predictions
            ground_truth_table_name: Name of the CSV file containing ground truth values
            input_size: Size of the input data

        Returns:
            Dictionary with evaluation metrics
        """
        pass

class DerivationEvaluator(BaseEvaluator):
    """Evaluator for derivation queries (column generation tasks)."""

    _model = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._model

    
    def evaluate(self, predicted_df, ground_truth_table_name: str, input_size: int, evaluation_cols: List[str],
                 new_col_name: str, ground_truth_col_name: str, similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """
        Evaluate derivation query results by comparing predicted and ground truth columns.
        
        Args:
            predicted_df: DataFrame with predicted new column
            evaluation_cols: List of columns to load from the ground truth CSV (including the ground truth column)
            new_col_name: Name of the predicted column (suffix: _pred if merged)
            ground_truth_col_name: Name of the ground truth column
            similarity_threshold: Threshold for considering a similarity match

        Returns:
            Dictionary containing:
                - exact_match_accuracy: Fraction of exact matches
                - similarity_accuracy: Fraction of similarity matches
                - incorrect_predictions: List of incorrect predictions with their ground truth
        """
        ground_truth_df = pd.read_csv(f"../{ground_truth_table_name}")[evaluation_cols].head(input_size)

        metrics = {
            'query_id': self.query_id,
            'class_name': self.class_name,
            'exact_match_accuracy': 0,
            'similarity_accuracy': 0.0,
            'incorrect_predictions_exact_match': [],
            'incorrect_predictions_semantic_match': [],
        }
        
        try:
            pred_list = predicted_df[new_col_name].astype(str).tolist()
            truth_list = ground_truth_df[ground_truth_col_name].astype(str).tolist()
            
            total_predictions = len(pred_list)
                    
            # Calculate exact matches
            exact_matches = [p == t for p, t in zip(pred_list, truth_list)]
            total_correct_predictions = sum(exact_matches)

            metrics['exact_match_accuracy'] = total_correct_predictions / total_predictions if total_predictions > 0 else 0.0

            # Calculate semantic similarity matches (e.g., substring match)
            model = self._get_model()
            pred_embeddings = model.encode(pred_list, convert_to_tensor=True)
            truth_embeddings = model.encode(truth_list, convert_to_tensor=True)

            cosine_scores_matrix = util.cos_sim(pred_embeddings, truth_embeddings)

            similarity_scores = cosine_scores_matrix.diag().cpu().numpy()
            semantic_matches = similarity_scores >= similarity_threshold

            metrics['similarity_accuracy'] = np.mean(semantic_matches) if total_predictions > 0 else 0.0
            
            # Collect incorrect predictions
            for i, (pred, truth) in enumerate(zip(pred_list, truth_list)):
                if not exact_matches[i]:
                    metrics['incorrect_predictions_exact_match'].append({
                        'row_index': i,
                        'predicted_row': predicted_df.iloc[i].to_dict(),
                        'ground_truth_row': ground_truth_df.iloc[i].to_dict(),
                        'predicted_value': pred,
                        'ground_truth_value': truth,
                        'semantic_score': round(float(similarity_scores[i]), 4),
                        'is_semantic_match': bool(semantic_matches[i])
                    })
                
                # Also collect predictions that don't meet semantic similarity threshold
                if not semantic_matches[i]:
                    metrics['incorrect_predictions_semantic_match'].append({
                        'row_index': i,
                        'predicted_row': predicted_df.iloc[i].to_dict(),
                        'ground_truth_row': ground_truth_df.iloc[i].to_dict(),
                        'predicted_value': pred,
                        'ground_truth_value': truth,
                        'semantic_score': round(float(similarity_scores[i]), 4),
                        'is_exact_match': bool(exact_matches[i])
                    })
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics

def evaluate_derivation_query(query_id: str, predicted_df: pd.DataFrame, 
                               ground_truth_table_name: str, input_size: int, evaluation_cols: List[str], new_col_name: str,
                               ground_truth_col_name: str) -> Dict[str, Any]:
    """
    Generic method for evaluating derivation queries.
    
    Args:
        query_id: Identifier for the query
        predicted_df: DataFrame with predicted new column
        ground_truth_table_name: Name of the CSV file containing ground truth values
        input_size: Size of the input data to evaluate
        evaluation_cols: List of columns to load from the ground truth CSV (including the ground truth column)
        new_col_name: Name of the predicted column
        ground_truth_col_name: Name of the ground truth column in ground_truth_df
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = DerivationEvaluator(query_id, 'derivation')
    return evaluator.evaluate(
        predicted_df, 
        ground_truth_table_name,
        input_size,
        evaluation_cols, 
        new_col_name, 
        ground_truth_col_name,
    )