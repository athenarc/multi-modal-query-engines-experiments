from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from openai import OpenAI
import json
from tqdm import tqdm

#TODO: Replace embedding models similarity with LLM as Judge --> working on it

class BaseEvaluator(ABC):
    """Template base class for query evaluation."""
    
    def __init__(self, query_id: str, class_name: str, llm_as_judge: str = "RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8"):
        self.query_id = query_id
        self.class_name = class_name
        self.llm = llm_as_judge

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

    _client = None
    _embedding_model = None

    @classmethod
    def _get_embedding_model(cls):
        if cls._embedding_model is None:
            cls._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._embedding_model

    @classmethod
    def _get_client(cls):
        if cls._client is None:
            cls._client = OpenAI(base_url="http://localhost:5001/v1", api_key="EMPTY")
        return cls._client
    
    # @classmethod
    # def _get_llm(cls):
    #     if cls._llm is None:
    #         cls._llm = "RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8"
    #     return cls._llm

    def _llm_judge_pair(self, nlq: str, row, predicted, ground_truth) -> Dict[str, Any]:
        JUDGE_SYSTEM_PROMPT = """You are a strict semantic equivalence judge. 
                                Your task is to determine whether a predicted cell value conveys the same meaning as the ground truth cell value.
                                You will be given both the input data and the natural language question, instructing what value to predict.

                                Rules:
                                - Respond ONLY with valid JSON — no explanation, no markdown, no extra text.
                                - "is_match" must be true ONLY if the predicted value matches the ground truth in both meaning AND format/structure expected by the question.
                                - "is_match" must be false if the predicted value contains additional content, reasoning, or explanation beyond what the question asked for.
                                - If the question asks for a specific token, label, or keyword (e.g. "answer either X or Y"), the prediction must contain ONLY that token — any surrounding explanation makes it a non-match.
                                - "is_match" must be false if they differ in meaning, contain conflicting facts, or if one is missing key information the other has.
                                - "confidence" must be a float between 0.0 and 1.0 reflecting your certainty.
                                - "reasoning" must be a single short sentence (max 20 words).

                                Response format:
                                {"is_match": <bool>, "confidence": <float>, "reasoning": "<string>"}"""
        
        user_message = f"Input data: {row}\n\nNatural Language Question: {nlq}\n\nPredicted value: {predicted}\n\nGround Truth Value: {ground_truth}"
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.llm,
                max_tokens=128,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                extra_body={
                    "guided_json": {
                        "type": "object",
                        "properties": {
                            "is_match": {"type": "boolean"},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["is_match", "confidence", "reasoning"]
                    }
                },
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)

            return {
                "is_match": bool(result["is_match"]),
                "confidence": float(result["confidence"]),
                "reasoning": str(result["reasoning"])
            }
        except Exception as e:
            print(f"Error during LLM-as-judge evaluation: {e}")
            return {"is_match": False, "confidence": 0.0, "reasoning": f"Judge call failed: {e}"}


    
    def evaluate(self, predicted_df, ground_truth_table_name: str, input_size: int, input_cols: List[str], evaluation_cols: List[str],
                 new_col_name: str, ground_truth_col_name: str, nlq: str, similarity_threshold: float = 0.85, is_pz: bool = False) -> Dict[str, Any]:
        """
        Evaluate derivation query results by comparing predicted and ground truth columns.
        
        Args:
            predicted_df: DataFrame with predicted new column
            input_cols: List of columns that were the input of the operation
            evaluation_cols: List of columns to load from the ground truth CSV (including the ground truth column)
            new_col_name: Name of the predicted column (suffix: _pred if merged)
            ground_truth_col_name: Name of the ground truth column
            similarity_threshold: Threshold for considering a similarity match
            is_pz: Whether the system will be evaluated is Palimpzest, as it produces the prediction rows in different order

        Returns:
            Dictionary containing:
                - exact_match_accuracy: Fraction of exact matches
                - similarity_accuracy: Fraction of similarity matches
                - incorrect_predictions: List of incorrect predictions with their ground truth
        """
        ground_truth_df = pd.read_csv(f"{ground_truth_table_name}")[evaluation_cols].head(input_size)

        # Palimpzest produces the derivation results in different order from the one of the input
        if is_pz:            
            if (ground_truth_col_name == new_col_name):
                predicted_df = predicted_df.rename(columns={new_col_name : new_col_name + "_pred"})

            cols = predicted_df.columns.tolist()
            predicted_df = ground_truth_df.merge(predicted_df, on=input_cols)[cols]

            if (ground_truth_col_name == new_col_name):
                predicted_df = predicted_df.rename(columns={new_col_name + "_pred": new_col_name})

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

            judge_results: List[Dict[str, Any]] = []
            for i, (pred, truth) in enumerate(tqdm(zip(pred_list, truth_list), total=total_predictions, desc=f"LLM Judge [{self.query_id}]")):
                if exact_matches[i]:
                    judge_results.append({"is_match": True, "confidence": 1.0, "reasoning": "exact match"})
                else:
                    curr_row = ground_truth_df.iloc[i][input_cols]
                    judge_results.append(self._llm_judge_pair(nlq=nlq, row=curr_row, predicted=pred, ground_truth=truth))

            semantic_matches = [
                r["is_match"] and r["confidence"] >= similarity_threshold
                for r in judge_results
            ]
            metrics["llm_judge_accuracy"] = sum(semantic_matches) / total_predictions if total_predictions > 0 else 0.0
  
            for i, (pred, truth) in enumerate(zip(pred_list, truth_list)):
                jr = judge_results[i]
                base = {
                    "row_index":         i,
                    "predicted_row":     predicted_df.iloc[i].to_dict(),
                    "ground_truth_row":  ground_truth_df.iloc[i].to_dict(),
                    "predicted_value":   pred,
                    "ground_truth_value": truth,
                    "judge_is_match":    jr["is_match"],
                    "judge_confidence":  round(jr["confidence"], 4),
                    "judge_reasoning":   jr["reasoning"],
                }
                if not exact_matches[i]:
                    metrics["incorrect_predictions_exact_match"].append({**base, "is_semantic_match": semantic_matches[i]})
                if not semantic_matches[i]:
                    metrics["incorrect_predictions_semantic_match"].append({**base, "is_exact_match": exact_matches[i]})
            
        except Exception as e:
            print("PROBLEMO")
            print(e)
            metrics['error'] = str(e)
        
        return metrics

class SelectionEvaluator(BaseEvaluator):
    """Evaluator for selection queries (filtering tasks)."""
    
    def evaluate(self, predicted_df, ground_truth_table_name: str, input_size: int, evaluation_cols: List[str],
                 filtering_col: str) -> Dict[str, Any]:
        """
        Evaluate selection query results by comparing predicted and ground truth filtering results.
        
        Args:
            predicted_df: DataFrame with predicted filtering results
            evaluation_cols: List of columns to load from the ground truth CSV (including the filtering column)
            filtering_col: Name of the ground truth filtering column (boolean)

        Returns:
            Dictionary containing:
                - recall: Fraction of true positives correctly identified
                - accuracy: Fraction of correct predictions (both true positives and true negatives)
                - incorrect_predictions: List of incorrect predictions with their ground truth
        """
        ground_truth_df = pd.read_csv(f"{ground_truth_table_name}")[evaluation_cols].head(input_size)

        metrics = {
            'query_id': self.query_id,
            'class_name': self.class_name,
            'accuracy': 0.0,
            'recall': 0.0,
            'precision': 0.0,
            'f1_score': 0.0,
            'incorrect_predictions': [],
        }
        
        try:
            key_cols = [col for col in evaluation_cols if col != filtering_col]

            df = ground_truth_df.merge(predicted_df, on=key_cols, how='left', indicator=True)

            def get_category(row):
                if row[filtering_col] == True and row['_merge'] == 'both':
                    return 'TP'
                elif row[filtering_col] == False and row['_merge'] == 'left_only':
                    return 'TN'
                elif row[filtering_col] == False and row['_merge'] == 'both':
                    return 'FP'
                elif row[filtering_col] == True and row['_merge'] == 'left_only':
                    return 'FN'

            df['category'] = df.apply(get_category, axis=1)

            counts = df['category'].value_counts()
            tp = counts.get('TP', 0)
            tn = counts.get('TN', 0)
            fp = counts.get('FP', 0)
            fn = counts.get('FN', 0)

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics['accuracy'] = accuracy
            metrics['recall'] = recall
            metrics['precision'] = precision
            metrics['f1_score'] = f1_score

            # Collect incorrect predictions
            for i, row in df.iterrows():
                if row['category'] in ['FP', 'FN']:
                    metrics['incorrect_predictions'].append({
                        'row_index': i,
                        'predicted_row': predicted_df.iloc[i].to_dict() if i < len(predicted_df) else None,
                        'ground_truth_row': ground_truth_df.iloc[i].to_dict() if i < len(ground_truth_df) else None,
                        'predicted_value': row['_merge'] == 'both',
                        'ground_truth_value': row[filtering_col],
                        'category': row['category']
                    })
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    

class JoinEvaluator(BaseEvaluator):
    """Evaluator for join queries (table merging tasks)."""
    
    def evaluate(self, predicted_df, table_left_name: str, table_right_name: str, input_size: tuple[int, int], evaluation_table_name: str, evaluation_cols: List[str],
                    left_key: str, right_key:str) -> Dict[str, Any]:
        """
        Evaluate join query results by comparing predicted and ground truth merged tables.
        
        Args:
            predicted_df: DataFrame with predicted merged results
            evaluation_cols: List of columns to load from the ground truth CSV (including join key columns)
            join_key_cols: List of columns used as join keys

        Returns:
            Dictionary containing:
                - precision: Fraction of predicted rows that are correct in the join
                - recall: Fraction of true rows correctly identified in the join
                - accuracy: Fraction of correct predictions in the join
                - f1_score: Harmonic mean of precision and recall
                - incorrect_predictions: List of incorrect predictions with their ground truth
        """
        left_table_df = pd.DataFrame(pd.read_csv(f"{table_left_name}").head(input_size[0])[left_key])
        right_table_df = pd.DataFrame(pd.read_csv(f"{table_right_name}").head(input_size[1])[right_key])

        cross_df = left_table_df.merge(right_table_df, how='cross')
        ground_truth_df = pd.read_csv(f"{evaluation_table_name}")[[left_key, right_key]]

        metrics = {
            'query_id': self.query_id,
            'class_name': self.class_name,
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0,
            'f1_score': 0.0,
            'incorrect_predictions': [],
        }
        try:
            if predicted_df is None or len(predicted_df.columns) == 0:
                predicted_df = pd.DataFrame(columns=[left_key, right_key])

            predicted_gt_df = ground_truth_df.merge(predicted_df, on=[left_key, right_key], how='outer', indicator=True)

            tp = (predicted_gt_df['_merge'] == 'both').sum()
            fn = (predicted_gt_df['_merge'] == 'left_only').sum()
            fp = (predicted_gt_df['_merge'] == 'right_only').sum()
            tn = input_size[0] * input_size[1] - (tp+fn+fp)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1_score
            metrics['accuracy'] = accuracy
        except Exception as e:
            metrics['error'] = str(e)

        return metrics    
        
# def evaluate_derivation_query(query_id: str, predicted_df: pd.DataFrame, ground_truth_table_name: str, input_size: int,
#                               evaluation_cols: List[str], new_col_name: str, ground_truth_col_name: str) -> Dict[str, Any]:
#     evaluator = DerivationEvaluator(query_id=query_id, class_name='derivation')
#     return evaluator.evaluate(
#         predicted_df=predicted_df,
#         ground_truth_table_name=ground_truth_table_name,
#         input_size=input_size,
#         evaluation_cols=evaluation_cols,
#         new_col_name=new_col_name,
#         ground_truth_col_name=ground_truth_col_name
#     )

# def evaluate_selection_query(query_id: str, predicted_df: pd.DataFrame, ground_truth_table_name: str, input_size: int,
#                              evaluation_cols: List[str], filtering_col: str) -> Dict[str, Any]:
#     evaluator = SelectionEvaluator(query_id=query_id, class_name='selection')
#     return evaluator.evaluate(
#         predicted_df=predicted_df,
#         ground_truth_table_name=ground_truth_table_name,
#         input_size=input_size,
#         evaluation_cols=evaluation_cols,
#         filtering_col=filtering_col
#     )