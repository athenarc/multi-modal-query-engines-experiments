import palimpzest as pz
from palimpzest.constants import Model
import pandas as pd
from dotenv import load_dotenv

def count_true_positives(output_df, input_df):
    """
    Count true positives by comparing filtered positive reviews with original input
    
    Args:
        output_df: DataFrame containing filtered positive reviews
        input_df: Original input DataFrame with all reviews
    
    Returns:
        int: Number of true positives
    """
    true_positives = 0
    
    # Create a set of reviews from output_df for faster lookup
    classified_positive = set(output_df['review'].values)
    
    # Check each review that was classified as positive against original input
    for review in classified_positive:
        # Find the corresponding row in input_df
        original_row = input_df[input_df['review'] == review]
        if not original_row.empty:
            # Check if the original sentiment was actually positive
            if original_row['sentiment'].iloc[0] == 'positive':
                true_positives += 1
    
    return true_positives

def count_false_negatives(output_df, input_df):
    """
    Count false negatives - positive reviews that weren't captured by the filter
    
    Args:
        output_df: DataFrame containing filtered positive reviews
        input_df: Original input DataFrame with all reviews
    
    Returns:
        int: Number of false negatives
    """
    false_negatives = 0
    
    # Create a set of reviews that were classified as positive for faster lookup
    classified_positive = set(output_df['review'].values)
    
    # Check each review in the original input
    for _, row in input_df.iterrows():
        # If the review was actually positive but wasn't captured by our filter
        if row['sentiment'] == 'positive' and row['review'] not in classified_positive:
            false_negatives += 1
    
    return false_negatives

def count_false_positives(output_df, input_df):
    """
    Count false positives - reviews classified as positive but actually negative
    
    Args:
        output_df: DataFrame containing filtered positive reviews
        input_df: Original input DataFrame with all reviews
    
    Returns:
        int: Number of false positives
    """
    false_positives = 0
    
    # Create a set of reviews from output_df for faster lookup
    classified_positive = set(output_df['review'].values)
    
    # Check each review that was classified as positive against original input
    for review in classified_positive:
        # Find the corresponding row in input_df
        original_row = input_df[input_df['review'] == review]
        if not original_row.empty:
            # Check if the original sentiment was actually negative
            if original_row['sentiment'].iloc[0] == 'negative':
                false_positives += 1
    
    return false_positives

def count_true_negatives(output_df, input_df):
    """
    Count true negatives - negative reviews that were correctly not included in filtered output
    
    Args:
        output_df: DataFrame containing filtered positive reviews
        input_df: Original input DataFrame with all reviews
    
    Returns:
        int: Number of true negatives
    """
    true_negatives = 0
    
    # Create a set of reviews that were classified as positive for faster lookup
    classified_positive = set(output_df['review'].values)
    
    # Check each review in the original input
    for _, row in input_df.iterrows():
        # If the review was actually negative and wasn't included in our filtered output
        if row['sentiment'] == 'negative' and row['review'] not in classified_positive:
            true_negatives += 1
    
    return true_negatives


if __name__ == "__main__":

    load_dotenv()
    input_df= pd.read_csv("mine/imdb_sentiment_analysis/imdb_dataset.csv").head(1000)
    input_reviews = input_df["review"]

    dataset = pz.Dataset(pd.DataFrame(input_reviews))
    dataset = dataset.sem_filter("The review is positive")

    # config = pz.QueryProcessorConfig(policy=pz.MaxQuality())
    config = pz.QueryProcessorConfig(policy=pz.MinCost())
    # config = pz.QueryProcessorConfig(policy=pz.MinTime())
    # config = pz.QueryProcessorConfig(policy=pz.MaxQualityAtFixedCost)

    output = dataset.run(config)

    print("Cost:", output.execution_stats.total_execution_cost, "\n\n")
    print("Time:", output.execution_stats.total_execution_time, "\n\n")
    output_df = output.to_df(cols=["review"])
    print(output_df)

    fn_count = count_false_negatives(output_df, input_df)
    print(f"False Negatives: {fn_count}")

    tp_count = count_true_positives(output_df, input_df)
    print(f"True Positives: {tp_count}")

    fp_count = count_false_positives(output_df, input_df)
    print(f"False Positives: {fp_count}")

    tn_count = count_true_negatives(output_df, input_df)
    print(f"True Negatives: {tn_count}")

    accuracy = (tp_count + tn_count) / (tp_count + fn_count + fp_count + tn_count)
    print(f"Accuracy: {accuracy:.2f}")
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    print(f"Recall: {recall:.2f}")
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    print(f"Precision: {precision:.2f}")
