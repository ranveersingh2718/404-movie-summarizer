# usage: python3 rouge.py reference_summaries.csv

import pandas as pd
import argparse
from rouge_score import rouge_scorer
import json
from typing import Dict, List, Tuple

def calculate_rouge_scores(generated: str, reference: str, rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL']) -> Dict[str, Dict[str, float]]:
    # init ROUGE scorer with stemming
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    
    scores = scorer.score(reference, generated) # calculate scores between reference and generated summaries
    
    # precision, recall, and F1 scores for each metric
    results = {}
    for rouge_type in rouge_types:
        results[rouge_type] = {
            'precision': scores[rouge_type].precision,
            'recall': scores[rouge_type].recall,
            'fmeasure': scores[rouge_type].fmeasure
        }
    
    return results

def process_csv(input_file: str) -> pd.DataFrame:
    # load CSV
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()
    
    # lists to store ROUGE scores for each metric
    rouge1_precision = []
    rouge1_recall = []
    rouge1_f1 = []
    rouge2_precision = []
    rouge2_recall = []
    rouge2_f1 = []
    rougeL_precision = []
    rougeL_recall = []
    rougeL_f1 = []
    
    # iterate each row to calculate ROUGE scores
    for idx, row in df.iterrows():
        generated = str(row['Generated_Summary']).strip()
        reference = str(row['Reference_Summary']).strip()
        
        # handle empty or invalid summaries
        if not generated or not reference or generated == 'nan' or reference == 'nan':
            rouge1_precision.append(0.0)
            rouge1_recall.append(0.0)
            rouge1_f1.append(0.0)
            rouge2_precision.append(0.0)
            rouge2_recall.append(0.0)
            rouge2_f1.append(0.0)
            rougeL_precision.append(0.0)
            rougeL_recall.append(0.0)
            rougeL_f1.append(0.0)
            continue
        
        # calculate scores for valid summary pair
        scores = calculate_rouge_scores(generated, reference)

        # store metric scores in lists
        rouge1_precision.append(scores['rouge1']['precision'])
        rouge1_recall.append(scores['rouge1']['recall'])
        rouge1_f1.append(scores['rouge1']['fmeasure'])
        rouge2_precision.append(scores['rouge2']['precision'])
        rouge2_recall.append(scores['rouge2']['recall'])
        rouge2_f1.append(scores['rouge2']['fmeasure'])
        rougeL_precision.append(scores['rougeL']['precision'])
        rougeL_recall.append(scores['rougeL']['recall'])
        rougeL_f1.append(scores['rougeL']['fmeasure'])
    
    # add score columns to the DataFrame
    df['ROUGE-1_Precision'] = rouge1_precision
    df['ROUGE-1_Recall'] = rouge1_recall
    df['ROUGE-1_F1'] = rouge1_f1
    df['ROUGE-2_Precision'] = rouge2_precision
    df['ROUGE-2_Recall'] = rouge2_recall
    df['ROUGE-2_F1'] = rouge2_f1
    df['ROUGE-L_Precision'] = rougeL_precision
    df['ROUGE-L_Recall'] = rougeL_recall
    df['ROUGE-L_F1'] = rougeL_f1
    
    return df

def print_summary_statistics(df: pd.DataFrame):
    print("\n=== ROUGE Score Summary Statistics ===\n")
    
    # define metric groups for output
    rouge_metrics = [
        ('ROUGE-1', ['ROUGE-1_Precision', 'ROUGE-1_Recall', 'ROUGE-1_F1']),
        ('ROUGE-2', ['ROUGE-2_Precision', 'ROUGE-2_Recall', 'ROUGE-2_F1']),
        ('ROUGE-L', ['ROUGE-L_Precision', 'ROUGE-L_Recall', 'ROUGE-L_F1'])
    ]
    
    # calculate and display stats for each metric type
    for metric_name, columns in rouge_metrics:
        print(f"{metric_name}:")
        for col in columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            metric_type = col.split('_')[1]
            print(f"  {metric_type:10s}: Mean={mean_val:.4f}, Std={std_val:.4f}, Min={min_val:.4f}, Max={max_val:.4f}")
        print()
    
    print(f"Total number of summaries evaluated: {len(df)}")
    
    empty_count = df['ROUGE-1_F1'].eq(0).sum()
    if empty_count > 0:
        print(f"Number of empty or invalid summaries: {empty_count}")

def main():
    # argument parser
    parser = argparse.ArgumentParser(description='Calculate ROUGE scores for movie summaries')
    parser.add_argument('input_file', type=str, help='Path to input CSV file')
    
    args = parser.parse_args()
    
    try:
        df = process_csv(args.input_file)
        
        # display detailed scores for each movie
        print("=== Detailed ROUGE Scores ===\n")
        for idx, row in df.iterrows():
            print(f"Title: {row['Title']}")
            print(f"  ROUGE-1: P={row['ROUGE-1_Precision']:.4f}, R={row['ROUGE-1_Recall']:.4f}, F1={row['ROUGE-1_F1']:.4f}")
            print(f"  ROUGE-2: P={row['ROUGE-2_Precision']:.4f}, R={row['ROUGE-2_Recall']:.4f}, F1={row['ROUGE-2_F1']:.4f}")
            print(f"  ROUGE-L: P={row['ROUGE-L_Precision']:.4f}, R={row['ROUGE-L_Recall']:.4f}, F1={row['ROUGE-L_F1']:.4f}")
            print()
        print_summary_statistics(df)
        
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()