import argparse
import os
import os.path
import pandas as pd
from typing import List
from evaluate import load
from rouge_score import rouge_scorer
from sklearn.metrics import confusion_matrix

def rouge(predictions: List[str], references: List[str], r_type: str = ""):
    precision = []
    recall = []
    f1 = []
    scorer = rouge_scorer.RougeScorer([r_type], use_stemmer=True)
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        precision.append(score[r_type].precision)
        recall.append(score[r_type].recall)
        f1.append(score[r_type].fmeasure)

    f1 = sum(f1) / len(f1)
    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    return f1, precision, recall


def bertS(predictions: List[str], references: List[str]):
    bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=predictions, references=references, lang="en"
    )

    f1 = sum(results["f1"])/len(results["f1"])
    precision = sum(results["precision"])/len(results["precision"])
    recall = sum(results["recall"])/len(results["recall"])
    return f1, precision, recall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Path to prediction file")
    parser.add_argument("--gold_file", type=str, required=True, help="Path to gold data file")
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to the output file with scores")
    parser.add_argument("--qa_pair_type", type=str,default=None, help="Specific type of the question-answer pair")

    args = parser.parse_args()
    os.makedirs(args.output_file_path, exist_ok=True)
    output_filename = os.path.join(args.output_file_path, 'scores.txt' if args.qa_pair_type is None else f'scores_{args.qa_pair_type}.txt')
    output_file = open(output_filename, 'w')

    gold_df = pd.read_json(args.gold_file)
    pred_df = pd.read_csv(args.pred_file, index_col=0)
    
    if len(gold_df) != len(pred_df):
        raise ValueError("The lengths of references and predictions do not match.")
    
    merged = gold_df.merge(pred_df, on='instance_id', how='left')
    if args.qa_pair_type is not None:
        merged = merged[merged['qa_pair_type'] == args.qa_pair_type]
    references = merged['answer'].tolist()
    predictions = merged['answer_pred'].tolist()
    if args.qa_pair_type is None:
        y_true = [ 1 if qa_pair_type == "unanswerable" else 0 for qa_pair_type in merged['qa_pair_type'].tolist() ]
        y_pred = [ 1 if answer_pred =="It is not possible to answer this question based only on the provided data." else 0 for answer_pred in predictions ]
        print("Analyzing performance of auxilary router")
        output_file.write("Analyzing performance of auxilary router\n")
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        output_file.write("TN: "+str(tn)+"\n")
        print(f"TN: {tn}")
        output_file.write("FP: "+str(fp)+"\n")
        print(f"FP: {fp}")
        output_file.write("FN: "+str(fn)+"\n")
        print(f"FN: {fn}")
        output_file.write("TP: "+str(tp)+"\n")
        print(f"TP: {tp}")
        output_file.write("Accuracy: "+str((tn+tp)/(tn+fp+fn+tp))+"\n")
        print(f"Accuracy: {(tn+tp)/(tn+fp+fn+tp)}")
        output_file.write("F1: "+str((2*tp)/(2*tp+fp+fn))+"\n")
        print(f"F1: {(2*tp)/(2*tp+fp+fn)}")
        output_file.write("Precision: "+str(tp/(tp+fp))+"\n")
        print(f"Precision: {tp/(tp+fp)}")
        output_file.write("Recall: "+str(tp/(tp+fn))+"\n")
        print(f"Recall: {tp/(tp+fn)}")

    rouge1_score_f1, rouge1_score_precision, rouge1_score_recall = rouge(predictions, references, "rouge1")
    rougeL_score_f1, rougeL_score_precision, rougeL_score_recall = rouge(predictions, references, "rougeL")
    bert_score_f1, bert_score_precision, bert_score_recall = bertS(predictions, references)

    output_file.write("rouge1.f1: "+str(rouge1_score_f1)+"\n")
    print(f"rouge1.f1: {rouge1_score_f1}")
    output_file.write("rouge1.precision: "+str(rouge1_score_precision)+"\n")
    print(f"rouge1.precision: {rouge1_score_precision}")
    output_file.write("rouge1.recall: "+str(rouge1_score_recall)+"\n")
    print(f"rouge1.recall: {rouge1_score_recall}")

    output_file.write("rougeL.f1: "+str(rougeL_score_f1)+"\n")
    print(f"rougeL.f1: {rougeL_score_f1}")
    output_file.write("rougeL.precision: "+str(rougeL_score_precision)+"\n")
    print(f"rougeL.precision: {rougeL_score_precision}")
    output_file.write("rougeL.recall: "+str(rougeL_score_recall)+"\n")
    print(f"rougeL.recall: {rougeL_score_recall}")

    output_file.write("bertS.f1: "+str(bert_score_f1)+"\n")
    print(f"bertS.f1: {bert_score_f1}")
    output_file.write("bertS.precision: "+str(bert_score_precision)+"\n")
    print(f"bertS.precision: {bert_score_precision}")
    output_file.write("bertS.recall: "+str(bert_score_recall)+"\n")
    print(f"bertS.recall: {bert_score_recall}")

    output_file.close()

if __name__ == "__main__":
    main()
    

      
