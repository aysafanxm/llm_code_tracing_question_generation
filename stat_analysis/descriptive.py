import pandas as pd
import numpy as np
import csv

from scipy.stats import mode
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score
from nltk.tokenize import word_tokenize

from tabulate import tabulate


def analysis1_descriptive(df):
    """
    Start with descriptive statistics of the evaluation ratings (mean, median, mode, range, standard deviation) for each of the criteria (1, 3, 4, 5) for both LLM-generated and human-created questions. This will provide a basic understanding of the overall quality and characteristics of the questions.
    To Answer: What are the general characteristics of the quality ratings for LLM-generated and human-created questions? Do the ratings for the two groups display significantly different central tendencies or dispersion?
    """
    df[["annotation1", "annotation3", "annotation4", "annotation5"]] = df[
        ["annotation1", "annotation3", "annotation4", "annotation5"]
    ].apply(pd.to_numeric)

    # Store the analyses in a dictionary
    analysis = {}

    for col in ["annotation1", "annotation3", "annotation4", "annotation5"]:
        # Calculate statistics
        analysis[col] = {
            "mean": np.mean(df[col]),
            "median": np.median(df[col]),
            "mode": mode(df[col]).mode,
            "range": np.ptp(df[col]),
            "std_dev": np.std(df[col]),
        }

    # Use df.describe() to get a summary of statistics
    describe = df[
        ["annotation1", "annotation3", "annotation4", "annotation5"]
    ].describe()

    return analysis, describe


def analysis_bleu_score(truth_path):
    refs = []
    cands = []
    with open(truth_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            refs.append(row["Human_Question"])
            cands.append(row["GPT4_Question"])

    # tokenize the sentences using NLTK's tokenizer
    tokenized_refs = [word_tokenize(ref) for ref in refs]
    tokenized_cands = [word_tokenize(cand) for cand in cands]

    # compute bleu score
    bleu_scores = [
        sentence_bleu([ref], cand) for ref, cand in zip(tokenized_refs, tokenized_cands)
    ]

    # compute rouge score
    rouge = Rouge()
    rouge_scores = [rouge.get_scores(cand, ref)[0] for cand, ref in zip(cands, refs)]
    rouge_1_f = [score["rouge-1"]["f"] for score in rouge_scores]
    rouge_2_f = [score["rouge-2"]["f"] for score in rouge_scores]
    rouge_l_f = [score["rouge-l"]["f"] for score in rouge_scores]

    # compute bert score
    P, R, F1 = score(
        cands,
        refs,
        lang="en",
        verbose=True,
        model_type="bert-base-uncased",
        rescale_with_baseline=True,
    )
    # print("BLEU score:", sum(bleu_scores)/len(bleu_scores))
    # print("ROUGE-1 F-score:", sum(rouge_1_f)/len(rouge_1_f))
    # print("ROUGE-2 F-score:", sum(rouge_2_f)/len(rouge_2_f))
    # print("ROUGE-L F-score:", sum(rouge_l_f)/len(rouge_l_f))
    # print('BERTScore Precision Mean: ', P.mean().item())
    # print('BERTScore Recall Mean: ', R.mean().item())
    # print('BERTScore F1-Score Mean: ', F1.mean().item())

    data = {
        "Metric": [
            "BLEU",
            "ROUGE-1 F-score",
            "ROUGE-2 F-score",
            "ROUGE-L F-score",
            "BERTScore Precision",
            "BERTScore Recall",
            "BERTScore F1",
        ],
        "Score": [
            round(sum(bleu_scores) / len(bleu_scores), 3),
            round(sum(rouge_1_f) / len(rouge_1_f), 3),
            round(sum(rouge_2_f) / len(rouge_2_f), 3),
            round(sum(rouge_l_f) / len(rouge_l_f), 3),
            round(P.mean().item(), 3),
            round(R.mean().item(), 3),
            round(F1.mean().item(), 3),
        ],
    }

    table = tabulate(data, headers="keys", tablefmt="latex")

    print(table)
