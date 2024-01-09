import csv
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from itertools import repeat
import pandas as pd

import numpy as np
from scipy.stats import mannwhitneyu


from sklearn.metrics import confusion_matrix

#from stat_analysis.neurlps_dataloader import *
from stat_analysis.emnlp_dataloader import *
from stat_analysis.descriptive import analysis1_descriptive, analysis_bleu_score

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from bert_score import score

from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns


E1_PATH = "our_data/E1.csv"
E2_PATH = "our_data/E2.csv"
E3_PATH = "our_data/E3.csv"
E4_PATH = "our_data/E4.csv"

HUMAN_GENERATED_PATH = "our_data/tracing_question.csv"
GPT4_GENERATED_PATH = "our_data/final_result_for_paper_gpt4.csv"
GPT35_GENERATED_PATH = "our_data/final_result_for_paper_gpt3.5.csv"

TRUTH_PATH = "our_data/tracing_data_truth.csv"

FUZZY_RATE = 50


def analysis2_mannwhitneyu(df_human, df_gpt):
    # Sample data: Quality ratings for LLM-generated and human-generated questions.
    # Replace these with your actual data.

    df_human[["annotation1", "annotation3", "annotation4", "annotation5"]] = df_human[
        ["annotation1", "annotation3", "annotation4", "annotation5"]
    ].apply(pd.to_numeric)

    df_gpt[["annotation1", "annotation3", "annotation4", "annotation5"]] = df_gpt[
        ["annotation1", "annotation3", "annotation4", "annotation5"]
    ].apply(pd.to_numeric)
    question_list = [
        "1. Relevance to Learning Objectives: The question is relevant to the learning objectives of an introductory programming course.",
        "3. Clarity of the Question: The question presented is clear and the language used in the question easy to understand.",
        "4. Difficulty Level: The difficulty level of the question is appropriate for an introductory programming course.",
        "5. Relevance of the Question to the Given Code Snippet: The question is appropriately relate to the code snippet provided in the question.",
    ]
    for i, col in enumerate(
        ["annotation1", "annotation3", "annotation4", "annotation5"]
    ):

        # Get ratings for LLM-generated questions and human-generated questions
        llm_ratings = df_gpt[col].tolist()
        human_ratings = df_human[col].tolist()
        # print(i, col)
        # print(llm_ratings)
        # print(human_ratings)
        # Perform Mann-Whitney U test
        u_statistic, p_value = mannwhitneyu(
            llm_ratings, human_ratings, alternative="two-sided"
        )

        # Print results
        # question_idx = col[-1]
        print(f"Question: {question_list[i]}")
        print()
        print(f"U Statistic: {u_statistic}")
        print()
        print(f"P Value: {p_value}")
        print()


def analysis3_confusion_matrix(df_human, df_gpt):
    """
    Confusion Matrix: Create a confusion matrix based on experts’ guesses on whether the questions were AI or human-generated versus the actual origin. This could reveal whether experts can reliably distinguish between the two types of questions.
    To answer: How accurately can experts distinguish between LLM-generated and human-created questions based on their quality?
    ---
                Predicted GPT4 | Predicted Human
    Actual GPT4 
    Actual Human
    """
    mapping = lambda x: {"AI-generated": "gpt", "Human-created": "human"}[x]
    pred = (
        df_human["annotation6"].apply(mapping).tolist()
        + df_gpt["annotation6"].apply(mapping).tolist()
    )
    half_len = len(df_human["annotation6"].tolist())
    label = ["human"] * half_len + ["gpt"] * half_len

    # Compute confusion matrix
    cm = confusion_matrix(label, pred, labels=["gpt", "human"])

    # Convert confusion matrix to DataFrame for pretty printing
    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted GPT", "Predicted Human"],
        index=["Actual GPT", "Actual Human"],
    )

    # Convert DataFrame to LaTeX
    latex_cm = cm_df.to_latex()

    return latex_cm


def dict_to_latex(dict_input):
    # Convert dictionary to DataFrame
    df = pd.DataFrame(dict_input)
    df.columns = ["Q1", "Q3", "Q4", "Q5"]

    # Transpose the DataFrame to swap rows and columns
    df = df.transpose()
    # Convert DataFrame to LaTeX
    latex = df.to_latex(float_format="{:.2f}".format)

    return latex


def analyze(df_human, df_gpt):

    # analysis 1
    # human analysis
    analysis1, describe1 = analysis1_descriptive(df_human)
    latex1 = dict_to_latex(analysis1)
    print(latex1)
    print("---------------------------------")
    # gpt analysis
    analysis2, describe2 = analysis1_descriptive(df_gpt)
    latex2 = dict_to_latex(analysis2)
    print(latex2)

    print("=================================")
    # analysis 2
    analysis2 = analysis2_mannwhitneyu(df_human, df_gpt)
    print("=================================")

    # analysis 3
    analysis3 = analysis3_confusion_matrix(df_human, df_gpt)
    print(analysis3)

    # bleu analysis
    analysis_bleu = analysis_bleu_score(TRUTH_PATH)
    
    # bertscore analysis
    analysis_multi_bert()
    
def all_bertscore_box_plot(all_generated_list):
    human_list = [x["human"] for x in all_generated_list]
    gpt35_list = [x["gpt35"] for x in all_generated_list]
    gpt4_list = [x["gpt4"] for x in all_generated_list]

    p_gpt35, r_gpt35, f1_gpt35 = all_bertscore(gpt35_list, human_list)
    p_gpt4, r_gpt4, f1_gpt4 = all_bertscore(gpt4_list, human_list)
    # draw a box plot
    # Combine all scores in a single DataFrame
    df_gpt35 = pd.DataFrame({
        'Model': ['GPT-3.5'] * len(p_gpt35),
        'Precision': p_gpt35,
        'Recall': r_gpt35,
        'F1': f1_gpt35
    })

    df_gpt4 = pd.DataFrame({
        'Model': ['GPT-4'] * len(p_gpt4),
        'Precision': p_gpt4,
        'Recall': r_gpt4,
        'F1': f1_gpt4
    })

    # Combine GPT-3.5 and GPT-4 data
    df = pd.concat([df_gpt35, df_gpt4])

    # Reshape DataFrame for plotting
    df_melt = df.melt(id_vars='Model', value_vars=['Precision', 'Recall', 'F1'], var_name='Metric')
    # df_melt = df.melt(id_vars='Model', value_vars=['Precision', 'Recall', 'F1'], var_name='Metric')

    # Create boxplot
    fig = plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_melt, x='Model', y='value', hue='Metric')
    plt.title('BERT Scores Comparison')
    plt.ylabel('Score')
    # plt.show()
    # Adjust legend size
    plt.legend(prop={'size': 9}, loc='lower left')
    fig.savefig('figures/bert_boxplot.png')
    
def all_bertscore(candss, refs):

    all_candidate = []
    all_reference = []
    for cands, ref in zip(candss, refs):
        for cand, r in zip(cands, repeat(ref)):
            all_candidate.append(cand)
            all_reference.append(r)
    # compute bert score
    P, R, F1 = score(all_candidate, all_reference, lang='en', verbose=True, model_type='bert-base-uncased', rescale_with_baseline=True)
    

    return F1.tolist(), P.tolist(), R.tolist()
    # fig = plt.figure(1, figsize=(9, 6))

    # # Create an axes instance
    # ax = fig.add_subplot(111)
    # # ax.set_yscale('log')

    # # Create the boxplot
    # bp = ax.boxplot(F1)

    # # Save the figure
    # fig.savefig('figures/bert_boxplot.png', bbox_inches='tight')
    

def all_bleuscore_box_plot(candss, refs):
    tokenized_refss = [word_tokenize(ref) for ref in refs]
    tokenized_candss = [[word_tokenize(cand) for cand in cands] for cands in candss]
    # min_bleu_score = [
    #     min(sentence_bleu([ref], cand) for cand in cands)
    #     for ref, cands in zip(tokenized_refs, tokenized_candss)
    # ]
    # return sum(min_bleu_score) / len(min_bleu_score)
    all_bleu = []
    for tokenized_refs, tokenized_cands in zip(tokenized_refss, tokenized_candss):
        for ref, cand in zip(repeat(tokenized_refs), tokenized_cands):
            all_bleu.append(sentence_bleu(ref, cand))
    # all_bleu = list(filter(lambda x: x > 1e-1, all_bleu))
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)
    # ax.set_yscale('log')

    # Create the boxplot
    bp = ax.boxplot(all_bleu)

    # Save the figure
    fig.savefig('figures/bleu_boxplot.png', bbox_inches='tight')
    
def analysis_multi_bert():
    all_generated_list = read_all_generated_data(
        HUMAN_GENERATED_PATH, GPT35_GENERATED_PATH, GPT4_GENERATED_PATH
    )
    # human_list = [x["human"] for x in all_generated_list]
    # gpt35_list = [x["gpt35"] for x in all_generated_list]
    # gpt4_list = [x["gpt4"] for x in all_generated_list]

    # p_gpt35, r_gpt35, f1_gpt35 = all_bertscore(gpt35_list, human_list)
    # p_gpt4, r_gpt4, f1_gpt4 = all_bertscore(gpt4_list, human_list)
    all_bertscore_box_plot(all_generated_list)

if __name__ == "__main__":
    truth_dic = find_truth_dic(TRUTH_PATH)

    human_questions, gpt_questions = zip(
        *[
            distribute_annotation_to_source(batch_to_annotation(annotation, truth_dic))
            for annotation in read_annotated_data_to_batchs(
                [E1_PATH, E2_PATH, E3_PATH, E4_PATH]
            )
        ]
    )

    human_df = questions_to_dataframe(human_questions)
    gpt_df = questions_to_dataframe(gpt_questions)

    analyze(human_df, gpt_df)
    # analysis_multi_bert()
