import csv
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd
import re

@dataclass
class Question_Label:
    question: str = ""
    source: str = "" # human or gpt
    
    annotation1: str = "" # 1. Relevance to Learning Objectives: The question is relevant to the learning objectives of an introductory programming course.
    annotation2: str = "" # 2. Tracing or not: Is this a tracing question?
    annotation3: str = "" # 3. Clarity of the Question: The question presented is clear and the language used in the question easy to understand.
    annotation4: str = "" # 4. Difficulty Level: The difficulty level of the question is appropriate for an introductory programming course.
    annotation5: str = "" # 5. Relevance of the Question to the Given Code Snippet: The question is appropriately relate to the code snippet provided in the question.
    annotation6: str = "" # 6. Ability to Distinguish Human-Authored from Automatically Generated Questions: Can you tell if the question is human-authored or automatically generated?
    annotation7: str = "" # I think this is a better tracing question. (Put an "X" in the corresponding cell for the better question)

@dataclass
class Annotation:
    requirement: str = ""
    code: str = ""
    question1: Question_Label = Question_Label()
    question2: Question_Label = Question_Label()

def split_questions(s: str) -> List[str]:
    # The pattern looks for a digit followed by a dot and a space, which is the start of each question.
    # The pattern is enclosed in parentheses to include the match in the split list.
    # The look-ahead assertion (?=) ensures that the split includes the text that follows the pattern.
    split_list = re.split("(?=\d+\. )", s)

    # The first element may be an empty string, so it should be removed if present.
    return [remove_num_prefix(x) for x in split_list if x]


def remove_num_prefix(question: str) -> str:
    return re.sub("^\d+\. ", "", question).strip()

def batch_to_annotation(batch: List[str], truth_table: Dict[str, Any]) -> Annotation:
    REQUIREMENT_KEY = '\ufeffRequirement'
    CODE_KEY = 'Code'
    question1_text = batch[0]['Question1']
    question2_text = batch[0]['Question2']
    requirement_text = batch[0][REQUIREMENT_KEY]
    code_text = batch[0][CODE_KEY]
    key = requirement_text + "$$$" + code_text

    candidates = truth_table[key]
    source1 = ""
    source2 = ""
    # question1_source
    for question in candidates:
        if question1_text in question:
            source1 = question[question1_text]
            break
    # question2_source
    for question in candidates:
        if question2_text in question:
            source2 = question[question2_text]
            break
        
    annotation = Annotation(
        requirement=requirement_text,
        code=code_text,
        question1=Question_Label(
            question=question1_text,
            source=source1,
            annotation1=batch[1]['Question1'],
            annotation2=batch[2]['Question1'],
            annotation3=batch[3]['Question1'],
            annotation4=batch[4]['Question1'],
            annotation5=batch[5]['Question1'],
            annotation6=batch[6]['Question1'],
            annotation7=batch[7]['Question1'],
        ),
        question2=Question_Label(
            question=question2_text,
            source=source2,
            annotation1=batch[1]['Question2'],
            annotation2=batch[2]['Question2'],
            annotation3=batch[3]['Question2'],
            annotation4=batch[4]['Question2'],
            annotation5=batch[5]['Question2'],
            annotation6=batch[6]['Question2'],
            annotation7=batch[7]['Question2'],
        )
    )
    return annotation

def distribute_annotation_to_source(annotation: Annotation) -> Tuple[Question_Label, Question_Label]:
    # return (human, gpt)
    question1 = annotation.question1
    question2 = annotation.question2
    if question1.source == "human":
        return question1, question2
    else:
        return question2, question1
    
    
def line_is_empty(line):
    return line['\ufeffRequirement'] == "" and line['Code'] == "" and line['Question1'] == "" and line['Question2'] == ""

def read_annotated_data_to_batchs(paths: list):
    for path in paths:
        with open(path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            batch = []
            for i, row in enumerate(reader):
                batch.append(row)
                # if i % 8 == 0 and i != 0:
                if line_is_empty(row):
                    yield batch
                    batch = []

def read_all_generated_data(human_path: str, gpt35_path: str, gpt4_path: str):
    result = []
    with open(human_path, newline="") as human_csvfile, open(gpt35_path, newline="") as gpt35_csvfile, open(gpt4_path, newline="") as gpt4_csvfile:
        human_reader = csv.DictReader(human_csvfile)
        gpt35_reader = csv.DictReader(gpt35_csvfile)
        gpt4_reader = csv.DictReader(gpt4_csvfile)        
        for i, (human_row, gpt35_row, gpt4_row) in enumerate(zip(human_reader, gpt35_reader, gpt4_reader)):
            result.append(
                {
                    "human": human_row["Question"],
                    "gpt35": split_questions(gpt35_row["Question"]),
                    "gpt4": split_questions(gpt4_row["Question"]),
                }
            )
    return result

def find_truth_dic(path):
    truth = defaultdict(list)
    # key is Requirement + "$$$" + Code
    # value is {question1: human, question2: gpt ...}
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            key = row['Requirement'] + "$$$" + row['Code']
            value = {row['GPT4_Question']: "gpt", row['Human_Question']: "human"}
            truth[key].append(value)
    return truth

def questions_to_dataframe(questions: List[Question_Label]):

    df = pd.DataFrame.from_records([question.__dict__ for question in questions])

    return df