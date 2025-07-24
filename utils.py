import random
import string
from collections import namedtuple
import random
import string
from collections import namedtuple
from typing import List
import pandas as pd
import numpy as np
import re

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])
ExampleDrop = namedtuple('ExampleDrop', ['question', 'answer'])
ExampleCodeGen = namedtuple('ExampleCodeGen', ['prompt', 'entrypoint', 'test', 'test_imports', 'test_list'])

LANG_TO_INSTRUCTIONS = {
    "en": """Solve this math problem.

{input}""",
    "bn": """এই গণিতের সমস্যাটি সমাধান করুন।

{input}""",
    "de": """Löse dieses Mathematikproblem.

{input}""",
    "es": """Resuelve este problema matemático.

{input}""",
    "fr": """Résolvez ce problème de mathématiques.

{input}""",
    "ja": """この数学の問題を解いてください。

{input}""",
    "ru": """Решите эту математическую задачу.

{input}""",
    "sw": """Suluhisha tatizo hili la hesabu.

{input}""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి.

{input}""",
    "th": """แก้ปัญหาคณิตศาสตร์นี้

{input}""",
    "zh": """解决这个数学问题。

{input}"""
}

LANG_TO_FPATH = lambda lang: f"dataset/mgsm/mgsm_{lang}.tsv"

ALL_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]


def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses 

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1) if match else ""

def score_mgsm(target: str, prediction: str) -> bool:
    if "." in prediction:
        prediction = prediction.rstrip("0").rstrip(".")

    target = target.replace(",", "")
    prediction = prediction.replace(",", "")

    return target == prediction


def get_lang_examples(lang: str) -> list[dict[str, str]]:
    fpath = LANG_TO_FPATH(lang)
    examples = []
    with open(fpath, mode='r', encoding='utf-8') as f:
        for line in f:
            inputs, targets = line.strip().split("\t")
            if "." in targets:
                raise ValueError(f"targets {targets} contains a decimal point.")
            # targets = int(targets.replace(",", ""))
            examples.append({"inputs": LANG_TO_INSTRUCTIONS[lang].format(input=inputs), "targets": targets, "lang": lang})
    return examples


def get_all_examples() -> list[dict[str, str]]:
    examples = []
    for lang in ALL_LANGUAGES:
        # if lang != "en":
        #     continue
        examples += get_lang_examples(lang)
    return examples


def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.
    
    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    # Convert data to a numpy array for easier manipulation
    data = np.array(data)

    # List to store the means of bootstrap samples
    bootstrap_means = []

    # Generate bootstrap samples and compute the mean for each sample
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # Convert bootstrap_means to a numpy array for percentile calculation
    bootstrap_means = np.array(bootstrap_means)

    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # Compute the median of the bootstrap means
    median = np.median(bootstrap_means)

    # Convert to percentages and format to one decimal place
    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    # Return the formatted string with confidence interval and median
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"


def load_questions(path: str, seed: int) -> List[Example]: #TODO: make answer
    """Load questions from csv file and return a list of Example namedtuples."""
    question_df = pd.read_csv(path)
    random.seed(seed)

    def shuffle_choices_and_create_example(row) -> Example:
        list_choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
        random.shuffle(list_choices)
        example = Example(row.Question, list_choices[0], list_choices[1], list_choices[2], list_choices[3],
                          list_choices.index(row['Correct Answer']) + 1)
        return example

    return [shuffle_choices_and_create_example(row) for _, row in question_df.iterrows()]

def load_questions_drop(path: str, seed: int) -> List[ExampleDrop]: #TODO: make answer
    """Load questions from csv file and return a list of Example namedtuples."""
    question_df = pd.read_csv(path)
    random.seed(seed)

    def compose_data(row) -> Example:
        example_drop = ExampleDrop(row.context, row.completion)
        return example_drop

    return [compose_data(row) for _, row in question_df.iterrows()]

def load_questions_gsm8k(path: str, seed: int) -> List[ExampleDrop]: #TODO: make answer
    """Load questions from csv file and return a list of Example namedtuples."""
    question_df = pd.read_csv(path)
    random.seed(seed)

    def compose_data(row) -> Example:
        example_drop = ExampleDrop(row.question, row.answer)
        return example_drop

    return [compose_data(row) for _, row in question_df.iterrows()]

def load_questions_hotpotqa(path: str, seed: int) -> List[ExampleDrop]: #TODO: make answer
    """Load questions from csv file and return a list of Example namedtuples."""
    question_df = pd.read_csv(path)
    random.seed(seed)

    def compose_data(row) -> Example:
        example_drop = ExampleDrop(f"Context: {row.context}\nQuestion: {row.question}", row.answer)
        return example_drop

    return [compose_data(row) for _, row in question_df.iterrows()]

def load_questions_mbpp(path: str, seed: int) -> List[ExampleCodeGen]: #TODO: make answer
    """Load questions from csv file and return a list of ExampleCodeGen namedtuples."""
    question_df = pd.read_csv(path)
    random.seed(seed)

    def compose_data(row) -> ExampleCodeGen:
        example_drop = ExampleCodeGen(prompt=row.prompt, entrypoint=row.entry_point, test=row.test, test_imports=row.test_imports, test_list=row.test_list)
        return example_drop

    return [compose_data(row) for _, row in question_df.iterrows()]

