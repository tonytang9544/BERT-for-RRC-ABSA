
    
from dataclasses import dataclass
import json

@dataclass
class InputFeatures(object):
    """A single set of features of data."""

    input_ids: object
    segment_ids: object
    input_mask: object
    start_position: int
    end_position: int


def read_json_examples(input_file, is_training=True):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    
    questions = []
    contexts = []
    question_ids = []
    answer_texts = []
    start_positions = []

    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph["qas"]:
                contexts.append(paragraph["context"])
                questions.append(qa["question"])
                question_ids.append(qa["id"])
                answer_texts.append(qa["answers"][0]["text"])
                start_positions.append(qa["answers"][0]["answer_start"])

    return contexts, questions, question_ids, answer_texts, start_positions

def read_json_examples_test():
    contexts, questions, answer_ids, answer_text = read_json_examples("../data/rrc/laptop/train.json")
    print(contexts[0], questions[0], answer_ids[0], answer_text[0])

if __name__ == "__main__":
    read_json_examples_test()