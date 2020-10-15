cache_dir = '/tmp'

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForQuestionAnswering
)

model = AutoModelForQuestionAnswering.from_pretrained(
    'deepset/roberta-base-squad2',
    cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(
    'deepset/roberta-base-squad2',
    cache_dir=cache_dir
)

nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

context = r"""
Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script.
"""

result = nlp(question="What is extractive question answering?", context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")

result = nlp(question="What is a good example of a question answering dataset?", context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
