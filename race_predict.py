# global vars
import os
# important directories
cache_dir = '/tmp'
base_dir = os.path.join(os.getcwd(), 'data')
race_data_dir = os.path.join(base_dir, 'RACE')
output_dir = os.path.join(base_dir, 'results')
tuned_race_dir = os.path.join(base_dir, 'bert-race')

# pass a local dir to use a custom model
model_name_or_path = 'bert-base-uncased'

# import modules
import json
import numpy as np

from collections import defaultdict
from mc_transformers.mc_transformers import softmax
from mc_transformers.data_classes import DataCollatorWithIds, PredictionOutputWithIds
from mc_transformers.utils_mc import MultipleChoiceDataset, Split, processors

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

# building blocks
def get_building_blocks(model_name_or_path):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=4,
        finetuning_task='race',
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir,
    )
    train_dataset = MultipleChoiceDataset(
        data_dir=race_data_dir,
        tokenizer=tokenizer,
        task='race',
        max_seq_length=484,
        overwrite_cache=False,
        mode=Split.train,
    )
    eval_dataset = MultipleChoiceDataset(
        data_dir=race_data_dir,
        tokenizer=tokenizer,
        task='race',
        max_seq_length=484,
        overwrite_cache=False,
        mode=Split.dev,
    )
    data_collator = DataCollatorWithIds()
    return config, tokenizer, model, train_dataset, eval_dataset, data_collator

# metrics utilities
def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}

# results utilities
def pair_predictions_with_ids(results, data_collator):
    return PredictionOutputWithIds(
        predictions=results.predictions,
        label_ids=results.label_ids,
        example_ids=data_collator.example_ids,
        metrics=results.metrics,
    )

def print_metrics(metrics):
    for key in ["eval_loss", "eval_acc"]:
        if metrics.get(key) is not None:
            print(f'{key}: {metrics.get(key)}')

def parse_predictions(processor, example_ids, label_ids, predictions):
    # cast to avoid json serialization issues
    example_ids = [processor._decode_id(int(ex_id)) for ex_id in example_ids]
    label_ids = [int(lab) for lab in label_ids]
    label_id_map = {i: chr(ord('A') + int(label)) for i, label in enumerate(processor.get_labels())}

    predictions = softmax(predictions, axis=1)
    predictions_dict = defaultdict(list)

    for (ex_id, q_id), true_label, preds in zip(example_ids, label_ids, predictions):
        pred_dict = {
            "probs": preds.tolist(),
            "pred_label": label_id_map[np.argmax(preds)],
            "label": label_id_map[true_label],
        }
        predictions_dict[ex_id].append(pred_dict)

    full_ids = ['-'.join([c_id, qa_id]) for c_id, qa_id in example_ids]
    predictions = np.argmax(predictions, axis=1)
    predicted_labels = [label_id_map[id] for id in predictions]
    predictions_list = dict(zip(full_ids, predicted_labels))

    return predictions_dict, predictions_list

def save_predictions(processor, results, output_dir):
    predictions_dict, predictions_list = parse_predictions(
        processor,
        example_ids=results.example_ids,
        label_ids=results.label_ids,
        predictions=results.predictions,
    )
    output_nbest_file = os.path.join(
        output_dir,
        "nbest_predictions.json"
    )
    output_predictions_file = os.path.join(
        output_dir,
        "predictions.json"
    )

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(predictions_dict) + '\n')

    with open(output_predictions_file, "w") as writer:
        writer.write(json.dumps(predictions_list) + '\n')

training_args = TrainingArguments(
    do_train=True,                   # ops to do
    do_eval=True,                    # ops to do
    output_dir=output_dir,           # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=2,   # batch size per device during training
    per_device_eval_batch_size=8,    # batch size for evaluation
    gradient_accumulation_steps=8,   # gradient accumulation for training, will mulitiply train_bs
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    learning_rate=5.0e-05,
    fp16=True,                       # floating point setup
    fp16_opt_level="O1",
)

config, tokenizer, model, train_dataset, eval_dataset, data_collator = get_building_blocks(model_name_or_path)
# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator.collate,
)
# trainer.train()
# trainer.save_model()
# For convenience, we also re-save the tokenizer to the same directory,
# if trainer.is_world_master():
#    tokenizer.save_pretrained(training_args.output_dir)
# we can now train/eval/predict with the dataset, slow
# results = trainer.predict(eval_dataset)

# we can load an already fine-tuned model on race
config, tokenizer, model, train_dataset, eval_dataset, data_collator = get_building_blocks(tuned_race_dir)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator.collate,
)

# and run predictions
processor = processors['race']()

result = trainer.predict(eval_dataset)
if trainer.is_world_master():
    result = pair_predictions_with_ids(result, data_collator)
    print_metrics(result.metrics)
    save_predictions(processor, result, output_dir)
    results['eval'] = result
    data_collator.drop_ids()