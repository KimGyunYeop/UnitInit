from datasets import load_dataset, load_metric
from transformers import AdamW, get_scheduler, DebertaV2Tokenizer

from Debertav2_transformers import DebertaV2ForMaskedLM, DebertaV2Config, DebertaV2ForSequenceClassification
from utils import parse_args, tf_make_result_path, seed_fix

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm
import os
import json
import wandb
import argparse

MODEL_LIST = {
    "deberta":{
        "tokenizer" : DebertaV2Tokenizer,
        "model" : DebertaV2ForSequenceClassification,
        "config" : DebertaV2Config,
        "model_load_path" : "microsoft/deberta-v3-large"
    },
    "t5":{
        "tokenizer" : None,
        "model" : None,
        "config" : None 
    }
}

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

args = parse_args()
seed_fix(args.seed)
server_env = argparse.Namespace(**json.load(open("server_envs.json","r")))
device = "cuda:"+str(args.gpu)

model_type = None
for i in list(MODEL_LIST.keys()):
    if i in args.result_path.split("_"):
        model_type = i
if model_type is None:
    assert "result_path is must include modeltype!!"    

args.result_path = tf_make_result_path(args)

api = wandb.Api()

runs = api.runs(path="isnlp_lab/unit_init_glue")

a = set()
print("check duplicate")
for r in runs:
    if r.state == "crashed" or r.state == "failed":
        continue
    
    if r.name == args.result_path:
        assert "duplicate experiment"


model_utils = MODEL_LIST[model_type]
if args.model_load_path is not None:
    model_utils['model_load_path'] = args.model_load_path
    
if args.dev:
    args.result_path = "test_"+args.result_path
    
task = args.glue_task

dataset = load_dataset("glue", task , cache_dir=server_env.data_path)
metric = load_metric('glue', task , cache_dir=server_env.data_path)
print(dataset)

num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
tmp_matric = metric.compute(predictions=torch.Tensor([0,1]), references=torch.Tensor([0,1]))
    
tokenizer = model_utils["tokenizer"].from_pretrained(model_utils["model_load_path"])


def custom_collate_fn(batches):
    sentence1_key, sentence2_key = task_to_keys[task]
    if sentence2_key is None:
        texts = [batch[sentence1_key] for batch in batches]
    else:
        texts = [batch[sentence1_key] + "\n" + batch[sentence2_key] for batch in batches]
    
    # texts = [batch['sentence'] for batch in batches]
    labels = [batch['label'] for batch in batches]
    
    
    tokenized_inputs = tokenizer(
    texts, truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt", padding=True
    )
    
    tokenized_inputs["labels"] = torch.FloatTensor(labels)
    
    return tokenized_inputs


train_dataloader = DataLoader(dataset["train"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4, shuffle=True)

if "mnli" in args.glue_task:
    val_dataloader = DataLoader(dataset["test_matched"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4,)
    test_dataloader = DataLoader(dataset["test_mismatched"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4,)
    
else:
    try:
        val_dataloader = DataLoader(dataset["validation"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4,)
    except:
        val_dataloader = DataLoader(dataset["test"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4,)
    test_dataloader = DataLoader(dataset["test"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4,)

model = model_utils["model"].from_pretrained(model_utils["model_load_path"], num_labels=num_labels)
print(model.config)

if not args.no_add_linear:
    if args.add_linear_layer is None:
        if args.add_linear_num is None:
            args.add_linear_layer = range(model.config.num_hidden_layers)
        elif args.add_linear_num > 0:
            args.add_linear_layer = range(args.add_linear_num)
        else:
            args.add_linear_layer = range(model.config.num_hidden_layers + args.add_linear_num, model.config.num_hidden_layers)
    print(args.add_linear_layer)
    if args.add_position == "befdot":
        model.deberta.add_unit_init_before_dotpro(layer_num=args.add_linear_layer, head_indi=args.head_indi, init_type=args.init_type, act_type=args.act_type)
    elif args.add_position == "aftffnn1":
        pass

model.to(device)
print(model)


args.model_config = model.config
if not args.dev:
    wandb.init(project="unit_init_glue", entity="isnlp_lab",name="{}_{}".format(args.result_path, task), reinit=True)
    for i,_ in tmp_matric.items():
        wandb.define_metric("{}_dev_{}".format(task,i), summary="max")
        wandb.define_metric("{}_test_{}".format(task,i), summary="max")
    wandb.define_metric("{}_dev_{}".format(task,"acc"), summary="max")
    wandb.define_metric("{}_test_{}".format(task,"acc"), summary="max")
    wandb.config.update(args)

optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=[args.beta1,args.beta2], weight_decay=args.weight_decay, eps=args.eps)
scheduler = get_scheduler("linear", optimizer, args.warmup_steps, len(train_dataloader)* args.epoch)

for E in range(1, args.epoch+1):
    model.train()
    
    losses = []
    dl = tqdm(train_dataloader)
    for batches in dl:
        for idx in batches.keys():
            batches[idx] = batches[idx].to(device)
        
        out = model(**batches)
        
        out.loss.backward()
        losses.append(out.loss.item())
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        dl.set_description("loss="+str(out.loss.item()))

    print("train_loss = {}".format(sum(losses)/len(losses)))
    
    model.eval()
    losses = []
    best_dev_score = 0
    best_test_score = 0
    with torch.no_grad():
        for batches in tqdm(val_dataloader):
            for idx in batches.keys():
                batches[idx] = batches[idx].to(device)
                
            out = model(**batches)
            
            losses.append(out.loss.item())
            if num_labels == 1:
                metric.add_batch(predictions=out.logits, references=batches["labels"])
            else:   
                metric.add_batch(predictions=torch.argmax(out.logits, dim=-1), references=batches["labels"])
            
        final_score = metric.compute()

        print("dev_loss = {}".format(sum(losses)/len(losses)))
        print("dev", final_score)
        
        change_score_name = dict()
        for i,j in final_score.items():
            change_score_name["{}_dev_{}".format(task, i)] = j
        # change_score_name["{}_dev_{}".format(task, "acc")] = sum(pred_list == label_list)/pred_list.size()[0]
        change_score_name["epoch"] = E+1
        
        
        for batches in tqdm(test_dataloader):
            for idx in batches.keys():
                batches[idx] = batches[idx].to(device)
            
            out = model(**batches)

            losses.append(out.loss.item())
            if num_labels == 1:
                metric.add_batch(predictions=out.logits, references=batches["labels"])
            else:   
                metric.add_batch(predictions=torch.argmax(out.logits, dim=-1), references=batches["labels"])
            
        final_score = metric.compute()
    
    
    print("test", final_score)
    for i,j in final_score.items():
        change_score_name["{}_test_{}".format(task, i)] = j
    # change_score_name["{}_test_{}".format(task, "acc")] = sum(pred_list == label_list)/pred_list.size()[0]
    change_score_name["epoch"] = E+1
    
    if not args.dev:
        wandb.log(change_score_name)