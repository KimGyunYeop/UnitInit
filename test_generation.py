from datasets import load_dataset, load_metric
from transformers import AdamW, get_scheduler, DebertaV2Tokenizer, T5Tokenizer

from T5_transformers import T5ForConditionalGeneration, T5Config
from utils import parse_args, gen_make_result_path, seed_fix

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm
import os
import json
import wandb
import argparse

MODEL_LIST = {
    "t5":{
        "tokenizer" : T5Tokenizer,
        "model" : T5ForConditionalGeneration,
        "config" : T5Config,
        "model_load_path" : "google-t5/t5-base"
    }
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

args.result_path = gen_make_result_path(args)

api = wandb.Api()

runs = api.runs(path="isnlp_lab/unit_init_generation")

task = args.generation_task

# print("{}_{}".format(args.result_path, task))
# print("check duplicate")
# for r in runs:
#     if r.state == "crashed" or r.state == "failed":
#         continue
    
#     if r.name == "{}_{}".format(args.result_path, task):
#         raise "duplicate experiment"


model_utils = MODEL_LIST[model_type]
if args.model_load_path is not None:
    model_utils['model_load_path'] = args.model_load_path
    
if args.dev:
    args.result_path = "test_"+args.result_path
    
if task == "cnndm":
    dataset = load_dataset("cnn_dailymail", "3.0.0" , cache_dir=server_env.data_path)
    metric = load_metric('rouge' , cache_dir=server_env.data_path)
elif task == "wmt_en_ro":
    dataset = load_dataset("wmt16", "ro_en" , cache_dir=server_env.data_path)
    metric = load_metric('bleu' , cache_dir=server_env.data_path)
print(dataset)

tmp_matric = metric.compute(predictions=["hello there", "general kenobi"], references=["hello there", "general kenobi"])
print(tmp_matric)
    
tokenizer = model_utils["tokenizer"].from_pretrained(model_utils["model_load_path"])
pad_token_id = tokenizer.pad_token_id
print(pad_token_id)

def custom_collate_fn(batches):
    
    if task == "cnndm":
        sentences = [batch["article"] for batch in batches]
        labels = [batch["highlights"] for batch in batches]
    elif task == "wmt_en_ro":
        sentences = [batch["translation"]["en"] for batch in batches]
        labels = [batch["translation"]["ro"] for batch in batches]
    
    
    tokenized_inputs = tokenizer(
        sentences, truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt", padding=True
    )
    
    tokenized_labels = tokenizer(
        labels, truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt", padding=True
    )
    
    tokenized_labels["input_ids"].masked_fill_(tokenized_labels["input_ids"] == pad_token_id, -100)
    tokenized_inputs["labels"] = tokenized_labels["input_ids"]
    # tokenized_inputs["decoder_attention_mask"] = tokenized_labels["attention_mask"]
    
    return tokenized_inputs


train_dataloader = DataLoader(dataset["train"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4, shuffle=True)
val_dataloader = DataLoader(dataset["validation"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4)
test_dataloader = DataLoader(dataset["test"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4,)
    
    
model = model_utils["model"].from_pretrained(model_utils["model_load_path"])


if not args.no_add_linear:
    if args.add_linear_layer is None:
        if args.add_linear_num is None:
            args.add_linear_layer = range(model.config.num_hidden_layers)
        elif args.add_linear_num > 0:
            args.add_linear_layer = range(args.add_linear_num)
        else:
            args.add_linear_layer = range(model.config.num_hidden_layers + args.add_linear_num, model.config.num_hidden_layers)
    
    if args.add_position == "befdot":
        model.add_unit_init_before_dotpro(layer_num=args.add_linear_layer, head_indi=args.head_indi, init_type=args.init_type, act_type=args.act_type)
    elif args.add_position == "afterffnn":
        model.add_unit_init_after_ffnn(layer_num=args.add_linear_layer, init_type=args.init_type, act_type=args.act_type)
    elif args.add_position == "both":
        model.add_unit_init_after_ffnn(layer_num=args.add_linear_layer, init_type=args.init_type, act_type=args.act_type)
        model.add_unit_init_before_dotpro(layer_num=args.add_linear_layer, head_indi=args.head_indi, init_type=args.init_type, act_type=args.act_type)
        
    elif args.add_position == "aftffnn1":
        model.add_unit_init_after_ffnn1(layer_num=args.add_linear_layer, init_type=args.init_type, act_type=args.act_type)
    elif args.add_position == "aftffnn2":
        model.add_unit_init_after_ffnn2(layer_num=args.add_linear_layer, init_type=args.init_type, act_type=args.act_type)

if args.adapter:
    for name, param in model.named_parameters():
        if "added" not in name and "classifi" not in name:
            param.requires_grad_(requires_grad=False)
            #param.requires_grad=False
        else:
            param.requires_grad_(requires_grad=True)
    
    for name, param in model.named_parameters():
        print(param.requires_grad,"\t/\t",name)

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

for name, param in model.named_parameters():
    if "added" in name:
        print(name, param)
        break


for E in range(1, args.epoch+1):
    model.train()
    
    losses = []
    dl = tqdm(train_dataloader)
    for batches in dl:
        for idx in batches.keys():
            batches[idx] = batches[idx].to(device)
        
        out = model(**batches)
        print(out)
        print(out.loss)
        assert 0
        out.loss.backward()
        losses.append(out.loss.item())
        
        if not args.adapter:
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

    for name, param in model.named_parameters():
        if "added" in name:
            print(name, param)
            break


        
    #     for batches in tqdm(test_dataloader):
    #         for idx in batches.keys():
    #             batches[idx] = batches[idx].to(device)
            
    #         out = model(**batches)

    #         # losses.append(out.loss.item())
    #         if num_labels == 1:
    #             metric.add_batch(predictions=out.logits, references=batches["labels"])
    #         else:   
    #             metric.add_batch(predictions=torch.argmax(out.logits, dim=-1), references=batches["labels"])
            
    #     final_score = metric.compute()
    
    
    # print("test", final_score)
    # for i,j in final_score.items():
    #     change_score_name["{}_test_{}".format(task, i)] = j
    # change_score_name["{}_test_{}".format(task, "acc")] = sum(pred_list == label_list)/pred_list.size()[0]
    change_score_name["epoch"] = E+1
    
    if not args.dev:
        wandb.log(change_score_name)