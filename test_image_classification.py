from datasets import load_dataset, load_metric
from transformers import AdamW, get_scheduler 
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoImageProcessor, ConvNextForImageClassification

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils import parse_args

from tqdm import tqdm
import os
import json
import wandb
import argparse


MODEL_LIST = {
    "vit":{
        "model" : ViTForImageClassification,
        "image_processor" : ViTImageProcessor,
        "model_load_path" : "google/vit-base-patch16-224-in21k"
    },
    "convnext":{
        "model" : ConvNextForImageClassification,
        "image_processor" : AutoImageProcessor,
        "model_load_path" : "facebook/convnext-tiny-224"
    }
}

dataset_to_imagedict = { 
    "cifar10": 'img',
    "cifar100": 'img',
    "imagenet-1k": 'image', 
}
dataset_to_labeldict = { 
    "cifar10": 'label',
    "cifar100": 'fine_label',
    "imagenet-1k": 'label', 
}


args = parse_args()
server_env = argparse.Namespace(**json.load(open("server_envs.json","r"))) # "data_path": "/home/nlplab/hdd1/gyop/dataset"
device = "cuda:"+str(args.gpu) 

model_type = None
for i in list(MODEL_LIST.keys()): # "vit", "convnext"
    if i in args.result_path.split("_"):
        model_type = i
if model_type is None:
    assert "result path must include model type!"    

model_utils = MODEL_LIST[model_type] # model, image_processor, model_load_path
if args.model_load_path is not None:
    model_utils['model_load_path'] = args.model_load_path
    
if args.dev:
    args.result_path = "test_"+args.result_path

##################################################
    
dataset_name = args.image_classification_dataset 
dataset_labeldict = dataset_to_labeldict[dataset_name]
dataset_imagedict = dataset_to_imagedict[dataset_name]
dataset = load_dataset(dataset_name , cache_dir=server_env.data_path, use_auth_token=True) 
metric = load_metric("accuracy") 
# tmp_metric = metric.compute(predictions=torch.Tensor([0,1]), references=torch.Tensor([0,1]))
# print(dataset)
# print()
# print(dataset["train"].features)
# print()
# print(dataset["train"][0])
# print()
# assert 0

processor = model_utils["image_processor"].from_pretrained(model_utils["model_load_path"])

def custom_collate_fn(batches):
    inputs = processor([batch[dataset_imagedict] for batch in batches], return_tensors='pt') #inputs['pixel_values']
    inputs['labels'] = torch.tensor([batch[dataset_labeldict] for batch in batches])
    
    return inputs

train_dataloader = DataLoader(dataset["train"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4, shuffle=True)
try:
    val_dataloader = DataLoader(dataset["validation"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4)
except:
    val_dataloader = DataLoader(dataset["test"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4)
test_dataloader = DataLoader(dataset["test"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4)

##################################################

labels = dataset['train'].features[dataset_labeldict].names

model = model_utils["model"].from_pretrained(
    model_utils["model_load_path"], 
    num_labels=len(labels),
    #id2label={str(i): c for i, c in enumerate(labels)},
    #label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True #####
    )

if not args.no_add_linear:
    if args.add_linear_layer is None:
        if args.add_linear_num is None:
            args.add_linear_layer = range(model.config.num_hidden_layers)
        elif args.add_linear_num > 0:
            args.add_linear_layer = range(args.add_linear_num)
        else:
            args.add_linear_layer = range(model.config.num_hidden_layers - args.add_linear_num, model.config.num_hidden_layers)
        
    model.deberta.add_eye_linear(random_init=args.random_init, add_linear_layer=args.add_linear_layer, share_eye=args.enc_dec_share, head_indi=args.head_indi)

model.to(device)

optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=[args.beta1,args.beta2], weight_decay=args.weight_decay, eps=args.eps)
scheduler = get_scheduler("linear", optimizer, args.warmup_steps, len(train_dataloader)* args.epoch)

##################################################

args.model_config = model.config
'''
if not args.dev:
    wandb.init(project=model_type+"_"+dataset_name, entity="gyunyeop",name="{}_{}".format(args.model_path, dataset_name), reinit=True)
    for i,_ in tmp_metric.items():
        wandb.define_metric("{}_dev_{}".format(dataset_name,i), summary="max")
        wandb.define_metric("{}_test_{}".format(dataset_name,i), summary="max")
    wandb.define_metric("{}_dev_{}".format(dataset_name,"acc"), summary="max")
    wandb.define_metric("{}_test_{}".format(dataset_name,"acc"), summary="max")
    wandb.config.update(args)
'''

##################################################

print("Model Type:", model_type)
print("Dataset Name:", dataset_name)

for E in range(1, args.epoch+1):
    model.train()
    
    losses = []
    for batches in tqdm(train_dataloader):
        for idx in batches.keys():
            batches[idx] = batches[idx].to(device)
        # print(batches)
        # print(batches["pixel_values"].shape) # torch.Size([bs, 3, 224, 224])
        # print(batches["labels"].shape) # torch.Size([bs])        

        out = model(**batches)
        #print(out.logits.shape) # torch.Size([bs, num_labels])
        
        out.loss.backward()
        losses.append(out.loss.item())
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    print("train_loss = {}".format(sum(losses)/len(losses)))
    
    ##########

    model.eval()

    losses = []
    pred_list = []
    label_list = []
    with torch.no_grad():
        for batches in tqdm(val_dataloader):
            for idx in batches.keys():
                batches[idx] = batches[idx].to(device)
            
            out = model(**batches)

            losses.append(out.loss.item())
            metric.add_batch(predictions=torch.argmax(out.logits, dim=-1), references=batches["labels"])
            # pred_list.append(torch.argmax(out.logits, dim=-1))
            # label_list.append(batches["labels"])
            
        final_score = metric.compute()

    print("dev_loss = {}".format(sum(losses)/len(losses)))
    print("dev", final_score)