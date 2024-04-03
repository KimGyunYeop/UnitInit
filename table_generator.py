import wandb
import pandas as pd
import numpy as np
import os
import json

pd.set_option('display.max_colwidth', None)

glue_keywords={
    "glue_task":"wnli",
    "model_type":"",
    "add_position":"",
    "act_type":"",
    "init_type":"",
    "no_add_linear":"",
    "add_linear_num":""
}
glue_tunings = {
    "learning_rate":["1.5e-5", "3e-5", "5e-5", "1e-4"],
    "warmup_steps":["50","500"]
}
glue_measurement = {
    "cola" : "dev_mat", 
    "sst2" : "dev_acc", 
    "mrpc" : "dev_acc", 
    "stsb" : "dev_pea", 
    "qqp" : "dev_acc", 
    "mnli" : "dev_acc",  
    "qnli" : "dev_acc", 
    "rte" : "dev_acc", 
    "wnli" : "dev_acc"
}
glue_columns = []
for i in glue_tunings["learning_rate"]:
    for j in glue_tunings["warmup_steps"]:
        glue_columns.append("_".join([str(i),str(j)]))
glue_columns.sort()
glue = [glue_keywords, glue_tunings, glue_columns]


project_name="unit_init_glue"
task = glue

api = wandb.Api()
runs = api.runs(path="isnlp_lab/{}".format(project_name))

# print(str(glue_tunings["learning_rate"][0]))
# print(str(1e-5))
# print(str(0.0005))
# quit()
def check_keywork(config, keywords):
    for i,j in keywords.items():
        if j == "":
            continue
        try:
            if config[i] != j:
                return False
        except KeyError:
            print("---------------ERROR-----------------")
            print(config)
            return False
        
    return True

def grouping_df(model, df):
    al_list = {
        "deberta":[20,24],
        "t5":[10,12]
    }
    
    group_best_score_df = dict()
    group_best_score = dict()

    for i in df.index:
            
        if glue_measurement[task[0]["glue_task"]] not in i:
            continue   

        if model in i.split("_"):

            if "baseline" in i:
                group_best_score_df.setdefault("_".join([model, "baseline"]), []).append(i)
                continue

            
            tmp = False
            for al in al_list[model]:
                if "bottom" in i:
                    tmp=True
                    if "bottom"+str(al) in i:
                        tmp=False
                        break
            
            if "top" in i:
                continue

            if tmp:
                print("skip", i)            
                continue
 
            
            for ap in ["befdot", "afterffnn", "aftffnn1", "aftffnn2", "both"]:
                if ap in i.split("_"):
                    for it in ["unit", "he"]:
                        if it in i.split("_"):
                            for at in ["act","noact", "midgelu", "gelu", "relu", "midrelu"]:
                                if at in i.split("_"):
                                    if "adapter" in i:
                                        # group_best_score["_".join([i, ap, it, "adapter"])]
                                        group_best_score_df.setdefault("_".join([model, ap, it, at, "adapter"]), []).append(i)

                                    else:
                                        # group_best_score["_".join([i, ap, it])] = i
                                        group_best_score_df.setdefault("_".join([model, ap, it, at]), []).append(i)
    
    for i, j in group_best_score_df.items():
        tmp_df = df.loc[j,:]
        
        print("\n\n",i)
        print(tmp_df)
        print(i,"best score : ",str(tmp_df.to_numpy().max()))

        group_best_score[i] = tmp_df.to_numpy().max()

    return(group_best_score)
                                
                
                                
                            
                    
            
    


result_dfs = dict()

a = set()
for r in runs:
    if r.state != "finished":
        continue
    
    if check_keywork(r.config, task[0]):
        if r.config[list(task[0].keys())[0]] not in result_dfs.keys():
            result_dfs[r.config[list(task[0].keys())[0]]] = pd.DataFrame(columns=task[2])
        
        print(r.name)
        print(r.summary)
        row_name = r.name.split("_")
        print(row_name)
        
        col_name = []
        for i,j in task[1].items():
            for k in j:
                # print(str(k))
                try:
                    row_name.remove(str(k))
                    col_name.append(k)
                except:
                    pass
                
        row_name = "_".join(row_name)
        print(row_name)
        col_name = "_".join(col_name)
        print(col_name)
        
        for i, j in r.summary.items():
            if "dev" in i or "test" in i:
                try:
                    while not pd.isna(result_dfs[r.config[list(task[0].keys())[0]]].loc[row_name+"__"+i, col_name]):
                        row_name = row_name + "_sub"
                except KeyError:
                    pass
                
                result_dfs[r.config[list(task[0].keys())[0]]].loc[i+"__"+row_name, col_name] = j["max"]
        
        # print(result_dfs[r.config[list(task[0].keys())[0]]])
        # print(result_dfs[r.config[list(task[0].keys())[0]]].sort_index())
        
        print("\n\n")

os.makedirs("tables", exist_ok=True)

for i, j in result_dfs.items():
    j = j.sort_index()
    j['max_value'] = j.max(axis=1)
    print("\n\n\n")
    print(i)
    print(j)
    print("save file to ",str(os.path.join("tables",i+".xlsx")))
    
    deberta_bs = grouping_df("deberta", j)
    t5_bs = grouping_df("t5", j)

    print(json.dumps(deberta_bs, indent=2))
    print(json.dumps(t5_bs, indent=2))
    
    j.to_excel(os.path.join("tables",i+".xlsx"))
        
        
            
        
