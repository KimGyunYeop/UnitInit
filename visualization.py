import wandb
import pandas as pd
import numpy as np
import os
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_colwidth', None)

glue_keywords={
    "glue_task":"cola",
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
    "mrpc" : "dev_acc", 
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

#=======================================================================================================================================

def check_keywork(config, keywords):
    for i,j in keywords.items():
        if j == "":
            continue
        try:
            if config[i] != j:
                return False
        except KeyError:
            # print("---------------ERROR-----------------")
            # print(config)
            return False
        
    return True        
     
#=======================================================================================================================================

result_dfs = dict()

a = set()
for r in runs:
    if r.state != "finished":
        continue
    
    if check_keywork(r.config, task[0]):
        if r.config[list(task[0].keys())[0]] not in result_dfs.keys():
            result_dfs[r.config[list(task[0].keys())[0]]] = pd.DataFrame(columns=task[2])

        row_name = r.name.split("_")

        col_name = []
        for i,j in task[1].items():
            for k in j:
                try:
                    row_name.remove(str(k))
                    col_name.append(k)
                except:
                    pass
                
        row_name = "_".join(row_name)
        col_name = "_".join(col_name)
        
        for i, j in r.summary.items():
            if "dev" in i or "test" in i:
                try:
                    while not pd.isna(result_dfs[r.config[list(task[0].keys())[0]]].loc[row_name+"__"+i, col_name]):
                        row_name = row_name + "_sub"
                except KeyError:
                    pass
                
                result_dfs[r.config[list(task[0].keys())[0]]].loc[i+"__"+row_name, col_name] = j["max"]

#=======================================================================================================================================

def make_he_plot(check_case, baseline_check_case):
    df = pd.DataFrame({'bottom': [0 for i in range(24)],
            'top': [0 for i in range(24)]})
        
    for i, j in result_dfs.items():
        j = j.sort_index()
        j['max_value'] = j.max(axis=1)

        for k in j.index:
            case_name=k.split("_")

            if (all(x in case_name for x in check_case)):
                for name in case_name:
                    if "bottom" in name:
                        if "-" in name: 
                            layer_num=int(name[7:])
                        else:
                            layer_num=int(name[6:])
                        layer_type=name[:6]
                    elif "top" in name:
                        if "-" in name: 
                            layer_num=int(name[4:])
                        else:
                            layer_num=int(name[3:])
                        layer_type=name[:3]
                    else: 
                        continue
                    df.loc[layer_num, layer_type]= j.loc[k,'max_value']
            
            if (all(x in case_name for x in baseline_check_case)):
                df.loc[0, "bottom"]= j.loc[k,'max_value']
                df.loc[0, "top"]= j.loc[k,'max_value']

    he_plot=sns.lineplot(data=df[['bottom', 'top']])
    plt.savefig(glue_keywords["glue_task"]+"_he_plot.png") 
    plt.clf()

#=======================================================================================================================================

def make_proposed_plot(proposed_check_case, he_check_case, baseline_check_case):
    df = pd.DataFrame({'proposed': [0 for i in range(24)],
            'w/o proposed': [0 for i in range(24)]})
        
    for i, j in result_dfs.items():
        j = j.sort_index()
        j['max_value'] = j.max(axis=1)

        for k in j.index:
            case_name=k.split("_")

            if (all(x in case_name for x in proposed_check_case)):
                for name in case_name:
                    if "bottom" in name:
                        if "-" in name: 
                            layer_num=int(name[7:])
                        else:
                            layer_num=int(name[6:])
                        layer_type='proposed'
                    else: 
                        continue
                    df.loc[layer_num, layer_type]= j.loc[k,'max_value']

            elif (all(x in case_name for x in he_check_case)):
                for name in case_name:
                    if "bottom" in name:
                        if "-" in name: 
                            layer_num=int(name[7:]) 
                        else:
                            layer_num=int(name[6:])
                        layer_type='w/o proposed'
                    else: 
                        continue
                    df.loc[layer_num, layer_type]= j.loc[k,'max_value']

            if (all(x in case_name for x in baseline_check_case)):
                df.loc[0, 'proposed']= j.loc[k,'max_value']
                df.loc[0, 'w/o proposed']= j.loc[k,'max_value']

    proposed_plot=sns.lineplot(data=df[['proposed', 'w/o proposed']])
    plt.savefig(glue_keywords["glue_task"]+"_proposed_plot.png") 
    plt.clf()

#=======================================================================================================================================

if glue_keywords["glue_task"]=="mrpc":
    check_case=['dev','deberta', glue_keywords["glue_task"], 'befdot', 'he', 'no', 'act', 'accuracy']
    baseline_check_case=['dev','deberta', glue_keywords["glue_task"], 'baseline', 'accuracy']
    make_he_plot(check_case, baseline_check_case)

    proposed_check_case=['dev','deberta', glue_keywords["glue_task"], 'befdot', 'unit', 'midgelu', 'accuracy']
    he_check_case=['dev','deberta', glue_keywords["glue_task"], 'befdot', 'he', 'no', 'act', 'accuracy']
    baseline_check_case=['dev','deberta', glue_keywords["glue_task"], 'baseline', 'accuracy']
    make_proposed_plot(proposed_check_case, he_check_case, baseline_check_case)

elif glue_keywords["glue_task"]=="cola":
    check_case=['dev','deberta', glue_keywords["glue_task"], 'befdot', 'he', 'no', 'act']
    baseline_check_case=['dev','deberta', glue_keywords["glue_task"], 'baseline']
    make_he_plot(check_case, baseline_check_case)

    proposed_check_case=['dev','deberta', glue_keywords["glue_task"], 'befdot', 'unit', 'midgelu']
    he_check_case=['dev','deberta', glue_keywords["glue_task"], 'befdot', 'he', 'no', 'act']
    baseline_check_case=['dev','deberta', glue_keywords["glue_task"], 'baseline']
    make_proposed_plot(proposed_check_case, he_check_case, baseline_check_case)

