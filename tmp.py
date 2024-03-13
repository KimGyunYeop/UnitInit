import wandb

api = wandb.Api()

runs = api.runs(path="isnlp_lab/unit_init_glue")

a = set()
for r in runs:
    if r.state == "crashed" or r.state == "failed":
        continue
    
    print(r.name)
    print(r.state)
    a.add(r.state)
    # break
print(a)