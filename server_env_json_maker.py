import json

server_env = {
    "data_path" : "/home/nlplab/hdd1/datasets"
}

json.dump(server_env, open("server_envs.json","w"), indent= 2)