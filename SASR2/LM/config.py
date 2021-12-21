import os

file_path = os.path.dirname(os.path.realpath(__file__))

train_path = file_path + "/data/lm_train.txt"
eval_path = file_path + "/data/lm_eval.txt"
bert_path = os.path.dirname(file_path) + "/ChineseBERT-base"

preprocess_config = {
    "max_len": 32
}

training_config = {
    "epoch": 20,
    "batch_size": 32,
    "shuffle": True,
    "save_path": file_path + "/lm_model"
}
