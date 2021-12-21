import os
def write_log(dialogue,begin:bool=False):
    log_file='./dialogue_log/log.txt'
    with open(log_file,'a') as f:
        if begin==True:
            f.write('==============================')
        f.write(dialogue)
        