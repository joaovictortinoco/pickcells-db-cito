import os

def getPath():
    os.makedirs(f'{os.getcwd()}/production', exist_ok=True)
    return f'{os.getcwd()}/production'

def getPathLocal():
    return f'{os.getcwd()}/experiments_round'