import os

def getPath():
    os.makedirs(f'{os.getcwd()}/production', exist_ok=True)
    return f'{os.getcwd()}/production'
