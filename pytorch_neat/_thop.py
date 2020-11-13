# proxy import of pytorch opcounter
import os

SLASH = "/"


this_directory = SLASH.join(__file__.split(SLASH)[:-1]) + SLASH
print(f"debug pytorch_neat._thop: trying to import from {this_directory}")
cwd = os.getcwd()
os.chdir(this_directory)
print(os.getcwd())
print("attempting to import pytorch-Opcounter from ",this_directory)

try:
    globals().update(vars(__import__("pytorch-OpCounter")))
except ModuleNotFoundError as MNFE:
    raise Exception(f"cwd:{cwd}; {MNFE}")

os.chdir(cwd)
