import argparse
import os
import json

from types import SimpleNamespace as Namespace

from src.CPP.Environment import CPPEnvironmentParams, CPPEnvironment


from utils import *


def main_cpp(p):
    env = CPPEnvironment(p)

    env.run()


if __name__ == "__main__":
    config_json_loc = "config/manhattan32_cpp.json"
    with open(config_json_loc, "r") as config_file:
        params = json.loads(config_file.read(), object_hook=lambda d: Namespace(**d))

        # Run Model
        main_cpp(p=params)