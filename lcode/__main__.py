import os
import argparse
from lcode import __version__

__EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'examples')
def get_available_configs():
    configs = os.listdir(__EXAMPLE_DIR)
    configs = [config.replace('.py', '') for config in configs]
    return configs

def main():
    parser = argparse.ArgumentParser(prog="LCODE", usage="%(prog)s [options]")

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")
    
    generate_parser = subparsers.add_parser("generate", help="create default config")
    __available_configs = get_available_configs()
    generate_parser.add_argument('config_name', type=str, help="Name of the config: {}".format(", ".join(__available_configs)))
    args = parser.parse_args()
    

    if args.subparser_name == "generate":
        if args.config_name not in __available_configs:
            print(f'Choose from {" ,".join(__available_configs)}')
            return
        with open(os.path.join(__EXAMPLE_DIR,'{}.py'.format(args.config_name)), 'r') as input, \
             open('run.py', 'w') as output:
            output.write(input.read())
            

        

        

if __name__ == "__main__":
    main()