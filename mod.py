#import yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

with open("environment.yml") as file_handle:
    environment_data = load(file_handle, Loader=Loader)
    
with open("requirements.txt", "w") as file_handle:
    for dependency in environment_data["dependencies"]:
        print(dependency)
        package_name, package_version = dependency.split("=")
        file_handle.write("{} == {}".format(package_name, package_version))
