#!/usr/bin/env python3
#
# This script will take care of pulling the specified
# AI based script and will accordingly run it.
#
# It is also going to make sure that the required dependencies
# for the script are installed and will install the package
# dependencies if they are not present already.

import requests
import sys
import subprocess
import shutil
from os import path, mkdir, getcwd
from importlib import import_module
from importlib import util

this_python = sys.version_info[:2]
min_version = (3, 7)
if this_python < min_version:
    message_parts = [
        "This script does not work on Python {}.{}.".format(*this_python),
        "The minimum supported Python version is {}.{}.".format(*min_version)
    ]
    print("ERROR: " + " ".join(message_parts))
    sys.exit(1)


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


required_packages = {
    "rich": "13.3.3",
    "requests": "2.28.2"
}

script_to_url = {
    "knn": "https://raw.githubusercontent.com/appbaseio/ai-scripts/master/knn_reindex/main.py",
    "metadata": "https://raw.githubusercontent.com/appbaseio/ai-scripts/master/metadata/main.py"
}

for package, version in required_packages.items():
    if util.find_spec(package) is None:
        print(f"{package}: is not installed but is a dependency, installing it!")
        install_package(f"{package}=={version}")

# Import requests since we will need to use it


def run_script(script):
    subprocess.check_call([sys.executable, script])


def pull_script(url, name, dir) -> str:
    """
    Pull the script as specified by the user.
    """
    if url is None:
        return

    # Pull the script, import it
    # and run `main()`
    file_response = requests.get(url, allow_redirects=True)

    if not file_response.ok:
        print("Couldn't pull script, please try again!")
        return None

    file_path = path.join(dir, name + ".py")
    open(file_path, "wb").write(file_response.content)
    return file_path


def main():
    tmpdir = None
    try:
        tmpdir = path.join(getcwd(), "tmp")
        mkdir(tmpdir)
        # Extract the argv value passed by the user
        if len(sys.argv) < 2:
            print("Please pass the type of script you want to run. Refer to https://github.com/appbaseio/ai-scripts for more details.")
            sys.exit(1)

        script_name = sys.argv[1]

        # Make sure script can be installed
        if script_name not in script_to_url:
            print(f"{script_name}: not a recognized script!")

        script_path = pull_script(script_to_url.get(
            script_name, None), script_name, tmpdir)
        if script_path is not None:
            # Import it and run main()
            module = import_module(f"tmp.{script_name}")
            main_method = getattr(module, "main")
            main_method()
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
