import pkg_resources
import subprocess

for dist in pkg_resources.working_set:
    package = dist.project_name
    print(f"Upgrading {package}...")
    subprocess.check_call(["pip", "install", "--upgrade", package])