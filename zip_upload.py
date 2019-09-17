
import os
import zipfile
from zipfile import ZipFile

ROOT_DIR = "XTXStarterKit-master/python"

# for version_0910_01
files = ["core.py", "submission.py", "requirements.txt",
         "data_utils.py", "model.py",  "w0910.hdf5"]
zipfile_name = "python_0911_02.zip"

files_needed = [os.path.join(ROOT_DIR, f) for f in files]
zipfile_name = os.path.join(ROOT_DIR, "zipfiles", zipfile_name)
if os.path.exists(zipfile_name):
    os.remove(zipfile_name)

with ZipFile(zipfile_name, "w", zipfile.ZIP_DEFLATED) as zip_file:
    for path, arcname in zip(files_needed, files):
        zip_file.write(path, arcname)
print("Zip completed!")
print(f"zip file name: {zipfile_name}")
