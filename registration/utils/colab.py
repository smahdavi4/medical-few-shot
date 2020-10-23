import os
import time
import subprocess

from google.colab import files

RESULT_FOLDER = 'result_{}'.format(time.time())

if not os.path.isdir(RESULT_FOLDER):
    os.mkdir(RESULT_FOLDER)


def download_results():
    zip_folder_name = '{}.zip'.format(RESULT_FOLDER)
    subprocess.call("zip -r {} {}".format(zip_folder_name, RESULT_FOLDER), shell=True)
    files.download(zip_folder_name)
