#!/usr/bin/env python
# coding: utf-8

import os
import requests
import tarfile


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar"

# Local paths
download_path = "dataset.tar"
extract_path = os.path.join(os.getcwd(),"../data/")

# Download the file
print("Downloading dataset...")
response = requests.get(url, stream=True)
with open(download_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)
print("Download complete.")


#  Extract the .tar.gz file
print("Extracting files...")
with tarfile.open(download_path, "r:") as tar:
    tar.extractall(path=extract_path)
print("Extraction complete.")

#  Remove the downloaded archive
os.remove(download_path)
print("Cleanup done.")
