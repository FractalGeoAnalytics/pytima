**pytima**
==========
an open source reader for tescan tima MINDIF MINeralogy Data Interchange Format files


# Installation
```
pip install git+https://github.com/FractalGeoAnalytics/pytima
```
# Example usage

We will use openfile data scanned by [John de Laeter Research Centre](https://jdlc.curtin.edu.au/)


```python
from pytima import mindif
import urllib.request
from pathlib import Path
import shutil
import os


urlpath = 'http://ddfe.curtin.edu.au/gswa-library/22/IECUR00A7/IECUR00A7.mindif.zip'
local_file = Path(urlpath).name
local_folder = local_file.replace('.zip','')
data_folder = Path('./data')
output_file = data_folder.joinpath(local_file)
# check if the output folder exists if not make it
if not data_folder.exists():
    data_folder.mkdir()

# check if the file exists if not download
if not output_file.exists():
    # download the data
    with urllib.request.urlopen(urlpath) as response:
        byte_string = response.read()

    # write the byte string
    with open(output_file,'wb') as out:
        out.write(byte_string)
    # unpack the zip file 
    shutil.unpack_archive(output_file,data_folder.joinpath(output_file.stem))

# read the data
scan = mindif.read(data_folder.joinpath(output_file.stem))
# make some plots
plot_types = ['phases_rgb', 'bse', 'mask','phases']
for i,j in enumerate(plot_types):
    scan.plotScan(j)

# access the data
scan.sample['bse']

scan.sample['phases_rgb']

scan.sample['mask']

scan.sample['phases']

```
