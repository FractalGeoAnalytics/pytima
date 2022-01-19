from pytima import mindif
import urllib.request
from pathlib import Path
import shutil
# path to open file data scanned by
# John de Laeter Research Centre 
# https://jdlc.curtin.edu.au/
urlpath = 'http://ddfe.curtin.edu.au/gswa-library/22/IECUR00A7/IECUR00A7.mindif.zip'

with urllib.request.urlopen(urlpath) as response:
   byte_string = response.read()
outfile = Path(urlpath).name
with open(outfile,'wb') as out:
    out.write(byte_string)

shutil.unpack_archive(outfile)

mindif.read(outfile)