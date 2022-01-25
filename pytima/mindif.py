import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Tuple, Union, Type

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io


class Scan:
    uuid: str
    name: str
    basepath: Path
    field_info: Any
    phases: pd.DataFrame
    fields: "dict[str,dict[str,np.ndarray]]"
    measurements:"dict[str,Union[str, int]]"
    sample:"dict[str, np.ndarray]"
    xml_namespace:str
    _int_map:"dict[str,Type[int]]" = {'ViewField':int, 'ImageWidth':int,'ImageHeight':int, 'SampleDiameter':int} 

    def __init__(
        self,
        uuid: str,
        name: str,
        basepath: Path,
        field_info: Any,
        phases: pd.DataFrame,
        fields: "dict[str,dict[str,np.ndarray]]",
        measurements:"dict[str,Union[str, int]]",
        sample:"dict[str, np.ndarray]",
        xml_ns:str
    ):
        self.uuid = uuid
        self.name = name
        self.basepath = basepath
        self.field_info = field_info
        self.phases = phases
        self.fields = fields
        self.sample = sample
        self.xml_namespace = xml_ns
        val:int
        for key in self._int_map:
            val = int(measurements[key])
            measurements[key] = val
        self.measurements = measurements


    def __repr__(self):
        outstr = f"Sample uuid:{self.uuid}\nSample Name:{self.name}"
        return outstr
    def plotScan(self,type:str='bse'):
        plt.imshow(self.sample[type],aspect='equal')        
        plt.show()

def read(foldername: Union[Path, str]) -> Scan:
    """the tima data structure is a heirarcy of nested folders conforming to the following
    structure
    NAME
    here is the top level function that calls everything required to reconstruct the TIMA data
    to something that can be used for processing
    """

    if isinstance(foldername, str):
        foldername = Path(foldername)

    next_folder: Path = foldername.joinpath("mindif")
    # extract the uuid for the scan I assume that in a single mindif structure there is only a single scan
    # uuid never more than one
    uuid_str: str = ''
    uuid_regex: str = (
        "[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}"
    )

    for i in next_folder.iterdir():
        uuid_search = re.findall(uuid_regex, str(i))
        if uuid_search != []:
            uuid_str = uuid_search[0]

    basepath: Path = next_folder.joinpath(uuid_str)

    field_info = _read_field_info(basepath)
    phases = _read_phases(basepath)
    xml_ns,measurements = _read_measurement(basepath)
    fields = _read_fields(basepath)
    sample = _fields_to_sample(field_info, phases,fields,measurements)
    sf = Scan(uuid_str, foldername.stem, basepath, field_info, phases, fields,measurements,sample,xml_ns)
    return sf


def _read_field_info(basepath: Path) -> pd.DataFrame:
    """the tima data structure is a nested set of folders
    each folder represents a field
    each field can have multiple datasets
    the fields.xml file descibes where in the sample the fields are located
    using the fields.xml we can then reconstruct the entire sample
    """
    filename: Path = basepath.joinpath("fields.xml")
    tree = ET.parse(filename)
    root = tree.getroot()
    items: "list[dict[str,str]" = []
    for i in list(root.getchildren())[1].getchildren():
        items.append(i.attrib)

    tima_locations: pd.DataFrame = pd.DataFrame(items)
    # ensure that the data is ints as the file read 
    # from ET and concatenation doesn't convert from string to numbers
    tima_locations.x = tima_locations.x.map(int)
    tima_locations.y = tima_locations.y.map(int)
    return tima_locations
    
def _read_phases(basepath: Path) -> pd.DataFrame:
    """reads the phases file and converts to and pd.DataFrame"""
    filename: Path = basepath.joinpath("phases.xml")
    tree = ET.parse(filename)
    root = tree.getroot()
    items = []
    for i in list(root.getchildren())[0].getchildren():
        items.append(i.attrib)
    phases = pd.DataFrame(items)
    phases["rgb"] = phases.color.map(matplotlib.colors.to_rgb)
    # ensure that we have ints as we need this later on for processing 
    # the .tif files
    phases.id = phases.id.map(int)
    return phases


def _read_measurement(basepath: Path) -> Tuple[str, "dict[str, dict[str, str]]"]:
    """reads the measurement file and converts to and pd.DataFrame"""
    filename: Path = basepath.joinpath("measurement.xml")
    tree:ET.ElementTree= ET.parse(filename)
    xmlns:str = "http://www.tescan.cz/tima/1_4"
    key:str
    val:str
    measurements:dict[str, str]={}
    for i in tree.iter():
        key = i.tag
        key = key.replace('{'+xmlns+'}','')
        val = i.text
        measurements.update({key: val})
    return xmlns, measurements


def _read_fields(basepath: Path) -> "dict[str,dict[str,np.ndarray]]":
    """reads the field information consisting of the bse, mask and phases images"""
    field_path: str = os.path.join(basepath, "fields")
    folders: list[str] = os.listdir(field_path)
    file_types: dict[str, str] = {
        "phases": "phases.tif",
        "bse": "bse.png",
        "mask": "mask.png",
    }
    fields: "dict[str,dict[str,np.ndarray]]" = {}
    tmp_dict: "dict[str,np.ndarray]"
    tmp_im: np.ndarray
    for f in folders:
        tmp_dict = {}
        for t in file_types:
            fullfile = os.path.join(field_path, f, file_types[t])
            if os.path.exists(fullfile):
                tmp_im: np.ndarray = io.imread(fullfile)
                tmp_dict.update({t: tmp_im})
            else:
                print(f"missing: {fullfile}")
        fields.update({f: tmp_dict})
    return fields


def _fields_to_sample(field_info:pd.DataFrame, phase_info:pd.DataFrame, fields:"dict[str,dict[str,np.ndarray]]",measurement:"dict[str,Union[str, int]]") -> "dict[str,np.ndarray]":
    """converts the fields to a set of images that cover the entire sample
    each of the data types is inserted into it's own larger array
    """
    field_height:int = int(measurement['ImageHeight'])
    field_width:int = int(measurement['ImageWidth'])
    # calculate the number of fields taken across the sample
    # you can't use the field dimensions so you need to calculate that from the locations table

    # get the step interval from the field locations
    x_step_size:int = np.max(np.unique(np.diff(np.sort(field_info['x'].values))))
    y_step_size:int = np.max(np.unique(np.diff(np.sort(field_info['y'].values))))

    steps = (field_info[["x", "y"]] / [x_step_size, y_step_size])
    # calculate the absolute steps from 0 as we end up with wrapping
    # so the simple process is to add a bias so that all steps start from 0
    # not negative
    
    # extract the outer dimensions of the sample
    # at that point we can then calculate where each image needs to sit inside the larger array
    max_x_step:int = (int(steps['x'].max())*2)+1
    max_y_step:int = (int(steps['y'].max())*2)+1

    x_dim:int = (max_x_step*field_width)
    y_dim:int = (max_y_step*field_height) 

    steps['x'] = steps['x']+max_x_step//2
    steps['y'] = steps['y']+max_y_step//2
    phases:np.ndarray = np.zeros((x_dim, y_dim))
    mask:np.ndarray   = np.zeros((x_dim, y_dim))
    bse:np.ndarray    = np.zeros((x_dim, y_dim))
    fieldim:np.ndarray    = np.zeros((x_dim, y_dim))


    idx:np.ndarray
    xrange:np.ndarray
    yrange:np.ndarray
    idx_img:np.ndarray
    iter = 1
    for f in field_info.name.values:
        idx = field_info.name.values == f
        locs = np.int32(steps[idx])
        xrange = np.arange(0,field_width)+(field_width*locs[0][0])
        yrange = np.arange(0,field_height)+(field_height*locs[0][1])
        #xrange = np.arange(locs[0][0], locs[0][0] + dim_png) + big_dim[0] // 2
        #yrange = np.arange(locs[0][1], locs[0][1] + dim_png) + big_dim[1] // 2
        idx_img = np.ix_(yrange, xrange)
        fieldim[idx_img] = iter
        try:
            phases[idx_img] = np.fliplr(fields[f]["phases"])
        except:
            pass
        try:
            mask[idx_img] = np.fliplr(fields[f]["mask"])
        except:
            pass
        try:
            bse[idx_img] = np.fliplr(fields[f]["bse"])
        except:
            pass
        
        iter=iter+1
    

    phases_rgb:np.ndarray = np.zeros((x_dim, y_dim,3))
    # map to the right colors
    rgb:np.ndarray = np.stack(phase_info.rgb.values)
    ids:np.ndarray = phase_info.id.values
    for i in ids:
        tidx = phases == i
        cidx = ids == i
        phases_rgb[tidx] = rgb[cidx, :]

    return {"phases": phases, "phases_rgb": phases_rgb, "bse": bse, "mask": mask}

if __name__ == '__main__':
    from pathlib import Path
    import mindif
    foldername = Path('data/IECUR00A7.mindif')
    mindif.read(foldername)