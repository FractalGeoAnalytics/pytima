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
    measurements: "dict[str,dict[str,str]]"
    sample:"dict[str, np.ndarray]"
    _int_map:"dict[str,Type[int]]" = {'ViewField':int, 'ImageWidth':int,'ImageHeight':int, 'SampleDiamter':int} 

    def __init__(
        self,
        uuid: str,
        name: str,
        basepath: Path,
        field_info: Any,
        phases: pd.DataFrame,
        fields: "dict[str,dict[str,np.ndarray]]",
        measurements:"dict[str,dict[str,str]]",
        sample:"dict[str, np.ndarray]"
    ):
        self.uuid = uuid
        self.name = name
        self.basepath = basepath
        self.field_info = field_info
        self.phases = phases
        self.fields = fields
        self.measurements = measurements
        self.sample = sample

    def __repr__(self):
        outstr = f"Sample uuid:{self.uuid}\nSample Name:{self.name}"
        return outstr


def read(foldername: Union[Path, str]) -> Scan:
    """the tima data structure is relatively complex and requires a lot of decoding
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
    measurements = _read_measurement(basepath)
    fields = _read_fields(basepath)
    sample = _fields_to_sample(field_info, phases,fields)
    sf = Scan(uuid_str, foldername.stem, basepath, field_info, phases, fields,measurements,sample)
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
    tmp:dict[str,str] = {}
    xmlns:str = "http://www.tescan.cz/tima/1_4"
    key:str
    val:str
    for i in tree.iter():
        key = i.tag
        key = key.replace('{'+xmlns+'}','')
        val = i.text
        tmp.update({key: val})
    measurements: dict[str, dict[str, str]] = {}
    measurements.update({xmlns:tmp})
    return xmlns, measurements


def _read_fields(basepath: Path) -> "dict[str:dict[str:np.ndarray]]":
    """reads the field information consisting of the bse, mask and phases images"""
    field_path: str = os.path.join(basepath, "fields")
    folders: str = os.listdir(field_path)
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


def _fields_to_sample(tima_locations, phase_info, fields):
    """converts the fields to a set of images that cover the entire sample
    each of the data types is inserted into it's own larger array
    """

    minl = tima_locations[["x", "y"]].min()
    maxl = tima_locations[["x", "y"]].max()
    tima_locations[["x", "y"]].max() - tima_locations[["x", "y"]].min()
    n_steps = ((maxl - minl)) / 760 + 1
    dim_png = 150
    orig = (tima_locations[["x", "y"]] / 760) * 150
    mino = orig.min()
    maxo = orig.max()
    maxdim = (maxo - mino) + dim_png * 2
    phases = np.zeros(np.int32(maxdim.values))
    mask = np.zeros(np.int32(maxdim.values))
    bse = np.zeros(np.int32(maxdim.values))

    big_dim = phases.shape
    for f in fields:
        idx = tima_locations.name == f
        locs = np.int32(orig[idx])
        xrange = np.arange(locs[0][0], locs[0][0] + dim_png) + big_dim[0] // 2
        yrange = np.arange(locs[0][1], locs[0][1] + dim_png) + big_dim[1] // 2
        idx_img = np.ix_(yrange, xrange)
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
    array_shape = np.int32(maxdim.values)
    phases_rgb = np.zeros(np.append(array_shape, 3))
    # map to the right colors
    rgb = np.stack(phase_info.rgb.values)
    ids = phase_info.id.values
    for i in ids:
        tidx = phases == i
        cidx = ids == i
        phases_rgb[tidx] = rgb[cidx, :]

    return {"phases": phases, "phases_rgb": phases_rgb, "bse": bse, "mask": mask}
