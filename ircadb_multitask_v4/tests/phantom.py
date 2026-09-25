"""构造微型、无患者信息的DICOM测试数据；默认参数可在make_phantom修改。"""
from pathlib import Path
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid


def write_slice(path, pixels, z, series, instance, mask=False):
    filemeta = FileMetaDataset()
    filemeta.MediaStorageSOPClassUID = CTImageStorage
    filemeta.MediaStorageSOPInstanceUID = generate_uid()
    filemeta.TransferSyntaxUID = ExplicitVRLittleEndian
    d = FileDataset(str(path), {}, file_meta=filemeta, preamble=b"\0"*128)
    d.SOPClassUID = CTImageStorage
    d.SOPInstanceUID = filemeta.MediaStorageSOPInstanceUID
    d.SeriesInstanceUID = series
    d.Modality = "CT"
    d.PatientName = "Synthetic^Phantom"
    d.PatientID = "SYNTHETIC"
    d.PatientSex = "F"
    d.Rows, d.Columns = pixels.shape
    d.InstanceNumber = instance
    d.ImagePositionPatient = [0., 0., float(z)]
    d.ImageOrientationPatient = [1., 0., 0., 0., 1., 0.]
    d.PixelSpacing = [1., 1.]
    d.SliceThickness = 2.5
    d.SamplesPerPixel = 1
    d.PhotometricInterpretation = "MONOCHROME2"
    d.BitsAllocated = d.BitsStored = 16
    d.HighBit = 15
    d.PixelRepresentation = 0
    d.RescaleSlope, d.RescaleIntercept = 1., -1024.
    d.PixelData = pixels.astype(np.uint16).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    d.save_as(path, enforce_file_format=True)


def make_phantom(root, patients=6, size=32):
    root = Path(root)
    yy, xx = np.mgrid[:size, :size]
    for pid in range(1, patients+1):
        folder = root / f"3Dircadb1.{pid}"
        series = generate_uid()
        n = 3 + pid % 3
        for z in range(n):
            body = ((xx-size/2)**2 + (yy-size/2)**2) < (size*.45)**2
            liver = ((xx-size*.60)**2 + (yy-size*.50)**2) < (size*.24)**2
            tumor = (((xx-size*.65)**2 + (yy-size*.50)**2) < (size*.09)**2) & (pid % 2 == 1) & (z == n//2)
            hu = np.full((size, size), -1000., dtype=np.float32)
            hu[body], hu[liver], hu[tumor] = 20., 100.+pid, 45.
            # 文件名刻意逆序；实际对齐依赖物理位置。
            name = f"image_{n-z:04d}"
            write_slice(folder / "PATIENT_DICOM" / name, hu+1024, z*2.5, series, z+1)
            write_slice(folder / "MASKS_DICOM" / "liver" / name, liver*255, z*2.5, series, z+1, True)
            if pid % 2:
                write_slice(folder / "MASKS_DICOM" / "livertumor01" / name, tumor*255, z*2.5, series, z+1, True)
    return root
