"""官网参考信息，不替代对用户本地 DICOM 的审计。2026-09-18 核对。"""
OFFICIAL_URL = "https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/"
SLICE_COUNTS = [129, 172, 200, 91, 139, 135, 151, 124, 111, 122,
                132, 260, 122, 113, 125, 155, 119, 74, 124, 225]
SEX = ["F", "F", "M", "M", "M", "M", "M", "F", "M", "F",
       "M", "F", "M", "F", "F", "M", "M", "F", "F", "F"]
# 官网表格的“0 tumour in Liver”仅描述肝内病灶，7号另有一个肾上腺病灶。
# 本项目的目标类别还包含7号 MASKS_DICOM/tumor，故参考行不能把7号标成阴性。
LIVER_NEGATIVE = [5, 7, 11, 14, 20]
TARGET_NEGATIVE_REFERENCE = [5, 11, 14, 20]
# 2026-09-23 逐例核对官方 MASKS_DICOM.zip 的目标目录名。
# 5号的 leftsurretumor/rightsurretumor 与7号的 tumor 必须区别处理。
EXPECTED_TARGET_MASKS_BY_PATIENT = {
    1: tuple(f"livertumor{i:02d}" for i in range(1, 8)),
    2: ("livertumor",), 3: ("livertumor",), 4: ("livertumor",),
    5: (), 6: ("livertumor",), 7: ("tumor",),
    8: ("livertumor01", "livertumor02", "livertumor03"),
    9: ("livertumor",), 10: ("livertumor",), 11: (),
    12: ("livertumor",), 13: ("livertumor",), 14: (),
    15: ("livertumor",), 16: ("livertumor",),
    17: ("livertumor1", "livertumor2"), 18: ("livertumor",),
    19: ("livertumors",), 20: (),
}


def reference_rows():
    return [{"id": f"3Dircadb1.{i}", "number": i, "n_slices": n,
             "sex": SEX[i - 1], "has_tumor": i not in TARGET_NEGATIVE_REFERENCE,
             "tumor_voxels": 0, "reference_only": True,
             "tumor_presence_scope": "configured_target_reference_not_local_dicom"}
            for i, n in enumerate(SLICE_COUNTS, 1)]
