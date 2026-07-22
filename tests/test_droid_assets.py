from pathlib import Path
from xml.etree import ElementTree


MODEL_ROOT = Path("assets/third_party/droid_franka_robotiq85")
MODEL_XML = MODEL_ROOT / "droid_panda_2f85.xml"


def test_droid_model_meshes_and_required_joints_are_versioned() -> None:
    root = ElementTree.parse(MODEL_XML).getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    mesh_root = MODEL_ROOT / compiler.attrib["meshdir"]
    missing = []
    for mesh in root.findall("./asset/mesh"):
        path = (mesh_root / mesh.attrib["file"]).resolve()
        if not path.is_file():
            missing.append(path)
    assert not missing

    joint_names = {joint.attrib.get("name") for joint in root.findall(".//joint")}
    assert {f"joint{index}" for index in range(1, 8)} <= joint_names
    assert {
        "finger_joint",
        "left_inner_knuckle_joint",
        "left_inner_finger_joint",
        "right_inner_knuckle_joint",
        "right_inner_finger_joint",
        "right_outer_knuckle_joint",
    } <= joint_names


def test_droid_matching_geometries_use_visual_group_two() -> None:
    root = ElementTree.parse(MODEL_XML).getroot()
    visual_default = root.find("./default/default/default[@class='visual']/geom")
    assert visual_default is not None
    assert visual_default.attrib["group"] == "2"
    for geom in root.findall(".//body[@name='robotiq_85_base_link']//geom"):
        assert geom.attrib["group"] == "2"
        assert geom.attrib["rgba"] == "0.1 0.1 0.1 1"
