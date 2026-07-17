#!/usr/bin/env python3
"""Convert Baxter COLLADA visual meshes into MuJoCo-friendly material-split OBJs.

MuJoCo does not load COLLADA directly.  The official Baxter meshes store several
flat-colour materials in each DAE, while the STL copies lose those material
boundaries.  This script preserves the original geometry and splits every DAE
mesh into one OBJ per material, then writes a new URDF whose visual elements
reference those OBJ files.  Collision geometry, joints, inertials, and every
non-visual element are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"
NS = {"c": COLLADA_NS}


def floats(text: str | None) -> list[float]:
    return [float(value) for value in (text or "").split()]


def integers(text: str | None) -> list[int]:
    return [int(value) for value in (text or "").split()]


def source_arrays(mesh: ET.Element) -> dict[str, list[tuple[float, ...]]]:
    arrays: dict[str, list[tuple[float, ...]]] = {}
    for source in mesh.findall("c:source", NS):
        source_id = source.attrib["id"]
        values = floats(source.findtext("c:float_array", default="", namespaces=NS))
        accessor = source.find("c:technique_common/c:accessor", NS)
        if accessor is None:
            raise ValueError(f"{source_id}: missing accessor")
        stride = int(accessor.attrib.get("stride", "1"))
        arrays[source_id] = [
            tuple(values[index : index + stride])
            for index in range(0, len(values), stride)
        ]
    return arrays


def material_colours(root: ET.Element) -> dict[str, tuple[float, float, float, float]]:
    effect_colours: dict[str, tuple[float, float, float, float]] = {}
    for effect in root.findall(".//c:library_effects/c:effect", NS):
        colour = effect.find(
            "c:profile_COMMON/c:technique/*/c:diffuse/c:color", NS
        )
        if colour is not None:
            rgba = tuple(floats(colour.text))
            if len(rgba) == 4:
                effect_colours[effect.attrib["id"]] = rgba  # type: ignore[assignment]

    colours: dict[str, tuple[float, float, float, float]] = {}
    for material in root.findall(".//c:library_materials/c:material", NS):
        instance = material.find("c:instance_effect", NS)
        if instance is None:
            continue
        effect_id = instance.attrib["url"].removeprefix("#")
        if effect_id in effect_colours:
            colours[material.attrib["id"]] = effect_colours[effect_id]
    return colours


def vertices_sources(mesh: ET.Element) -> dict[str, dict[str, str]]:
    mappings: dict[str, dict[str, str]] = {}
    for vertices in mesh.findall("c:vertices", NS):
        mappings[vertices.attrib["id"]] = {
            item.attrib["semantic"]: item.attrib["source"].removeprefix("#")
            for item in vertices.findall("c:input", NS)
        }
    return mappings


def primitive_faces(
    primitive: ET.Element,
    arrays: dict[str, list[tuple[float, ...]]],
    vertices: dict[str, dict[str, str]],
) -> list[list[tuple[tuple[float, ...], tuple[float, ...] | None]]]:
    inputs: dict[str, tuple[int, str]] = {}
    max_offset = 0
    for item in primitive.findall("c:input", NS):
        semantic = item.attrib["semantic"]
        offset = int(item.attrib.get("offset", "0"))
        source = item.attrib["source"].removeprefix("#")
        inputs[semantic] = (offset, source)
        max_offset = max(max_offset, offset)
    stride = max_offset + 1

    if "VERTEX" not in inputs:
        raise ValueError("primitive has no VERTEX input")
    vertex_offset, vertex_source = inputs["VERTEX"]
    position_source = vertices[vertex_source]["POSITION"]
    normal_input = inputs.get("NORMAL")

    packed = integers(primitive.findtext("c:p", default="", namespaces=NS))
    tag = primitive.tag.rsplit("}", 1)[-1]
    if tag == "triangles":
        counts = [3] * int(primitive.attrib["count"])
    elif tag == "polylist":
        counts = integers(primitive.findtext("c:vcount", default="", namespaces=NS))
    else:
        raise ValueError(f"unsupported COLLADA primitive: {tag}")

    faces = []
    cursor = 0
    for count in counts:
        polygon = []
        for _ in range(count):
            entry = packed[cursor : cursor + stride]
            cursor += stride
            position = arrays[position_source][entry[vertex_offset]][:3]
            normal = None
            if normal_input is not None:
                normal_offset, normal_source = normal_input
                normal = arrays[normal_source][entry[normal_offset]][:3]
            polygon.append((position, normal))
        for index in range(1, len(polygon) - 1):
            faces.append([polygon[0], polygon[index], polygon[index + 1]])
    return faces


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_obj(
    path: Path,
    faces: list[list[tuple[tuple[float, ...], tuple[float, ...] | None]]],
) -> dict[str, int]:
    vertex_map: dict[tuple[tuple[float, ...], tuple[float, ...] | None], int] = {}
    vertices_out: list[tuple[float, ...]] = []
    normals_out: list[tuple[float, ...] | None] = []
    face_indices: list[list[int]] = []

    for face in faces:
        indices = []
        for item in face:
            if item not in vertex_map:
                vertex_map[item] = len(vertices_out) + 1
                vertices_out.append(item[0])
                normals_out.append(item[1])
            indices.append(vertex_map[item])
        face_indices.append(indices)

    with path.open("w", encoding="utf-8") as output:
        output.write("# Converted from the official Rethink Robotics Baxter DAE\n")
        for vertex in vertices_out:
            output.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        has_normals = all(normal is not None for normal in normals_out)
        if has_normals:
            for normal in normals_out:
                assert normal is not None
                output.write(f"vn {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n")
        for face in face_indices:
            if has_normals:
                output.write("f " + " ".join(f"{i}//{i}" for i in face) + "\n")
            else:
                output.write("f " + " ".join(str(i) for i in face) + "\n")
    return {"vertices": len(vertices_out), "triangles": len(face_indices)}


def convert_dae(dae_path: Path, output_dir: Path) -> list[dict]:
    root = ET.parse(dae_path).getroot()
    colours = material_colours(root)
    records = []
    for geometry in root.findall(".//c:library_geometries/c:geometry", NS):
        mesh = geometry.find("c:mesh", NS)
        if mesh is None:
            continue
        arrays = source_arrays(mesh)
        vertex_mappings = vertices_sources(mesh)
        grouped_faces = defaultdict(list)
        for primitive in list(mesh):
            tag = primitive.tag.rsplit("}", 1)[-1]
            if tag not in {"triangles", "polylist"}:
                continue
            material = primitive.attrib.get("material", "default")
            grouped_faces[material].extend(
                primitive_faces(primitive, arrays, vertex_mappings)
            )

        for ordinal, (material, faces) in enumerate(grouped_faces.items()):
            name = (
                f"{safe_name(dae_path.stem)}__"
                f"{ordinal:02d}_{safe_name(material.removesuffix('-material'))}.obj"
            )
            obj_path = output_dir / name
            counts = write_obj(obj_path, faces)
            records.append(
                {
                    "source": dae_path.as_posix(),
                    "obj": obj_path.as_posix(),
                    "material": material,
                    "rgba": colours.get(material, (0.5, 0.5, 0.5, 1.0)),
                    **counts,
                }
            )
    return records


def visual_xml(origin: ET.Element | None, mesh_path: str, rgba: tuple[float, ...]) -> ET.Element:
    visual = ET.Element("visual")
    if origin is not None:
        visual.append(ET.fromstring(ET.tostring(origin, encoding="unicode")))
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "mesh", {"filename": mesh_path})
    material = ET.SubElement(visual, "material", {"name": f"dae_{Path(mesh_path).stem}"})
    ET.SubElement(material, "color", {"rgba": " ".join(f"{value:.9g}" for value in rgba)})
    return visual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-urdf", type=Path, required=True)
    parser.add_argument("--output-urdf", type=Path, required=True)
    parser.add_argument("--output-mesh-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_mesh_dir.mkdir(parents=True, exist_ok=True)
    urdf_root = ET.parse(args.source_urdf).getroot()
    urdf_dir = args.source_urdf.parent
    conversions: dict[str, list[dict]] = {}

    for link in urdf_root.findall("link"):
        visuals = list(link.findall("visual"))
        replacements: list[tuple[ET.Element, list[ET.Element]]] = []
        for visual in visuals:
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.attrib.get("filename", "")
            if not filename.lower().endswith(".stl"):
                continue
            dae_filename = filename[:-4] + ".DAE"
            dae_path = urdf_dir / dae_filename
            if not dae_path.exists():
                continue
            if dae_filename not in conversions:
                conversions[dae_filename] = convert_dae(dae_path, args.output_mesh_dir)
            origin = visual.find("origin")
            new_visuals = []
            for record in conversions[dae_filename]:
                relative_obj = Path(record["obj"]).relative_to(urdf_dir).as_posix()
                new_visuals.append(visual_xml(origin, relative_obj, tuple(record["rgba"])))
            replacements.append((visual, new_visuals))

        for original, new_visuals in replacements:
            position = list(link).index(original)
            link.remove(original)
            for offset, replacement in enumerate(new_visuals):
                link.insert(position + offset, replacement)

    ET.indent(urdf_root, space="  ")
    args.output_urdf.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(urdf_root).write(
        args.output_urdf, encoding="utf-8", xml_declaration=True
    )
    manifest = {
        "source_urdf": args.source_urdf.as_posix(),
        "output_urdf": args.output_urdf.as_posix(),
        "converted_dae_count": len(conversions),
        "obj_count": sum(len(records) for records in conversions.values()),
        "meshes": conversions,
    }
    manifest_path = args.output_mesh_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "output_urdf", "converted_dae_count", "obj_count"
    )}, indent=2))


if __name__ == "__main__":
    main()
