"""Export Blender liquid Flow objects to Babylon Lite's flat fluid JSON format.

Run from Blender:
  blender scene.blend --background --python scripts/export-blender-fluid.py -- fluid.json

Optional object custom properties:
  blite_shape: box | sphere | cylinder | cone | capsule | polygonPrism
  blite_volume_rate: inflow/recycle volume per second (omit for unlimited)
  blite_targets: comma-separated emitter object names
  blite_inner_radius, blite_top_radius, blite_points, blite_spread
"""

import json
import re
import sys

import bpy
from mathutils import Quaternion, Vector


def blite_vec(v):
    return [float(v.x), float(v.z), float(-v.y)]


def object_id(obj):
    return re.sub(r"[^a-z0-9_-]+", "-", obj.name.lower()).strip("-") or "flow"


def flow_settings(obj):
    for modifier in obj.modifiers:
        if modifier.type == "FLUID" and modifier.fluid_type == "FLOW":
            return modifier.flow_settings
    return None


def local_size(obj):
    corners = [Vector(corner) for corner in obj.bound_box]
    minimum = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maximum = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return maximum - minimum


def shape_for(obj):
    size = local_size(obj)
    kind = str(obj.get("blite_shape", "box"))
    if kind == "sphere":
        return {"type": kind, "radius": float(max(size) * 0.5)}
    if kind == "cylinder":
        shape = {"type": kind, "radius": float(max(size.x, size.y) * 0.5), "height": float(size.z)}
        inner = float(obj.get("blite_inner_radius", 0))
        if inner > 0:
            shape["innerRadius"] = inner
        return shape
    if kind == "cone":
        return {
            "type": kind,
            "bottomRadius": float(max(size.x, size.y) * 0.5),
            "topRadius": float(obj.get("blite_top_radius", 0)),
            "height": float(size.z),
        }
    if kind == "capsule":
        radius = float(max(size.x, size.y) * 0.5)
        return {"type": kind, "radius": radius, "height": float(size.z)}
    if kind == "polygonPrism":
        points = json.loads(str(obj.get("blite_points", "[]")))
        if len(points) < 3:
            raise ValueError(f"{obj.name}: polygonPrism requires blite_points=[[x,z], ...]")
        return {"type": kind, "points": points, "thickness": float(size.z)}
    return {"type": "box", "size": [float(size.x), float(size.z), float(size.y)]}


def transform_for(obj):
    position, rotation, scale = obj.matrix_world.decompose()
    basis = Quaternion((1, 0, 0), -1.5707963267948966)
    q = basis @ rotation @ basis.conjugated()
    return {
        "position": blite_vec(position),
        "rotation": [float(q.x), float(q.y), float(q.z), float(q.w)],
        "scale": [float(scale.x), float(scale.z), float(scale.y)],
    }


def main():
    output = next((arg for arg in sys.argv[sys.argv.index("--") + 1 :] if not arg.startswith("-")), "//fluid.json") if "--" in sys.argv else "//fluid.json"
    emitters = []
    pending_sinks = []
    object_ids = {}

    for obj in bpy.context.scene.objects:
        settings = flow_settings(obj)
        if settings is None:
            continue
        flow_id = object_id(obj)
        suffix = 2
        base_id = flow_id
        while flow_id in object_ids.values():
            flow_id = f"{base_id}-{suffix}"
            suffix += 1
        object_ids[obj.name] = flow_id
        behavior = settings.flow_behavior
        if behavior == "OUTFLOW":
            pending_sinks.append((obj, settings))
            continue
        velocity = Vector(settings.velocity_coord) if getattr(settings, "use_initial_velocity", False) else Vector((0, 0, 0))
        emitter = {
            "id": flow_id,
            "name": obj.name,
            "enabled": bool(getattr(settings, "use_inflow", True)),
            "behavior": "initial" if behavior == "GEOMETRY" else "inflow",
            "transform": transform_for(obj),
            "shape": shape_for(obj),
            "sampling": "surface" if getattr(settings, "use_plane_init", False) else "volume",
            "velocity": blite_vec(velocity),
            "velocitySpace": "local",
            "spread": float(obj.get("blite_spread", 0)),
        }
        if behavior != "GEOMETRY" and "blite_volume_rate" in obj:
            emitter["volumeRate"] = float(obj["blite_volume_rate"])
        emitters.append(emitter)

    inflow_ids = [emitter["id"] for emitter in emitters if emitter["behavior"] == "inflow"]
    sinks = []
    for obj, settings in pending_sinks:
        names = [name.strip() for name in str(obj.get("blite_targets", "")).split(",") if name.strip()]
        targets = [object_ids[name] for name in names if name in object_ids] or inflow_ids
        sink = {
            "id": object_ids[obj.name],
            "name": obj.name,
            "enabled": bool(getattr(settings, "use_inflow", True)),
            "transform": transform_for(obj),
            "shape": shape_for(obj),
            "targets": targets,
        }
        if "blite_volume_rate" in obj:
            sink["volumeRate"] = float(obj["blite_volume_rate"])
        sinks.append(sink)

    path = bpy.path.abspath(output)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump({"formatVersion": 2, "emitters": emitters, "sinks": sinks}, stream, indent=2)
        stream.write("\n")
    print(f"Exported {len(emitters)} emitters and {len(sinks)} sinks to {path}")


main()
