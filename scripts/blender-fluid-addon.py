"""Blender add-on exporting live Babylon Lite fluid setups as .blitefluid bundles."""

bl_info = {
    "name": "Babylon Lite Fluid Bundle",
    "author": "Babylon Lite contributors",
    "version": (1, 2, 0),
    "blender": (4, 0, 0),
    "location": "Properties > Scene > Babylon Lite Fluid; File > Export",
    "description": "Export a liquid domain, flows, visuals, and collision SDF",
    "category": "Import-Export",
}

import json
import math
import os
import re
import struct
import tempfile
import zipfile

import bmesh
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree

PHYSICS_SCHEMAS = {
    "PBF": (
        ("gravity", "Gravity", 0, 200, 0.1, 9.8, "FLOAT"),
        ("viscosity", "Viscosity (XSPH)", 0, 3, 0.005, 0.08, "FLOAT"),
        ("relaxation", "Relaxation", 1, 1000, 1, 50, "INT"),
        ("scorr", "Artificial pressure", 0, 0.5, 0.001, 0.02, "FLOAT"),
        ("iterations", "Solver iterations", 1, 8, 1, 3, "INT"),
        ("restDensity", "Rest density", 100, 2000, 10, 341, "INT"),
        ("boundaryDensity", "Boundary density", 0, 1, 0.05, 0, "FLOAT"),
    ),
    "MLS-MPM": (
        ("gravity", "Gravity", 0, 200, 0.1, 9.8, "FLOAT"),
        ("stiffness", "Stiffness (EOS)", 10, 5000, 10, 350, "INT"),
        ("viscosity", "Viscosity", 0, 1, 0.01, 0.3, "FLOAT"),
        ("restDensity", "Rest density (/cell)", 1, 100, 0.5, 3, "FLOAT"),
        ("damping", "Velocity damping", 0.9, 1, 0.001, 0.995, "FLOAT"),
        ("affineDamping", "Affine damping", 0.1, 1, 0.005, 0.9, "FLOAT"),
        ("groundDamp", "Ground damping", 0.7, 1, 0.01, 0.85, "FLOAT"),
        ("groundDampHeight", "Ground damp height", 0, 10, 0.1, 1.5, "FLOAT"),
        ("restitution", "Restitution", 0, 1, 0.05, 0.3, "FLOAT"),
        ("substeps", "Substeps / frame", 1, 8, 1, 3, "INT"),
        ("maxSubDtMs", "Max sub-step (ms)", 2, 20, 0.1, 8.4, "FLOAT"),
    ),
    "PB-MPM": (
        ("gravity", "Gravity", 0, 200, 0.1, 9.8, "FLOAT"),
        ("iterations", "PB iterations", 1, 12, 1, 5, "INT"),
        ("liquidRelaxation", "Liquid relaxation", 0.1, 3, 0.05, 1.5, "FLOAT"),
        ("liquidViscosity", "Liquid viscosity", 0, 0.2, 0.005, 0.01, "FLOAT"),
        ("elasticityRatio", "Elasticity ratio", 0, 1, 0.01, 0.3, "FLOAT"),
        ("elasticRelaxation", "Elastic relaxation", 0.05, 1, 0.01, 0.3, "FLOAT"),
        ("frictionAngle", "Sand friction angle", 0, 60, 1, 35, "INT"),
        ("plasticity", "Visco plasticity", 0, 1, 0.01, 0.8, "FLOAT"),
        ("restitution", "Restitution", 0, 1, 0.05, 0, "FLOAT"),
        ("substeps", "Substeps / frame", 1, 8, 1, 3, "INT"),
        ("maxSubDtMs", "Max sub-step (ms)", 2, 20, 0.1, 8.4, "FLOAT"),
    ),
}
METHOD_PROPERTY_PREFIX = {"PBF": "pbf", "MLS-MPM": "mlsmpm", "PB-MPM": "pbmpm"}
MAX_FLUID_EMITTERS = 16
MAX_FLUID_SINKS = 16
MAX_FLUID_POLYGON_POINTS = 256
MAX_SDF_VOXELS = 16 * 1024 * 1024


def blite_vec(value):
    return [float(value.x), float(value.z), float(-value.y)]


def blender_vec(value):
    return Vector((value[0], -value[2], value[1]))


def object_id(obj):
    return re.sub(r"[^a-z0-9_-]+", "-", obj.name.lower()).strip("-") or "flow"


def physics_property_name(method, key):
    return f"blitefluid_{METHOD_PROPERTY_PREFIX[method]}_{key}"


def physics_values(scene, method):
    return {key: getattr(scene, physics_property_name(method, key)) for key, _, _, _, _, _, _ in PHYSICS_SCHEMAS[method]}


def fluid_modifier(obj, fluid_type):
    for modifier in obj.modifiers:
        if modifier.type == "FLUID" and modifier.fluid_type == fluid_type:
            return modifier
    return None


def flow_settings(obj):
    modifier = fluid_modifier(obj, "FLOW")
    return modifier.flow_settings if modifier is not None else None


def authored_property(obj, name, default):
    try:
        if obj.is_property_set(name):
            return getattr(obj, name)
    except TypeError:
        pass
    if name in obj:
        return obj[name]
    return getattr(obj, name, default)


def is_collision_object(obj):
    return bool(authored_property(obj, "blite_collision", False)) or fluid_modifier(obj, "EFFECTOR") is not None


def volume_rate(obj):
    value = float(authored_property(obj, "blite_volume_rate", 0))
    return value if value > 0 else None


def local_size(obj):
    corners = [Vector(corner) for corner in obj.bound_box]
    minimum = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maximum = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return maximum - minimum


def world_bounds(obj):
    # A liquid DOMAIN modifier evaluates to the current liquid surface, so obj.bound_box can
    # collapse around the flow. The authored domain cage is the original mesh data.
    points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices] if obj.type == "MESH" else [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    if not points:
        raise ValueError(f"{obj.name}: mesh has no vertices")
    minimum = Vector((min(v.x for v in points), min(v.y for v in points), min(v.z for v in points)))
    maximum = Vector((max(v.x for v in points), max(v.y for v in points), max(v.z for v in points)))
    return minimum, maximum


def shape_for(obj):
    size = local_size(obj)
    kind = str(authored_property(obj, "blite_shape", "box"))
    if kind == "sphere":
        return {"type": kind, "radius": float(max(size) * 0.5)}
    if kind == "cylinder":
        shape = {"type": kind, "radius": float(max(size.x, size.y) * 0.5), "height": float(size.z)}
        inner = float(authored_property(obj, "blite_inner_radius", 0))
        if inner > 0:
            shape["innerRadius"] = inner
        return shape
    if kind == "cone":
        return {
            "type": kind,
            "bottomRadius": float(max(size.x, size.y) * 0.5),
            "topRadius": float(authored_property(obj, "blite_top_radius", 0)),
            "height": float(size.z),
        }
    if kind == "capsule":
        return {"type": kind, "radius": float(max(size.x, size.y) * 0.5), "height": float(size.z)}
    if kind == "polygonPrism":
        points = json.loads(str(authored_property(obj, "blite_points", "[]")))
        if len(points) < 3:
            raise ValueError(f"{obj.name}: polygonPrism requires blite_points=[[x,z], ...]")
        return {"type": kind, "points": points, "thickness": float(size.z)}
    return {"type": "box", "size": [float(size.x), float(size.z), float(size.y)]}


def transform_for(obj, grid_position):
    position, rotation, scale = obj.matrix_world.decompose()
    basis = Quaternion((1, 0, 0), -math.pi * 0.5)
    converted = basis @ rotation @ basis.conjugated()
    world_position = blite_vec(position)
    return {
        "position": [world_position[index] - grid_position[index] for index in range(3)],
        "rotation": [float(converted.x), float(converted.y), float(converted.z), float(converted.w)],
        "scale": [float(scale.x), float(scale.z), float(scale.y)],
    }


def find_domain(scene):
    active = bpy.context.view_layer.objects.active
    candidates = ([active] if active is not None else []) + list(scene.objects)
    seen = set()
    for obj in candidates:
        if obj is None or obj.name in seen:
            continue
        seen.add(obj.name)
        modifier = fluid_modifier(obj, "DOMAIN")
        settings = modifier.domain_settings if modifier is not None else None
        if settings is not None and getattr(settings, "domain_type", "") == "LIQUID":
            return obj
    raise ValueError("No liquid Fluid Domain object was found")


def extract_flows(scene, grid_position):
    emitters = []
    pending_sinks = []
    object_ids = {}
    for obj in scene.objects:
        settings = flow_settings(obj)
        if settings is None or getattr(settings, "flow_type", "") != "LIQUID":
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
            "enabled": behavior == "GEOMETRY" or bool(getattr(settings, "use_inflow", True)),
            "behavior": "initial" if behavior == "GEOMETRY" else "inflow",
            "transform": transform_for(obj, grid_position),
            "shape": shape_for(obj),
            "sampling": "surface" if getattr(settings, "use_plane_init", False) else "volume",
            "velocity": blite_vec(velocity),
            "velocitySpace": "local",
            "spread": float(authored_property(obj, "blite_spread", 0)),
        }
        rate = volume_rate(obj)
        if behavior != "GEOMETRY" and rate is not None:
            emitter["volumeRate"] = rate
        emitters.append(emitter)

    inflow_ids = [emitter["id"] for emitter in emitters if emitter["behavior"] == "inflow"]
    sinks = []
    for obj, settings in pending_sinks:
        names = [name.strip() for name in str(authored_property(obj, "blite_targets", "")).split(",") if name.strip()]
        targets = [object_ids[name] for name in names if name in object_ids] or inflow_ids
        sink = {
            "id": object_ids[obj.name],
            "name": obj.name,
            "enabled": bool(getattr(settings, "use_inflow", True)),
            "transform": transform_for(obj, grid_position),
            "shape": shape_for(obj),
            "targets": targets,
        }
        rate = volume_rate(obj)
        if rate is not None:
            sink["volumeRate"] = rate
        sinks.append(sink)
    return emitters, sinks


def build_bvh(objects):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices = []
    triangles = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            base = len(vertices)
            vertices.extend(evaluated.matrix_world @ vertex.co for vertex in mesh.vertices)
            triangles.extend(tuple(base + index for index in triangle.vertices) for triangle in mesh.loop_triangles)
        finally:
            evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(vertices, triangles, all_triangles=True) if triangles else None


def grid_metrics(grid_size, resolution):
    cell_size = max(grid_size) / float(resolution - 1)
    dims = [max(2, int(math.ceil(axis / cell_size)) + 1) for axis in grid_size]
    voxel_count = dims[0] * dims[1] * dims[2]
    return cell_size, dims, voxel_count


def bake_collision(objects, grid_position, grid_size, resolution, window_manager):
    cell_size, dims, voxel_count = grid_metrics(grid_size, resolution)
    if voxel_count > MAX_SDF_VOXELS:
        raise ValueError(f"Collision grid has {voxel_count:,} voxels; lower SDF resolution")
    origin = [grid_position[index] - grid_size[index] * 0.5 for index in range(3)]
    bvh = build_bvh(objects)
    empty_distance = max(grid_size)
    output = bytearray(64 + voxel_count * 4)
    struct.pack_into(
        "<6I4f",
        output,
        0,
        0x46534C42,
        1,
        dims[0],
        dims[1],
        dims[2],
        0,
        origin[0],
        origin[1],
        origin[2],
        cell_size,
    )

    window_manager.progress_begin(0, dims[2])
    try:
        index = 0
        for z in range(dims[2]):
            for y in range(dims[1]):
                for x in range(dims[0]):
                    point_blite = (
                        origin[0] + x * cell_size,
                        origin[1] + y * cell_size,
                        origin[2] + z * cell_size,
                    )
                    distance = empty_distance
                    if bvh is not None:
                        point = blender_vec(point_blite)
                        nearest, normal, _, unsigned = bvh.find_nearest(point)
                        if nearest is not None:
                            distance = -unsigned if (point - nearest).dot(normal) < 0 else unsigned
                    struct.pack_into("<f", output, 64 + index * 4, float(distance))
                    index += 1
            window_manager.progress_update(z + 1)
    finally:
        window_manager.progress_end()
    return bytes(output)


def scene_meshes(scene, domain):
    flow_objects = {obj for obj in scene.objects if flow_settings(obj) is not None}
    visual_objects = [
        obj
        for obj in scene.objects
        if obj.type == "MESH" and obj != domain and obj not in flow_objects and not obj.hide_render and obj.visible_get()
    ]
    return visual_objects, [obj for obj in visual_objects if is_collision_object(obj)]


def bounds_overlap(first, second):
    return all(first[0][axis] <= second[1][axis] and first[1][axis] >= second[0][axis] for axis in range(3))


def collision_mesh_stats(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        return len(bm.faces), sum(1 for edge in bm.edges if not edge.is_manifold)
    finally:
        bm.free()
        evaluated.to_mesh_clear()


def validate_setup(context):
    scene = context.scene
    errors = []
    warnings = []
    summary = {}
    try:
        domain = find_domain(scene)
        domain_bounds = world_bounds(domain)
        domain_size = domain_bounds[1] - domain_bounds[0]
        grid_size = [float(domain_size.x), float(domain_size.z), float(domain_size.y)]
        if min(grid_size) <= 0:
            errors.append("Fluid domain must have a positive size on every axis")
        else:
            cell_size, dims, voxel_count = grid_metrics(grid_size, scene.blitefluid_sdf_resolution)
            summary.update(
                domain=domain,
                domain_bounds=domain_bounds,
                grid_size=grid_size,
                cell_size=cell_size,
                dims=dims,
                voxel_count=voxel_count,
            )
            if voxel_count > MAX_SDF_VOXELS:
                errors.append(f"Collision grid has {voxel_count:,} voxels; lower SDF resolution")
    except ValueError as error:
        errors.append(str(error))
        return errors, warnings, summary

    liquid_flows = []
    emitter_count = 0
    sink_count = 0
    inflow_names = {
        obj.name
        for obj in scene.objects
        if (settings := flow_settings(obj)) is not None
        and getattr(settings, "flow_type", "") == "LIQUID"
        and settings.flow_behavior == "INFLOW"
    }
    for obj in scene.objects:
        settings = flow_settings(obj)
        if settings is None or getattr(settings, "flow_type", "") != "LIQUID":
            continue
        liquid_flows.append((obj, settings))
        if settings.flow_behavior == "OUTFLOW":
            sink_count += 1
        else:
            emitter_count += 1
        try:
            shape = shape_for(obj)
            if shape["type"] == "polygonPrism" and len(shape["points"]) > MAX_FLUID_POLYGON_POINTS:
                errors.append(f"{obj.name}: polygonPrism exceeds {MAX_FLUID_POLYGON_POINTS} points")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(str(error))
        try:
            if not bounds_overlap(world_bounds(obj), summary["domain_bounds"]):
                warnings.append(f"{obj.name}: flow object is outside the liquid domain")
        except ValueError as error:
            errors.append(str(error))
        if settings.flow_behavior == "OUTFLOW":
            targets = [name.strip() for name in str(authored_property(obj, "blite_targets", "")).split(",") if name.strip()]
            missing = [name for name in targets if name not in inflow_names]
            if missing:
                warnings.append(f"{obj.name}: unresolved sink targets: {', '.join(missing)}")

    if emitter_count > MAX_FLUID_EMITTERS:
        errors.append(f"Fluid flow supports at most {MAX_FLUID_EMITTERS} emitters")
    if sink_count > MAX_FLUID_SINKS:
        errors.append(f"Fluid flow supports at most {MAX_FLUID_SINKS} sinks")
    if emitter_count == 0:
        warnings.append("No liquid initial-volume or inflow objects were found")

    visual_objects, collision_objects = scene_meshes(scene, summary["domain"])
    summary.update(visual_objects=visual_objects, collision_objects=collision_objects, emitter_count=emitter_count, sink_count=sink_count)
    if not visual_objects:
        errors.append("No visible presentation meshes are available for scene.glb")
    if not collision_objects:
        warnings.append("No liquid effectors or Babylon Lite collider meshes were found")
    for obj in collision_objects:
        face_count, open_edges = collision_mesh_stats(obj)
        if face_count == 0:
            warnings.append(f"{obj.name}: collision mesh has no faces")
        if open_edges:
            warnings.append(f"{obj.name}: collision mesh has {open_edges} non-manifold edges")
        try:
            if not bounds_overlap(world_bounds(obj), summary["domain_bounds"]):
                warnings.append(f"{obj.name}: collision mesh is outside the liquid domain")
        except ValueError as error:
            errors.append(str(error))
    return errors, warnings, summary


def default_preset(scene, grid_position, grid_size, emitters, sinks):
    return {
        "formatVersion": 5,
        "meta": {"demo": "blender", "method": scene.blitefluid_method},
        "physics": physics_values(scene, scene.blitefluid_method),
        "demoParams": {},
        "demoState": {},
        "simulationDuration": scene.blitefluid_duration,
        "alphaDecay": scene.blitefluid_alpha_decay,
        "emitters": emitters,
        "sinks": sinks,
        "showContainer": False,
        "envIntensity": 1,
        "msaa": True,
        "activeBlocks": False,
        "pagedGrid": False,
        "fusedBlockDiscovery": False,
        "physicsParticleSize": scene.blitefluid_particle_size,
        "gridPosition": grid_position,
        "gridSize": grid_size,
        "showGridBounds": False,
        "particleCount": scene.blitefluid_particle_count,
        "material": 0,
        "render": {
            "renderAsSpheres": False,
            "waterColor": "#16a3c3",
            "absorption": 1,
            "particleSize": 0.7,
            "refractionStrength": 0.1,
            "specularPower": 250,
            "reflectionExposure": 2,
            "reflectionContrast": 0.6,
            "waterReflectivity": 0.02,
            "surfaceDepthBlur": 3,
            "depthBlurEdgeThreshold": 0.05,
            "surfaceThicknessBlur": 1,
            "halfRendering": False,
            "thicknessDownscale": 1,
            "surfaceFilter": "bilateral",
            "narrowRangeDelta": 1,
            "narrowRangeMu": 1,
            "anisotropicSurface": False,
            "anisoRadiusDamping": 0.2,
        },
        "foam": {
            "enableFoam": False,
            "activeParticles": False,
            "trappedAirRate": 40,
            "waveCrestRate": 40,
            "foamLifetime": 2,
            "foamLifetimeMin": 0.3,
            "bubbleBuoyancy": 0.8,
            "bubbleDrag": 0.5,
            "poolSize": 3,
            "foamSoftness": 1,
            "foamDensity": 1,
            "subsurfaceBubbleStrength": 0,
            "subsurfaceBubbleColor": "#ffffff",
            "foamBlurRadius": 2,
            "foamLightIntensity": 1,
            "foamAmbient": 0.2,
            "foamAO": 0,
            "foamNormalStrength": 1,
            "foamDebug": "none",
            "foamSize": 1,
        },
    }


def export_glb(path, objects):
    if not objects:
        raise ValueError("No visible presentation meshes are available for scene.glb")
    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    previous_selected = list(bpy.context.selected_objects)
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        view_layer.objects.active = objects[0]
        bpy.ops.export_scene.gltf(
            filepath=path,
            export_format="GLB",
            use_selection=True,
            export_apply=True,
            export_yup=True,
        )
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in previous_selected:
            if obj.name in view_layer.objects:
                obj.select_set(True)
        view_layer.objects.active = previous_active


def zip_entry(name, data):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o600 << 16
    return info, data


def export_bundle(context, filepath):
    scene = context.scene
    errors, warnings, summary = validate_setup(context)
    if errors:
        raise ValueError("; ".join(errors))
    for warning in warnings:
        print(f"[blitefluid] warning: {warning}")

    domain = summary["domain"]
    minimum, maximum = summary["domain_bounds"]
    center_blender = (minimum + maximum) * 0.5
    size_blender = maximum - minimum
    grid_position = blite_vec(center_blender)
    grid_size = [float(size_blender.x), float(size_blender.z), float(size_blender.y)]

    emitters, sinks = extract_flows(scene, grid_position)
    visual_objects = summary["visual_objects"]
    collision_objects = summary["collision_objects"]
    collision = bake_collision(
        collision_objects,
        grid_position,
        grid_size,
        scene.blitefluid_sdf_resolution,
        context.window_manager,
    )
    preset = default_preset(scene, grid_position, grid_size, emitters, sinks)
    manifest = {
        "bundleVersion": 1,
        "generator": {"name": "Babylon Lite Blender add-on", "version": "1.2.0"},
        "preset": preset,
        "scene": {"glb": "scene.glb", "collision": "collision.blsdf"},
    }

    with tempfile.TemporaryDirectory(prefix="blitefluid-") as temporary:
        glb_path = os.path.join(temporary, "scene.glb")
        export_glb(glb_path, visual_objects)
        with open(glb_path, "rb") as stream:
            glb = stream.read()
        with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
            for name, payload in (
                ("manifest.json", json.dumps(manifest, indent=2).encode("utf-8") + b"\n"),
                ("scene.glb", glb),
                ("collision.blsdf", collision),
            ):
                info, data = zip_entry(name, payload)
                archive.writestr(info, data)
    return len(emitters), len(sinks), len(collision_objects)


ROLE_ITEMS = (
    ("DOMAIN", "Domain", "Use the selected mesh as the liquid simulation domain"),
    ("INITIAL", "Initial volume", "Seed fluid from the selected mesh when the simulation starts"),
    ("INFLOW", "Inflow", "Continuously emit fluid from the selected mesh"),
    ("SINK", "Sink", "Recycle particles entering the selected mesh"),
    ("COLLIDER", "Collider", "Use the selected mesh as a visible static collision object"),
)


def ensure_fluid_role(context, obj, role):
    modifier = next((candidate for candidate in obj.modifiers if candidate.type == "FLUID"), None)
    if modifier is None:
        modifier = obj.modifiers.new("Babylon Lite Fluid", "FLUID")
    fluid_type = "DOMAIN" if role == "DOMAIN" else "EFFECTOR" if role == "COLLIDER" else "FLOW"
    modifier.fluid_type = fluid_type
    context.view_layer.update()
    if role == "DOMAIN":
        if modifier.domain_settings is None:
            raise ValueError("Blender did not initialize liquid domain settings")
        modifier.domain_settings.domain_type = "LIQUID"
    elif role == "COLLIDER":
        obj.blite_collision = True
    else:
        if modifier.flow_settings is None:
            raise ValueError("Blender did not initialize liquid flow settings")
        modifier.flow_settings.flow_type = "LIQUID"
        modifier.flow_settings.flow_behavior = {"INITIAL": "GEOMETRY", "INFLOW": "INFLOW", "SINK": "OUTFLOW"}[role]
        if role in {"INFLOW", "SINK"}:
            modifier.flow_settings.use_inflow = True
    context.view_layer.update()


class BLITEFLUID_OT_set_role(bpy.types.Operator):
    bl_idname = "object.blitefluid_set_role"
    bl_label = "Set Babylon Lite Fluid Role"
    bl_options = {"REGISTER", "UNDO"}

    role: EnumProperty(items=ROLE_ITEMS)

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        try:
            ensure_fluid_role(context, context.active_object, self.role)
        except ValueError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"{context.active_object.name}: {dict((value, label) for value, label, _ in ROLE_ITEMS)[self.role]}")
        return {"FINISHED"}


class BLITEFLUID_OT_validate(bpy.types.Operator):
    bl_idname = "scene.blitefluid_validate"
    bl_label = "Validate Babylon Lite Fluid"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            errors, warnings, summary = validate_setup(context)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        for error in errors:
            print(f"[blitefluid] error: {error}")
        for warning in warnings:
            print(f"[blitefluid] warning: {warning}")
        if errors:
            context.scene.blitefluid_status = f"{len(errors)} error(s), {len(warnings)} warning(s)"
            self.report({"ERROR"}, f"Validation failed: {errors[0]}")
            return {"CANCELLED"}
        context.scene.blitefluid_status = (
            f"{summary['emitter_count']} emitter(s), {summary['sink_count']} sink(s), "
            f"{len(summary['collision_objects'])} collider(s); {len(warnings)} warning(s)"
        )
        if warnings:
            self.report({"WARNING"}, f"Valid with {len(warnings)} warning(s); see the system console")
        else:
            self.report({"INFO"}, "Babylon Lite fluid setup is valid")
        return {"FINISHED"}


class BLITEFLUID_OT_export(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.blitefluid"
    bl_label = "Export Babylon Lite Fluid"
    bl_options = {"REGISTER"}

    filename_ext = ".blitefluid"
    filter_glob: bpy.props.StringProperty(default="*.blitefluid", options={"HIDDEN"})

    def execute(self, context):
        try:
            emitters, sinks, colliders = export_bundle(context, self.filepath)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        context.scene.blitefluid_status = f"Exported {emitters} emitter(s), {sinks} sink(s), {colliders} collider(s)"
        self.report({"INFO"}, f"Exported {emitters} emitters, {sinks} sinks, and {colliders} colliders")
        return {"FINISHED"}


class BLITEFLUID_PT_export(bpy.types.Panel):
    bl_label = "Babylon Lite Fluid"
    bl_idname = "BLITEFLUID_PT_export"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        domain_box = layout.box()
        domain_box.label(text="Simulation Domain", icon="MOD_FLUIDSIM")
        try:
            domain = find_domain(scene)
            minimum, maximum = world_bounds(domain)
            center = blite_vec((minimum + maximum) * 0.5)
            size_blender = maximum - minimum
            size = [float(size_blender.x), float(size_blender.z), float(size_blender.y)]
            _, dims, voxel_count = grid_metrics(size, scene.blitefluid_sdf_resolution)
            domain_box.label(text=domain.name)
            domain_box.label(text=f"Center: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}")
            domain_box.label(text=f"Size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}")
            domain_box.label(text=f"SDF: {dims[0]} x {dims[1]} x {dims[2]} ({voxel_count * 4 / (1024 * 1024):.1f} MiB)")
        except ValueError:
            domain_box.label(text="No liquid domain found", icon="ERROR")

        settings_box = layout.box()
        settings_box.label(text="Babylon Lite Simulation")
        settings_box.prop(scene, "blitefluid_method")
        settings_box.prop(scene, "blitefluid_particle_count")
        settings_box.prop(scene, "blitefluid_particle_size")
        settings_box.prop(scene, "blitefluid_sdf_resolution")
        settings_box.prop(scene, "blitefluid_duration")
        settings_box.prop(scene, "blitefluid_alpha_decay")
        physics_box = settings_box.box()
        physics_box.label(text=f"{scene.blitefluid_method} Physics")
        for key, _, _, _, _, _, _ in PHYSICS_SCHEMAS[scene.blitefluid_method]:
            physics_box.prop(scene, physics_property_name(scene.blitefluid_method, key))

        object_box = layout.box()
        object_box.label(text="Selected Mesh Authoring", icon="OBJECT_DATA")
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            object_box.label(text="Select a mesh object")
        else:
            object_box.label(text=obj.name)
            first_row = object_box.row(align=True)
            for role, label, _ in ROLE_ITEMS[:3]:
                operator = first_row.operator(BLITEFLUID_OT_set_role.bl_idname, text=label)
                operator.role = role
            second_row = object_box.row(align=True)
            for role, label, _ in ROLE_ITEMS[3:]:
                operator = second_row.operator(BLITEFLUID_OT_set_role.bl_idname, text=label)
                operator.role = role
            settings = flow_settings(obj)
            if settings is None and fluid_modifier(obj, "DOMAIN") is None:
                object_box.prop(obj, "blite_collision")
            if settings is not None:
                object_box.prop(settings, "flow_behavior")
                object_box.prop(obj, "blite_shape")
                if obj.blite_shape == "cylinder":
                    object_box.prop(obj, "blite_inner_radius")
                elif obj.blite_shape == "cone":
                    object_box.prop(obj, "blite_top_radius")
                elif obj.blite_shape == "polygonPrism":
                    object_box.prop(obj, "blite_points")
                if settings.flow_behavior != "GEOMETRY":
                    object_box.prop(obj, "blite_volume_rate")
                if settings.flow_behavior == "OUTFLOW":
                    object_box.prop(obj, "blite_targets")
                else:
                    object_box.prop(obj, "blite_spread")
                    object_box.prop(settings, "use_plane_init")
                    object_box.prop(settings, "use_initial_velocity")
                    if settings.use_initial_velocity:
                        object_box.prop(settings, "velocity_coord")

        layout.operator(BLITEFLUID_OT_validate.bl_idname, icon="CHECKMARK")
        layout.operator(BLITEFLUID_OT_export.bl_idname, icon="EXPORT")
        if scene.blitefluid_status:
            layout.label(text=scene.blitefluid_status)


CLASSES = (BLITEFLUID_OT_set_role, BLITEFLUID_OT_validate, BLITEFLUID_OT_export, BLITEFLUID_PT_export)


def menu_func_export(self, context):
    self.layout.operator(BLITEFLUID_OT_export.bl_idname, text="Babylon Lite Fluid (.blitefluid)")


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.types.Scene.blitefluid_method = EnumProperty(
        name="Simulation method",
        items=(("PBF", "PBF", ""), ("MLS-MPM", "MLS-MPM", ""), ("PB-MPM", "PB-MPM", "")),
        default="PBF",
    )
    bpy.types.Scene.blitefluid_particle_count = IntProperty(name="Particle capacity", default=100000, min=1000, max=1000000)
    bpy.types.Scene.blitefluid_particle_size = FloatProperty(name="Physics particle size", default=1, min=0.1, max=8, step=1)
    bpy.types.Scene.blitefluid_sdf_resolution = IntProperty(name="Collision SDF resolution", default=64, min=8, max=192)
    bpy.types.Scene.blitefluid_duration = FloatProperty(name="Simulation duration", default=0, min=0, soft_max=60)
    bpy.types.Scene.blitefluid_alpha_decay = FloatProperty(name="Alpha decay", default=2, min=0, soft_max=10)
    bpy.types.Scene.blitefluid_status = StringProperty(name="Validation status", default="")
    for method, schema in PHYSICS_SCHEMAS.items():
        for key, label, minimum, maximum, step, default, kind in schema:
            property_name = physics_property_name(method, key)
            if kind == "INT":
                property_value = IntProperty(name=label, default=int(default), min=int(minimum), max=int(maximum), step=max(1, int(step)))
            else:
                precision = max(2, min(4, int(math.ceil(-math.log10(step))) if step < 1 else 2))
                property_value = FloatProperty(
                    name=label,
                    default=float(default),
                    min=float(minimum),
                    max=float(maximum),
                    step=max(1, min(100, int(round(step * 100)))),
                    precision=precision,
                )
            setattr(bpy.types.Scene, property_name, property_value)
    bpy.types.Object.blite_shape = EnumProperty(
        name="Emitter / sink shape",
        items=(
            ("box", "Box", ""),
            ("sphere", "Sphere", ""),
            ("cylinder", "Cylinder", ""),
            ("cone", "Cone", ""),
            ("capsule", "Capsule", ""),
            ("polygonPrism", "Polygon prism", ""),
        ),
        default="box",
    )
    bpy.types.Object.blite_volume_rate = FloatProperty(name="Volume rate (0 = unlimited)", default=0, min=0, soft_max=100)
    bpy.types.Object.blite_targets = StringProperty(name="Sink targets", description="Comma-separated inflow object names", default="")
    bpy.types.Object.blite_inner_radius = FloatProperty(name="Inner radius", default=0, min=0)
    bpy.types.Object.blite_top_radius = FloatProperty(name="Top radius", default=0, min=0)
    bpy.types.Object.blite_points = StringProperty(name="Polygon points", description='JSON array such as [[-1,-1],[1,-1],[1,1],[-1,1]]', default="[]")
    bpy.types.Object.blite_spread = FloatProperty(name="Velocity spread", default=0, min=0, max=1)
    bpy.types.Object.blite_collision = BoolProperty(name="Babylon Lite collider", default=False)


def unregister():
    del bpy.types.Object.blite_collision
    del bpy.types.Object.blite_spread
    del bpy.types.Object.blite_points
    del bpy.types.Object.blite_top_radius
    del bpy.types.Object.blite_inner_radius
    del bpy.types.Object.blite_targets
    del bpy.types.Object.blite_volume_rate
    del bpy.types.Object.blite_shape
    for method, schema in PHYSICS_SCHEMAS.items():
        for key, _, _, _, _, _, _ in schema:
            delattr(bpy.types.Scene, physics_property_name(method, key))
    del bpy.types.Scene.blitefluid_status
    del bpy.types.Scene.blitefluid_alpha_decay
    del bpy.types.Scene.blitefluid_duration
    del bpy.types.Scene.blitefluid_sdf_resolution
    del bpy.types.Scene.blitefluid_particle_size
    del bpy.types.Scene.blitefluid_particle_count
    del bpy.types.Scene.blitefluid_method
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
