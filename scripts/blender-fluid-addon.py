"""Fluid scene exporter for native and add-on simulation setups."""

bl_info = {
    "name": "Babylon Lite Fluid JSON",
    "author": "Babylon Lite contributors",
    "version": (3, 2, 0),
    "blender": (4, 0, 0),
    "location": "Properties > Scene > Babylon Lite Fluid; File > Export",
    "description": "Export a native or add-on fluid setup to Babylon Lite JSON",
    "category": "Import-Export",
}

import base64
import json
import math
import os
import re
import struct
import tempfile

import bmesh
import bpy
from bpy.props import IntProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree

MAX_FLUID_EMITTERS = 16
MAX_FLUID_SINKS = 16
MAX_FLUID_POLYGON_POINTS = 256
MAX_SDF_VOXELS = 16 * 1024 * 1024
PBF_BASE_CELL_SIZE = 0.4
PBF_PARTICLE_SIZE_CALIBRATION = 2.1333333333333333
FLIP_BASE_CELL_SIZE = 0.25
FLIP_MARKERS_PER_CELL = 8
BLITE_INFLOW_VOLUME_RATE = 20.0


def blite_vec(value):
    return [float(value.x), float(value.z), float(-value.y)]


def blender_vec(value):
    return Vector((value[0], -value[2], value[1]))


def object_id(obj):
    return re.sub(r"[^a-z0-9_-]+", "-", obj.name.lower()).strip("-") or "flow"


def fluid_modifier(obj, fluid_type):
    for modifier in obj.modifiers:
        if modifier.type == "FLUID" and modifier.fluid_type == fluid_type:
            return modifier
    return None


def flow_settings(obj):
    modifier = fluid_modifier(obj, "FLOW")
    return modifier.flow_settings if modifier is not None else None


def flip_object_type(obj):
    props = getattr(obj, "flip_fluid", None)
    return getattr(props, "object_type", "TYPE_NONE") if props is not None and getattr(props, "is_active", False) else "TYPE_NONE"


def is_flip_fluids_domain(obj):
    return flip_object_type(obj) == "TYPE_DOMAIN"


def is_flip_fluids_flow(obj):
    return flip_object_type(obj) in {"TYPE_FLUID", "TYPE_INFLOW", "TYPE_OUTFLOW"}


def is_collision_object(obj):
    if flip_object_type(obj) == "TYPE_OBSTACLE":
        return bool(getattr(obj.flip_fluid.obstacle, "is_enabled", True))
    modifier = fluid_modifier(obj, "EFFECTOR")
    settings = modifier.effector_settings if modifier is not None else None
    return settings is not None and settings.effector_type == "COLLISION" and settings.use_effector


def is_flow_object(obj):
    return (is_flip_fluids_flow(obj) or flow_settings(obj) is not None) and not is_collision_object(obj)


def local_size(obj):
    corners = [Vector(corner) for corner in obj.bound_box]
    minimum = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maximum = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return maximum - minimum


def world_bounds(obj, transform=None):
    # A liquid DOMAIN modifier evaluates to the current liquid surface, so obj.bound_box can
    # collapse around the flow. The authored domain cage is the original mesh data.
    transform = transform if transform is not None else obj.matrix_world
    points = [transform @ vertex.co for vertex in obj.data.vertices] if obj.type == "MESH" else [transform @ Vector(corner) for corner in obj.bound_box]
    if not points:
        raise ValueError(f"{obj.name}: mesh has no vertices")
    minimum = Vector((min(v.x for v in points), min(v.y for v in points), min(v.z for v in points)))
    maximum = Vector((max(v.x for v in points), max(v.y for v in points), max(v.z for v in points)))
    return minimum, maximum


def authored_domain_bounds(domain):
    # Some saved Mantaflow domains expose a stale identity matrix_world while their
    # unparented authored transform remains valid in matrix_basis.
    transform = domain.matrix_basis if domain.parent is None and not domain.constraints else domain.matrix_world
    return world_bounds(domain, transform)


def shape_for(obj):
    size = local_size(obj)
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
        if is_flip_fluids_domain(obj):
            return obj
        modifier = fluid_modifier(obj, "DOMAIN")
        settings = modifier.domain_settings if modifier is not None else None
        if settings is not None and getattr(settings, "domain_type", "") == "LIQUID":
            return obj
    raise ValueError("No native or add-on liquid Domain object was found")


def clamp(value, minimum, maximum):
    return min(maximum, max(minimum, value))


def derived_domain_values(scene, domain, grid_size):
    if is_flip_fluids_domain(domain):
        domain_props = domain.flip_fluid.domain
        simulation = domain_props.simulation
        advanced = domain_props.advanced
        world = domain_props.world
        resolution = max(1, int(simulation.resolution))
        cell_size = max(grid_size) / resolution
        raw_particle_size = cell_size / FLIP_BASE_CELL_SIZE
        particle_size = clamp(raw_particle_size, 0.1, 8)
        cells = [max(1, int(math.ceil(axis / cell_size))) for axis in grid_size]
        raw_particle_count = cells[0] * cells[1] * cells[2] * FLIP_MARKERS_PER_CELL
        particle_count = max(1, int(raw_particle_count))
        gravity_vector = Vector(world.gravity)
        if getattr(world, "gravity_type", "GRAVITY_TYPE_SCENE") == "GRAVITY_TYPE_SCENE":
            gravity_vector = Vector(scene.gravity) if scene.use_gravity else Vector((0, 0, 0))
        time_steps = advanced.min_max_time_steps_per_frame
        min_substeps = int(clamp(int(time_steps.value_min), 1, 16))
        max_substeps = int(clamp(max(min_substeps, int(time_steps.value_max)), 1, 32))
        pressure_iterations = int(clamp(int(advanced.pressure_solver_max_iterations), 1, 100))
        pressure_tolerance = clamp(float(getattr(advanced, "pressure_solver_error_tolerance", 1e-3)), 0, 0.1)
        surface = getattr(domain_props, "surface", None)
        particle_sheeting = bool(getattr(surface, "enable_sheet_seeding", False))
        sheeting_strength = clamp(float(getattr(surface, "sheet_fill_rate", 0.5)), 0.05, 1)
        return {
            "method": "FLIP",
            "particle_count": particle_count,
            "raw_particle_count": raw_particle_count,
            "particle_size": particle_size,
            "raw_particle_size": raw_particle_size,
            "cell_size": cell_size,
            "cells": cells,
            "resolution": resolution,
            "markers_per_cell": FLIP_MARKERS_PER_CELL,
            "physics": {
                "gravity": clamp(float(-gravity_vector.z), 0, 200),
                "flipRatio": clamp(1.0 - float(advanced.PICFLIP_ratio), 0, 1),
                "kinematicViscosity": 0,
                "surfaceTension": 0,
                "minSubsteps": min_substeps,
                "maxSubsteps": max_substeps,
                "cflNumber": clamp(float(getattr(advanced, "CFL_condition_number", 2)), 0, 10),
                "restitution": 0,
                "velocityDamping": 0,
                "pressureSolver": 1,
                "pressureIterations": pressure_iterations,
                "pressureRelaxation": 0.8,
                "multigridCycles": int(clamp(math.ceil(pressure_iterations / 25), 2, 8)),
                "pressureTolerance": pressure_tolerance,
                "pressureDiagnostics": 1,
                "liquidSdf": 1,
                "ghostFluid": 1,
                "fractionalSolids": 1,
                "movingSolidBoundaries": 1,
                "reseedParticles": 1,
                "reseedMinParticles": max(1, FLIP_MARKERS_PER_CELL // 2),
                "reseedTargetParticles": FLIP_MARKERS_PER_CELL,
                "reseedMaxParticles": int(math.ceil(FLIP_MARKERS_PER_CELL * 1.5)),
                "reseedInterval": 5,
                "particleSheeting": 1 if particle_sheeting else 0,
                "sheetingStrength": sheeting_strength,
                "sheetingInterval": 5,
                "polygonSurface": 1,
                "viscosityIterations": 12,
                "maxSubDtMs": 8.4,
            },
        }

    modifier = fluid_modifier(domain, "DOMAIN")
    settings = modifier.domain_settings if modifier is not None else None
    if settings is None:
        raise ValueError("Liquid domain settings are unavailable")

    resolution = max(1, int(settings.resolution_max))
    cell_size = max(grid_size) / resolution
    raw_particle_size = (cell_size / PBF_BASE_CELL_SIZE) * PBF_PARTICLE_SIZE_CALIBRATION
    particle_size = clamp(raw_particle_size, 0.7, 8)
    cells = [max(1, int(math.ceil(axis / cell_size))) for axis in grid_size]
    explicit_limit = max(0, int(settings.sys_particle_maximum))
    raw_particle_count = explicit_limit or cells[0] * cells[1] * cells[2] * max(1, int(settings.particle_max))
    particle_count = max(1, int(raw_particle_count))

    viscosity = 0.08
    if settings.use_diffusion:
        physical_viscosity = max(1e-8, float(settings.viscosity_base) * (10 ** -int(settings.viscosity_exponent)))
        viscosity = clamp(0.08 + 0.35 * max(0, math.log10(physical_viscosity / 1e-6)), 0, 1.9)
    physics = {
        "gravity": clamp(float(-settings.gravity.z), 0, 200),
        "viscosity": viscosity,
        "relaxation": clamp(100 / max(float(settings.cfl_condition), 0.1), 1, 1000),
        "scorr": clamp(float(settings.surface_tension) / 120, 0, 0.5) if settings.use_diffusion else 0.02,
        "iterations": int(clamp(int(settings.timesteps_max), 1, 8)),
        "restDensity": 341,
        "boundaryDensity": 0,
    }

    return {
        "method": "PBF",
        "particle_count": particle_count,
        "raw_particle_count": raw_particle_count,
        "particle_size": particle_size,
        "raw_particle_size": raw_particle_size,
        "cell_size": cell_size,
        "cells": cells,
        "resolution": resolution,
        "physics": physics,
    }


def extract_flows(scene, grid_position):
    emitters = []
    pending_sinks = []
    object_ids = {}
    domain = find_domain(scene)
    domain_bounds = authored_domain_bounds(domain)
    domain_size = domain_bounds[1] - domain_bounds[0]
    grid_size = [float(domain_size.x), float(domain_size.z), float(domain_size.y)]
    derived = derived_domain_values(scene, domain, grid_size)
    flow_objects = [obj for obj in scene.objects if is_flow_object(obj)]
    for obj in flow_objects:
        flow_id = object_id(obj)
        suffix = 2
        base_id = flow_id
        while flow_id in object_ids.values():
            flow_id = f"{base_id}-{suffix}"
            suffix += 1
        object_ids[obj.name] = flow_id
        flip_type = flip_object_type(obj)
        if flip_type == "TYPE_OUTFLOW":
            pending_sinks.append((obj, obj.flip_fluid.outflow, "FLIP"))
            continue
        if flip_type in {"TYPE_FLUID", "TYPE_INFLOW"}:
            settings = obj.flip_fluid.fluid if flip_type == "TYPE_FLUID" else obj.flip_fluid.inflow
            behavior = "initial" if flip_type == "TYPE_FLUID" else "inflow"
            velocity = Vector(settings.initial_velocity if flip_type == "TYPE_FLUID" else settings.inflow_velocity)
            enabled = True if flip_type == "TYPE_FLUID" else bool(settings.is_enabled)
            append_velocity = bool(settings.append_object_velocity)
            velocity_factor = float(settings.append_object_velocity_influence)
            sampling = "volume"
        else:
            settings = flow_settings(obj)
            if settings is None or getattr(settings, "flow_type", "") != "LIQUID":
                continue
            if settings.flow_behavior == "OUTFLOW":
                pending_sinks.append((obj, settings, "MANTA"))
                continue
            behavior = "initial" if settings.flow_behavior == "GEOMETRY" else "inflow"
            use_initial_velocity = bool(settings.use_initial_velocity)
            velocity = Vector(settings.velocity_coord) if use_initial_velocity else Vector((0, 0, 0))
            enabled = behavior == "initial" or bool(getattr(settings, "use_inflow", True))
            append_velocity = use_initial_velocity
            velocity_factor = float(settings.velocity_factor)
            sampling = "surface" if settings.use_plane_init else "volume"
        transform = transform_for(obj, grid_position)
        shape = shape_for(obj)
        emitter = {
            "id": flow_id,
            "name": obj.name,
            "enabled": enabled,
            "behavior": behavior,
            "transform": transform,
            "shape": shape,
            "sampling": sampling,
            "velocity": blite_vec(velocity),
            "velocitySpace": "world",
            "sourceNode": obj.name,
            "spread": 0,
        }
        if append_velocity:
            emitter["sourceVelocityFactor"] = velocity_factor
        if flip_type == "TYPE_NONE" and bool(settings.use_initial_velocity):
            emitter["normalVelocity"] = float(settings.velocity_normal)
        if behavior == "inflow":
            emitter["volumeRate"] = BLITE_INFLOW_VOLUME_RATE
        emitters.append(emitter)

    sinks = []
    for obj, settings, source_kind in pending_sinks:
        transform = transform_for(obj, grid_position)
        shape = shape_for(obj)
        if source_kind == "MANTA":
            surface_margin = max(0.0, float(settings.surface_distance)) * derived["cell_size"]
            shape["size"] = [
                size + 2 * surface_margin / max(abs(transform["scale"][index]), 1e-6)
                for index, size in enumerate(shape["size"])
            ]
            enabled = bool(getattr(settings, "use_inflow", True))
        else:
            enabled = bool(settings.is_enabled and settings.remove_fluid)
        sink = {
            "id": object_ids[obj.name],
            "name": obj.name,
            "enabled": enabled,
            "transform": transform,
            "shape": shape,
            "mode": "delete",
            "targets": [],
        }
        if source_kind == "FLIP" and settings.enable_gradual_outflow:
            world_volume = math.prod(shape["size"][index] * abs(transform["scale"][index]) for index in range(3))
            sink["volumeRate"] = world_volume * max(0.0, float(settings.outflow_rate))
        sinks.append(sink)
    return emitters, sinks


def triangle_indices(triangle, base, reverse_winding):
    first, second, third = (base + index for index in triangle.vertices)
    return (first, third, second) if reverse_winding else (first, second, third)


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
            reverse_winding = evaluated.matrix_world.to_3x3().determinant() < 0
            triangles.extend(triangle_indices(triangle, base, reverse_winding) for triangle in mesh.loop_triangles)
        finally:
            evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(vertices, triangles, all_triangles=True) if triangles else None


def grid_metrics(grid_size, resolution):
    cell_size = max(grid_size) / float(resolution - 1)
    dims = [max(2, int(math.ceil(axis / cell_size)) + 1) for axis in grid_size]
    voxel_count = dims[0] * dims[1] * dims[2]
    return cell_size, dims, voxel_count


def collision_texture_metrics(scene):
    domain = find_domain(scene)
    domain_bounds = authored_domain_bounds(domain)
    domain_size = domain_bounds[1] - domain_bounds[0]
    grid_size = [float(domain_size.x), float(domain_size.z), float(domain_size.y)]
    _, dims, voxel_count = grid_metrics(grid_size, int(scene.blitefluid_sdf_resolution))
    return dims, voxel_count * 4


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


def scene_objects(scene, domain):
    presentation_objects = [
        obj
        for obj in scene.objects
        if obj != domain
        and (obj.type == "MESH" or (obj.type == "LIGHT" and obj.data.type in {"POINT", "SUN", "SPOT"}))
        and not obj.hide_render
        and obj.visible_get()
    ]
    visual_objects = [obj for obj in presentation_objects if obj.type == "MESH"]
    collision_objects = [obj for obj in scene.objects if obj != domain and obj.type == "MESH" and is_collision_object(obj)]
    return presentation_objects, visual_objects, collision_objects


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
        domain_bounds = authored_domain_bounds(domain)
        domain_size = domain_bounds[1] - domain_bounds[0]
        grid_size = [float(domain_size.x), float(domain_size.z), float(domain_size.y)]
        if min(grid_size) <= 0:
            errors.append("Fluid domain must have a positive size on every axis")
        else:
            sdf_resolution = int(scene.blitefluid_sdf_resolution)
            cell_size, dims, voxel_count = grid_metrics(grid_size, sdf_resolution)
            summary.update(
                domain=domain,
                domain_bounds=domain_bounds,
                grid_size=grid_size,
                cell_size=cell_size,
                dims=dims,
                voxel_count=voxel_count,
                sdf_resolution=sdf_resolution,
                derived=derived_domain_values(scene, domain, grid_size),
            )
            if voxel_count > MAX_SDF_VOXELS:
                errors.append(f"Collision grid has {voxel_count:,} voxels; lower SDF resolution")
    except ValueError as error:
        errors.append(str(error))
        return errors, warnings, summary

    liquid_flows = []
    emitter_count = 0
    sink_count = 0
    for obj in scene.objects:
        if is_collision_object(obj):
            continue
        flip_type = flip_object_type(obj)
        settings = flow_settings(obj)
        if not is_flip_fluids_flow(obj) and (settings is None or getattr(settings, "flow_type", "") != "LIQUID"):
            continue
        liquid_flows.append((obj, settings))
        is_outflow = flip_type == "TYPE_OUTFLOW" or (settings is not None and settings.flow_behavior == "OUTFLOW")
        if is_outflow:
            sink_count += 1
        else:
            emitter_count += 1
        try:
            shape_for(obj)
        except (TypeError, ValueError) as error:
            errors.append(str(error))
        try:
            flow_bounds = world_bounds(obj)
            if not bounds_overlap(flow_bounds, summary["domain_bounds"]):
                errors.append(f"{obj.name}: flow object is outside the liquid domain")
            elif (
                flip_type == "TYPE_NONE"
                and settings.flow_behavior == "OUTFLOW"
                and flow_bounds[1].z + max(0.0, float(settings.surface_distance)) * summary["derived"]["cell_size"]
                <= summary["domain_bounds"][0].z + summary["derived"]["cell_size"] * 2
            ):
                warnings.append(f"{obj.name}: floor-level outflow reaches less than two Mantaflow cells into the domain and may miss resting liquid")
        except ValueError as error:
            errors.append(str(error))

    if emitter_count > MAX_FLUID_EMITTERS:
        errors.append(f"Fluid flow supports at most {MAX_FLUID_EMITTERS} emitters")
    if sink_count > MAX_FLUID_SINKS:
        errors.append(f"Fluid flow supports at most {MAX_FLUID_SINKS} sinks")
    if emitter_count == 0:
        warnings.append("No liquid initial-volume or inflow objects were found")
    derived = summary["derived"]
    if abs(derived["particle_size"] - derived["raw_particle_size"]) > 1e-8:
        label = "FLIP cell-size compatibility scale" if derived["method"] == "FLIP" else "physics particle size"
        warnings.append(f"Derived {label} {derived['raw_particle_size']:.3g} will be clamped to {derived['particle_size']:.3g}")
    if derived["method"] == "PBF":
        domain_settings = fluid_modifier(summary["domain"], "DOMAIN").domain_settings
        if abs(domain_settings.gravity.x) > 1e-5 or abs(domain_settings.gravity.y) > 1e-5 or domain_settings.gravity.z > 1e-5:
            warnings.append("Babylon Lite currently uses only the downward source-scene Z gravity component")
    unsupported_lights = [
        obj.name
        for obj in scene.objects
        if obj.type == "LIGHT" and obj.data.type not in {"POINT", "SUN", "SPOT"} and not obj.hide_render and obj.visible_get()
    ]
    if unsupported_lights:
        warnings.append(f"glTF cannot export non-punctual lights; skipped: {', '.join(unsupported_lights)}")

    presentation_objects, visual_objects, collision_objects = scene_objects(scene, summary["domain"])
    summary.update(
        presentation_objects=presentation_objects,
        visual_objects=visual_objects,
        collision_objects=collision_objects,
        emitter_count=emitter_count,
        sink_count=sink_count,
    )
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


def json_setting_value(value):
    if isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "to_tuple"):
        return list(value.to_tuple())
    return None


def settings_snapshot(settings):
    output = {}
    for prop in settings.bl_rna.properties:
        if prop.identifier == "rna_type" or prop.is_readonly:
            continue
        try:
            value = json_setting_value(getattr(settings, prop.identifier))
        except Exception:
            continue
        if value is not None:
            output[prop.identifier] = value
    return output


def source_snapshot(scene, domain):
    flows = {}
    effectors = {}
    for obj in scene.objects:
        flip_type = flip_object_type(obj)
        if flip_type == "TYPE_FLUID":
            flows[obj.name] = settings_snapshot(obj.flip_fluid.fluid)
            continue
        if flip_type == "TYPE_INFLOW":
            flows[obj.name] = settings_snapshot(obj.flip_fluid.inflow)
            continue
        if flip_type == "TYPE_OUTFLOW":
            flows[obj.name] = settings_snapshot(obj.flip_fluid.outflow)
            continue
        if flip_type == "TYPE_OBSTACLE":
            effectors[obj.name] = settings_snapshot(obj.flip_fluid.obstacle)
            continue
        flow = flow_settings(obj)
        if flow is not None and flow.flow_type == "LIQUID":
            flows[obj.name] = settings_snapshot(flow)
        modifier = fluid_modifier(obj, "EFFECTOR")
        if modifier is not None and modifier.effector_settings is not None:
            effectors[obj.name] = settings_snapshot(modifier.effector_settings)
    if is_flip_fluids_domain(domain):
        dprops = domain.flip_fluid.domain
        domain_settings = {
            "simulation": settings_snapshot(dprops.simulation),
            "advanced": settings_snapshot(dprops.advanced),
            "world": settings_snapshot(dprops.world),
            "surface": settings_snapshot(dprops.surface),
            "whitewater": settings_snapshot(dprops.whitewater),
        }
        application = "FLIP add-on"
    else:
        domain_settings = settings_snapshot(fluid_modifier(domain, "DOMAIN").domain_settings)
        application = "Native fluid"
    return {
        "application": application,
        "version": bpy.app.version_string,
        "settings": {
            "timeline": {
                "frameStart": scene.frame_start,
                "frameEnd": scene.frame_end,
                "fps": scene.render.fps,
                "fpsBase": scene.render.fps_base,
            },
            "collisionSdfResolution": scene.blitefluid_sdf_resolution,
            "domain": domain_settings,
            "flows": flows,
            "effectors": effectors,
        },
    }


def default_preset(scene, grid_position, grid_size, emitters, sinks):
    domain = find_domain(scene)
    derived = derived_domain_values(scene, domain, grid_size)
    if is_flip_fluids_domain(domain):
        settings = domain.flip_fluid.domain
        foam_enabled = bool(getattr(settings.whitewater, "enable_whitewater_simulation", False))
        bubbles_enabled = foam_enabled
        simulation_time_scale = clamp(float(settings.simulation.time_scale), 0.01, 100)
        initial_volume = sum(
            math.prod(emitter["shape"]["size"][index] * abs(emitter["transform"]["scale"][index]) for index in range(3))
            for emitter in emitters
            if emitter["enabled"] and emitter["behavior"] == "initial" and emitter["sampling"] == "volume" and emitter["shape"]["type"] == "box"
        )
        initial_markers = int(math.ceil(initial_volume / (derived["cell_size"] ** 3) * derived["markers_per_cell"])) if initial_volume > 0 else 0
        has_inflow = any(emitter["enabled"] and emitter["behavior"] == "inflow" for emitter in emitters)
        particle_count = max(initial_markers, 80000 if has_inflow else 1)
    else:
        settings = fluid_modifier(domain, "DOMAIN").domain_settings
        foam_enabled = bool(settings.use_spray_particles or settings.use_bubble_particles or settings.use_foam_particles)
        bubbles_enabled = bool(settings.use_bubble_particles)
        simulation_time_scale = clamp(float(settings.time_scale), 0.01, 100)
        particle_count = derived["particle_count"]
    preset = {
        "formatVersion": 13,
        "meta": {"demo": "blender", "method": derived["method"]},
        "source": source_snapshot(scene, domain),
        "physics": derived["physics"],
        "demoParams": {},
        "demoState": {},
        "simulationDuration": 0,
        "alphaDecay": 0,
        "simulationTimeScale": simulation_time_scale,
        "emitters": emitters,
        "sinks": sinks,
        "showContainer": False,
        "envIntensity": 1,
        "msaa": True,
        "activeBlocks": False,
        "pagedGrid": False,
        "fusedBlockDiscovery": False,
        **({"physicsParticleSize": derived["particle_size"]} if derived["method"] != "FLIP" else {}),
        "gridPosition": grid_position,
        "gridSize": grid_size,
        **(
            {
                "gridResolution": derived["resolution"],
                "markersPerCell": derived["markers_per_cell"],
            }
            if derived["method"] == "FLIP"
            else {}
        ),
        "showGridBounds": False,
        "particleCount": particle_count,
        "material": 0,
        "render": {
            "renderAsSpheres": False,
            "waterColor": "#16a3c3",
            "absorption": 1,
            "particleSize": 0.6,
            "refractionStrength": 0.1,
            "specularPower": 250,
            "reflectionExposure": 2,
            "reflectionContrast": 0.6,
            "waterReflectivity": 0.02,
            "surfaceDepthBlur": 17,
            "depthBlurEdgeThreshold": 0.05,
            "surfaceThicknessBlur": 2,
            "halfRendering": True,
            "thicknessDownscale": 8,
            "surfaceFilter": "narrowRange",
            "narrowRangeDelta": 2,
            "narrowRangeMu": 1,
            "anisotropicSurface": False,
            "anisoRadiusDamping": 0.2,
        },
        "foam": {
            "enableFoam": foam_enabled,
            "activeParticles": False,
            "trappedAirRate": 51,
            "waveCrestRate": 48,
            "foamLifetime": 1.0416666666666667,
            "foamLifetimeMin": 0.4166666666666667,
            "bubbleBuoyancy": 4.2,
            "bubbleDrag": 0.45,
            "poolSize": 3.5,
            "foamSoftness": 0,
            "foamDensity": 8.25,
            "subsurfaceBubbleStrength": 0.2 if bubbles_enabled else 0,
            "subsurfaceBubbleColor": "#5380ea",
            "foamBlurRadius": 1,
            "foamLightIntensity": 1,
            "foamAmbient": 1,
            "foamAO": 0,
            "foamNormalStrength": 1,
            "foamDebug": "off",
            "foamSize": 0.15,
        },
    }
    return preset


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
            export_lights=True,
        )
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in previous_selected:
            if obj.name in view_layer.objects:
                obj.select_set(True)
        view_layer.objects.active = previous_active


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
    presentation_objects = summary["presentation_objects"]
    collision_objects = summary["collision_objects"]
    collision = bake_collision(
        collision_objects,
        grid_position,
        grid_size,
        summary["sdf_resolution"],
        context.window_manager,
    )
    preset = default_preset(scene, grid_position, grid_size, emitters, sinks)

    with tempfile.TemporaryDirectory(prefix="blitefluid-") as temporary:
        glb_path = os.path.join(temporary, "scene.glb")
        export_glb(glb_path, presentation_objects)
        with open(glb_path, "rb") as stream:
            glb = stream.read()
        preset["scene"] = {
            "encoding": "base64",
            "glb": base64.b64encode(glb).decode("ascii"),
            "collision": base64.b64encode(collision).decode("ascii"),
            "anchorPosition": grid_position,
        }
        with open(filepath, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(preset, stream, indent=2)
            stream.write("\n")
    return len(emitters), len(sinks), len(collision_objects), summary["dims"]


def ensure_fluid_role(context, obj, role):
    fluid_type = "DOMAIN" if role == "DOMAIN" else "EFFECTOR" if role == "COLLIDER" else "FLOW"
    fluid_modifiers = [candidate for candidate in obj.modifiers if candidate.type == "FLUID"]
    modifier = next((candidate for candidate in fluid_modifiers if candidate.fluid_type == fluid_type), None)
    for stale_modifier in fluid_modifiers:
        if stale_modifier != modifier:
            obj.modifiers.remove(stale_modifier)
    if modifier is None:
        modifier = obj.modifiers.new("Babylon Lite Fluid", "FLUID")
        modifier.fluid_type = fluid_type
    obj["blite_collision"] = role == "COLLIDER"
    context.view_layer.update()
    if role == "DOMAIN":
        if modifier.domain_settings is None:
            raise ValueError("The host application did not initialize liquid domain settings")
        modifier.domain_settings.domain_type = "LIQUID"
    elif role == "COLLIDER":
        if modifier.effector_settings is None:
            raise ValueError("The host application did not initialize fluid effector settings")
        modifier.effector_settings.effector_type = "COLLISION"
        if modifier.effector_settings.surface_distance <= 0:
            modifier.effector_settings.surface_distance = 1.5
    elif role != "COLLIDER":
        if modifier.flow_settings is None:
            raise ValueError("The host application did not initialize liquid flow settings")
        modifier.flow_settings.flow_type = "LIQUID"
        modifier.flow_settings.flow_behavior = {"INITIAL": "GEOMETRY", "INFLOW": "INFLOW", "SINK": "OUTFLOW"}[role]
        if role in {"INFLOW", "SINK"}:
            modifier.flow_settings.use_inflow = True
        if role == "SINK":
            modifier.flow_settings.surface_distance = 1.0
    context.view_layer.update()


class BLITEFLUID_OT_export(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.blitefluid"
    bl_label = "Export Babylon Lite Fluid JSON"
    bl_options = {"REGISTER"}

    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        try:
            emitters, sinks, colliders, dims = export_bundle(context, self.filepath)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {emitters} emitters, {sinks} sinks, {colliders} colliders; SDF {dims[0]} x {dims[1]} x {dims[2]}")
        return {"FINISHED"}


class BLITEFLUID_PT_export(bpy.types.Panel):
    bl_label = "Babylon Lite Fluid"
    bl_idname = "BLITEFLUID_PT_export"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"

    def draw(self, context):
        self.layout.prop(context.scene, "blitefluid_sdf_resolution")
        try:
            dims, texture_bytes = collision_texture_metrics(context.scene)
            self.layout.label(text=f"Texture: {dims[0]} x {dims[1]} x {dims[2]} R32Float, {texture_bytes / (1024 * 1024):.2f} MiB")
        except (AttributeError, TypeError, ValueError):
            self.layout.label(text="Texture size unavailable until a valid liquid domain exists")
        self.layout.operator(BLITEFLUID_OT_export.bl_idname, icon="EXPORT")


CLASSES = (BLITEFLUID_OT_export, BLITEFLUID_PT_export)


def menu_func_export(self, context):
    self.layout.operator(BLITEFLUID_OT_export.bl_idname, text="Babylon Lite Fluid JSON (.json)")


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.types.Scene.blitefluid_sdf_resolution = IntProperty(
        name="Collision SDF resolution",
        description="Grid points on the longest collision-SDF axis; independent from Mantaflow Resolution Divisions",
        default=64,
        min=8,
        max=192,
    )


def unregister():
    del bpy.types.Scene.blitefluid_sdf_resolution
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
