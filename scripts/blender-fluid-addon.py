"""Fluid scene exporter for native and add-on simulation setups."""

bl_info = {
    "name": "Babylon Lite Fluid JSON",
    "author": "Babylon Lite contributors",
    "version": (3, 17, 0),
    "blender": (4, 0, 0),
    "location": "Properties > Scene > Babylon Lite Fluid; File > Export",
    "description": "Export a native or add-on fluid setup to Babylon Lite JSON",
    "category": "Import-Export",
}

import base64
import glob
import json
import math
import os
import re
import struct
import tempfile
import zlib
from array import array

import bmesh
import bpy
from bpy.props import BoolProperty, IntProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

MAX_FLUID_EMITTERS = 16
MAX_FLUID_SINKS = 16
MAX_FLUID_POLYGON_POINTS = 256
MAX_SDF_VOXELS = 16 * 1024 * 1024
INITIAL_STATE_MAGIC = 0x49464C42
INITIAL_STATE_HEADER_BYTES = 32
MIN_DECIMATABLE_TRIANGLES = 256
FLIP_BASE_CELL_SIZE = 0.25
FLIP_MARKERS_PER_CELL = 8
BLITE_INFLOW_VOLUME_RATE = 20.0
PARTICLE_TARGET_PREVIEW_CACHE = {}
FLIP_FLUIDS_GENERATED_OBJECTS = {
    "fluid_surface",
    "whitewater_foam",
    "whitewater_bubble",
    "whitewater_spray",
    "whitewater_dust",
    "whitewater_foam_particle",
    "whitewater_bubble_particle",
    "whitewater_spray_particle",
    "whitewater_dust_particle",
}


def blite_vec(value):
    return [float(value.x), float(value.z), float(-value.y)]


def blender_vec(value):
    return Vector((value[0], -value[2], value[1]))


def object_id(obj):
    return re.sub(r"[^a-z0-9_-]+", "-", obj.name.lower()).strip("-") or "flow"


def linear_to_srgb(value):
    value = clamp(float(value), 0, 1)
    return value * 12.92 if value <= 0.0031308 else 1.055 * (value ** (1 / 2.4)) - 0.055


def color_hex(value):
    channels = [int(round(linear_to_srgb(value[index]) * 255)) for index in range(3)]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def material_base_color(material):
    if material is None:
        return None
    if material.node_tree:
        output = next((node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeOutputMaterial" and node.is_active_output), None)
        if output is not None:
            surface = output.inputs.get("Surface")
            if surface and surface.is_linked:
                shader = surface.links[0].from_node
                base_color = shader.inputs.get("Base Color")
                if base_color is not None and not base_color.is_linked:
                    return base_color.default_value
    return material.diffuse_color


def fluid_render_color(scene, domain):
    candidates = []
    if is_flip_fluids_domain(domain):
        surface = scene.objects.get("fluid_surface")
        if surface is not None and surface.type == "MESH":
            candidates.extend(surface.data.materials)
    if domain.type == "MESH":
        candidates.extend(domain.data.materials)
    for material in candidates:
        color = material_base_color(material)
        if color is not None:
            return color_hex(color)
    return "#16a3c3"


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


def is_emitter_source_object(obj):
    if not is_flow_object(obj):
        return False
    if flip_object_type(obj) == "TYPE_OUTFLOW":
        return False
    settings = flow_settings(obj)
    return settings is None or settings.flow_behavior != "OUTFLOW"


def flow_emitter_counts(scene):
    initial_count = 0
    inflow_count = 0
    for obj in scene.objects:
        if not is_emitter_source_object(obj):
            continue
        flip_type = flip_object_type(obj)
        if flip_type == "TYPE_FLUID":
            initial_count += 1
        elif flip_type == "TYPE_INFLOW":
            inflow_count += 1
        else:
            settings = flow_settings(obj)
            if settings is not None and settings.flow_behavior == "GEOMETRY":
                initial_count += 1
            elif settings is not None and settings.flow_behavior == "INFLOW":
                inflow_count += 1
    return initial_count, inflow_count


def has_animation_data(value):
    animation_data = getattr(value, "animation_data", None)
    return bool(
        animation_data
        and (
            animation_data.action is not None
            or any(not track.mute for track in animation_data.nla_tracks)
            or len(animation_data.drivers) > 0
        )
    )


def is_animated_mesh(obj):
    if obj.type != "MESH":
        return False
    current = obj
    while current is not None:
        if has_animation_data(current) or len(current.constraints) > 0:
            return True
        current = current.parent
    if has_animation_data(obj.data) or has_animation_data(getattr(obj.data, "shape_keys", None)):
        return True
    for modifier in obj.modifiers:
        if modifier.type == "ARMATURE" and modifier.object is not None and has_animation_data(modifier.object):
            return True
    return False


def animated_meshes(scene):
    return sorted((obj for obj in scene.objects if is_animated_mesh(obj)), key=lambda obj: obj.name.casefold())


def simulation_frame_range(scene, domain):
    if is_flip_fluids_domain(domain):
        frame_start, frame_end = domain.flip_fluid.domain.simulation.get_frame_range()
    else:
        settings = fluid_modifier(domain, "DOMAIN").domain_settings
        frame_start = settings.cache_frame_start
        frame_end = settings.cache_frame_end
    frame_start = int(frame_start)
    frame_end = int(frame_end)
    if frame_end < frame_start:
        raise ValueError(f"Fluid simulation frame range is invalid: {frame_start}..{frame_end}")
    return frame_start, frame_end


def local_bounds(obj):
    corners = [Vector(corner) for corner in obj.bound_box]
    minimum = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maximum = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return minimum, maximum


def authored_object_transform(obj):
    return obj.matrix_basis if obj.parent is None and not obj.constraints else obj.matrix_world


def world_bounds(obj, transform=None):
    # A liquid DOMAIN modifier evaluates to the current liquid surface, so obj.bound_box can
    # collapse around the flow. The authored domain cage is the original mesh data.
    transform = transform if transform is not None else authored_object_transform(obj)
    points = [transform @ vertex.co for vertex in obj.data.vertices] if obj.type == "MESH" else [transform @ Vector(corner) for corner in obj.bound_box]
    if not points:
        raise ValueError(f"{obj.name}: mesh has no vertices")
    minimum = Vector((min(v.x for v in points), min(v.y for v in points), min(v.z for v in points)))
    maximum = Vector((max(v.x for v in points), max(v.y for v in points), max(v.z for v in points)))
    return minimum, maximum


def authored_domain_bounds(domain):
    # Some saved Mantaflow domains expose a stale identity matrix_world while their
    # unparented authored transform remains valid in matrix_basis.
    return world_bounds(domain, authored_object_transform(domain))


def prism_shape_data(obj, minimum, maximum):
    if obj.type != "MESH":
        return None
    size = maximum - minimum
    tolerance = max(size) * 1e-5
    center = (minimum + maximum) * 0.5
    for axis in range(3):
        axis_min = minimum[axis]
        axis_max = maximum[axis]
        if axis_max - axis_min <= tolerance or any(min(abs(vertex.co[axis] - axis_min), abs(vertex.co[axis] - axis_max)) > tolerance for vertex in obj.data.vertices):
            continue
        caps = [
            polygon
            for polygon in obj.data.polygons
            if len(polygon.vertices) >= 3
            and all(abs(obj.data.vertices[index].co[axis] - axis_min) <= tolerance for index in polygon.vertices)
        ]
        if not caps:
            continue
        cap = max(caps, key=lambda polygon: len(polygon.vertices))
        vertices = [obj.data.vertices[index].co for index in cap.vertices]
        if axis == 0:
            points = [(float(vertex.z - center.z), float(vertex.y - center.y)) for vertex in vertices]
            basis = Matrix(((0, 1, 0), (1, 0, 0), (0, 0, -1))).to_quaternion()
            scale_order = (2, 0, 1)
        elif axis == 1:
            points = [(float(vertex.x - center.x), float(center.z - vertex.z)) for vertex in vertices]
            basis = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0))).to_quaternion()
            scale_order = (0, 1, 2)
        else:
            points = [(float(vertex.x - center.x), float(center.y - vertex.y)) for vertex in vertices]
            basis = Quaternion()
            scale_order = (0, 2, 1)
        return {"type": "polygonPrism", "points": points, "thickness": float(size[axis])}, center, basis, scale_order
    return None


def flow_shape_data(obj):
    minimum, maximum = local_bounds(obj)
    prism = prism_shape_data(obj, minimum, maximum)
    if prism is not None:
        return prism
    return box_shape_data(obj, minimum, maximum)


def box_shape_data(obj, minimum=None, maximum=None):
    if minimum is None or maximum is None:
        minimum, maximum = local_bounds(obj)
    size = maximum - minimum
    return {"type": "box", "size": [float(size.x), float(size.z), float(size.y)]}, (minimum + maximum) * 0.5, Quaternion(), (0, 2, 1)


def shape_for(obj):
    return flow_shape_data(obj)[0]


def ensure_flow_shape_thickness(shape, transform, cell_size):
    if shape["type"] == "box":
        shape["size"] = [
            max(size, cell_size / max(abs(transform["scale"][index]), 1e-6))
            for index, size in enumerate(shape["size"])
        ]
    elif shape["type"] == "polygonPrism":
        shape["thickness"] = max(shape["thickness"], cell_size / max(abs(transform["scale"][1]), 1e-6))


def transform_for(obj, grid_position, local_center=None, shape_basis=None, scale_order=(0, 2, 1)):
    authored_transform = authored_object_transform(obj)
    _, rotation, scale = authored_transform.decompose()
    position = authored_transform @ (local_center or Vector((0, 0, 0)))
    basis = Quaternion((1, 0, 0), -math.pi * 0.5)
    converted = basis @ rotation @ basis.conjugated()
    converted = converted @ (shape_basis or Quaternion())
    world_position = blite_vec(position)
    scales = (float(scale.x), float(scale.y), float(scale.z))
    return {
        "position": [world_position[index] - grid_position[index] for index in range(3)],
        "rotation": [float(converted.x), float(converted.y), float(converted.z), float(converted.w)],
        "scale": [scales[index] for index in scale_order],
    }


def flow_shape_transform_for(obj, grid_position):
    shape, center, basis, scale_order = flow_shape_data(obj)
    return shape, transform_for(obj, grid_position, center, basis, scale_order)


def box_shape_transform_for(obj, grid_position):
    shape, center, basis, scale_order = box_shape_data(obj)
    return shape, transform_for(obj, grid_position, center, basis, scale_order)


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


def fluid_grid_cells(grid_size, resolution):
    cell_size = max(grid_size) / float(resolution)
    cells = []
    for extent in grid_size:
        exact_cells = extent / cell_size
        nearest_integer = round(exact_cells)
        tolerance = 2.220446049250313e-16 * 16 * max(1, abs(exact_cells))
        cells.append(max(4, nearest_integer if abs(exact_cells - nearest_integer) <= tolerance else math.ceil(exact_cells)))
    return cell_size, cells


def flow_shape_volume(shape):
    if shape["type"] == "box":
        return math.prod(shape["size"])
    if shape["type"] == "polygonPrism":
        points = shape["points"]
        twice_area = sum(
            points[index][0] * points[(index + 1) % len(points)][1] - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        return abs(twice_area) * 0.5 * shape["thickness"]
    return 0


def initial_emitter_volume(emitters):
    return sum(
        flow_shape_volume(emitter["shape"]) * math.prod(abs(value) for value in emitter["transform"]["scale"])
        for emitter in emitters
        if emitter["enabled"] and emitter["behavior"] == "initial"
    )


def quaternion_rotate(rotation, value):
    x, y, z, w = rotation
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length <= 1e-8:
        x, y, z, w = 0, 0, 0, 1
    else:
        x, y, z, w = x / length, y / length, z / length, w / length
    tx = 2 * (y * value[2] - z * value[1])
    ty = 2 * (z * value[0] - x * value[2])
    tz = 2 * (x * value[1] - y * value[0])
    return (
        value[0] + w * tx + y * tz - z * ty,
        value[1] + w * ty + z * tx - x * tz,
        value[2] + w * tz + x * ty - y * tx,
    )


def point_in_polygon(point, points):
    x, y = point
    inside = False
    previous = points[-1]
    for current in points:
        ax, ay = previous
        bx, by = current
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) <= 1e-7 and min(ax, bx) - 1e-7 <= x <= max(ax, bx) + 1e-7 and min(ay, by) - 1e-7 <= y <= max(ay, by) + 1e-7:
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
        previous = current
    return inside


def initial_shape_bounds(shape):
    if shape["type"] == "box":
        half = [abs(value) * 0.5 for value in shape["size"]]
        return [-half[0], -half[1], -half[2]], half
    if shape["type"] == "polygonPrism":
        points = shape["points"]
        half = abs(shape["thickness"]) * 0.5
        return [min(point[0] for point in points), -half, min(point[1] for point in points)], [max(point[0] for point in points), half, max(point[1] for point in points)]
    raise ValueError(f'Unsupported initial target shape: {shape["type"]}')


def transform_initial_point(transform, point):
    scaled = [point[index] * transform["scale"][index] for index in range(3)]
    rotated = quaternion_rotate(transform["rotation"], scaled)
    return [rotated[index] + transform["position"][index] for index in range(3)]


def distribute_particle_count(total_count, weights, emitters):
    total_weight = sum(weights)
    if total_count <= 0 or total_weight <= 0:
        return [0 for _ in weights]
    allocations = []
    for index, weight in enumerate(weights):
        exact = total_count * weight / total_weight
        count = math.floor(exact)
        allocations.append({"index": index, "count": count, "remainder": exact - count})
    left = total_count - sum(allocation["count"] for allocation in allocations)
    ranked = sorted(allocations, key=lambda allocation: (-allocation["remainder"], emitters[allocation["index"]]["id"], allocation["index"]))
    for index in range(left):
        ranked[index]["count"] += 1
    return [allocation["count"] for allocation in allocations]


def count_clipped_emitter_lattice(emitter, count, world_volume, grid_size):
    if count <= 0 or world_volume <= 0:
        return 0
    scale = [abs(value) for value in emitter["transform"]["scale"]]
    if any(value == 0 for value in scale):
        return count
    minimum, maximum = initial_shape_bounds(emitter["shape"])
    extents = [maximum[index] - minimum[index] for index in range(3)]
    center = [(minimum[index] + maximum[index]) * 0.5 for index in range(3)]
    world_spacing = (world_volume / count) ** (1 / 3)
    accepted_count = 0
    grid_min = [-value * 0.5 for value in grid_size]
    grid_max = [value * 0.5 for value in grid_size]
    for _ in range(20):
        spacing = [world_spacing / scale[index] for index in range(3)]
        dimensions = [max(1, int(math.ceil(extents[index] / spacing[index] - 1e-8))) for index in range(3)]
        start = [center[index] - (dimensions[index] - 1) * spacing[index] * 0.5 for index in range(3)]
        world_start = transform_initial_point(emitter["transform"], start)
        axis_x = quaternion_rotate(emitter["transform"]["rotation"], [emitter["transform"]["scale"][0] * spacing[0], 0, 0])
        axis_y = quaternion_rotate(emitter["transform"]["rotation"], [0, emitter["transform"]["scale"][1] * spacing[1], 0])
        axis_z = quaternion_rotate(emitter["transform"]["rotation"], [0, 0, emitter["transform"]["scale"][2] * spacing[2]])
        xz_points = []
        for z in range(dimensions[2]):
            local_z = start[2] + z * spacing[2]
            for x in range(dimensions[0]):
                local_x = start[0] + x * spacing[0]
                if emitter["shape"]["type"] == "polygonPrism" and not point_in_polygon((local_x, local_z), emitter["shape"]["points"]):
                    continue
                xz_points.append(
                    (
                        world_start[0] + x * axis_x[0] + z * axis_z[0],
                        world_start[1] + x * axis_x[1] + z * axis_z[1],
                        world_start[2] + x * axis_x[2] + z * axis_z[2],
                    )
                )
        accepted_count = 0
        unclipped_count = len(xz_points) * dimensions[1]
        for y in range(dimensions[1]):
            yx = y * axis_y[0]
            yy = y * axis_y[1]
            yz = y * axis_y[2]
            for base_x, base_y, base_z in xz_points:
                world_x = base_x + yx
                world_y = base_y + yy
                world_z = base_z + yz
                if (
                    grid_min[0] <= world_x <= grid_max[0]
                    and grid_min[1] <= world_y <= grid_max[1]
                    and grid_min[2] <= world_z <= grid_max[2]
                ):
                    accepted_count += 1
        if unclipped_count >= count:
            break
        world_spacing *= 0.96
    return min(count, accepted_count)


def clipped_initial_particle_count(emitters, grid_size, cell_size, markers_per_cell):
    initial = [emitter for emitter in emitters if emitter["enabled"] and emitter["behavior"] == "initial"]
    volumes = [flow_shape_volume(emitter["shape"]) * math.prod(abs(value) for value in emitter["transform"]["scale"]) for emitter in initial]
    total_volume = sum(volumes)
    if total_volume <= 0:
        return 0
    particle_volume = cell_size**3 / markers_per_cell
    density_count = int(math.ceil(total_volume / particle_volume))
    allocations = distribute_particle_count(density_count, volumes, initial)
    return sum(
        count_clipped_emitter_lattice(emitter, allocations[index], volumes[index], grid_size)
        if emitter["sampling"] == "volume"
        else allocations[index]
        for index, emitter in enumerate(initial)
    )


def target_flip_resolution(grid_size, emitters, markers_per_cell, target_particles):
    initial_volume = initial_emitter_volume(emitters)
    if initial_volume <= 0:
        raise ValueError("Target particle count requires at least one enabled volume-sampled initial fluid object")
    approximate_cell = (initial_volume * markers_per_cell / target_particles) ** (1 / 3)
    candidate = max(16, int(round(max(grid_size) / approximate_cell)))
    counts = {}

    def clipped_count(resolution):
        if resolution not in counts:
            cell_size, _ = fluid_grid_cells(grid_size, resolution)
            counts[resolution] = clipped_initial_particle_count(emitters, grid_size, cell_size, markers_per_cell)
        return counts[resolution]

    for _ in range(5):
        count = clipped_count(candidate)
        if count <= 0:
            break
        next_candidate = max(16, int(round(candidate * (target_particles / count) ** (1 / 3))))
        if next_candidate == candidate:
            break
        candidate = next_candidate
    nearby = range(max(16, candidate - 4), candidate + 5)
    return min(nearby, key=lambda resolution: (abs(clipped_count(resolution) - target_particles), resolution))


def derived_domain_values(scene, domain, grid_size, resolution_override=None):
    if is_flip_fluids_domain(domain):
        domain_props = domain.flip_fluid.domain
        simulation = domain_props.simulation
        advanced = domain_props.advanced
        world = domain_props.world
        raw_resolution = max(1, int(simulation.resolution))
        resolution = max(16, int(resolution_override if resolution_override is not None else raw_resolution))
        cell_size, cells = fluid_grid_cells(grid_size, resolution)
        markers_per_cell = FLIP_MARKERS_PER_CELL
        raw_particle_size = cell_size / FLIP_BASE_CELL_SIZE
        particle_size = clamp(raw_particle_size, 0.1, 8)
        raw_particle_count = cells[0] * cells[1] * cells[2] * markers_per_cell
        particle_count = max(1, int(raw_particle_count))
        gravity_vector = Vector(world.gravity)
        if getattr(world, "gravity_type", "GRAVITY_TYPE_SCENE") == "GRAVITY_TYPE_SCENE":
            gravity_vector = Vector(scene.gravity) if scene.use_gravity else Vector((0, 0, 0))
        time_steps = advanced.min_max_time_steps_per_frame
        min_substeps = int(clamp(int(time_steps.value_min), 1, 16))
        max_substeps = int(clamp(max(min_substeps, int(time_steps.value_max)), 1, 32))
        simulation_fps = max(float(simulation.get_frame_rate()), 1e-6)
        pressure_iterations = int(clamp(int(advanced.pressure_solver_max_iterations), 1, 100))
        pressure_tolerance = clamp(float(getattr(advanced, "pressure_solver_error_tolerance", 1e-3)), 0, 0.1)
        surface = getattr(domain_props, "surface", None)
        particle_sheeting = bool(getattr(surface, "enable_sheet_seeding", False))
        sheeting_strength = clamp(float(getattr(surface, "sheet_fill_rate", 0.5)), 0.05, 1)
        flip_ratio = clamp(1.0 - float(advanced.PICFLIP_ratio), 0, 1)
        kinematic_viscosity = 0
        surface_tension = 0
        fractional_solids = 1
        reseed_min = max(1, markers_per_cell // 2)
        reseed_max = int(math.ceil(markers_per_cell * 1.5))
        capacity_hint = 0
        cfl_number = clamp(float(getattr(advanced, "CFL_condition_number", 2)), 0, 10)
    else:
        modifier = fluid_modifier(domain, "DOMAIN")
        settings = modifier.domain_settings if modifier is not None else None
        if settings is None:
            raise ValueError("Liquid domain settings are unavailable")
        raw_resolution = max(1, int(settings.resolution_max))
        resolution = max(16, int(resolution_override if resolution_override is not None else raw_resolution))
        cell_size, cells = fluid_grid_cells(grid_size, resolution)
        marker_axis = max(1, int(settings.particle_number))
        markers_per_cell = int(clamp(marker_axis**3, 1, 64))
        raw_particle_size = cell_size / FLIP_BASE_CELL_SIZE
        particle_size = clamp(raw_particle_size, 0.1, 8)
        raw_particle_count = cells[0] * cells[1] * cells[2] * markers_per_cell
        capacity_hint = max(0, int(settings.sys_particle_maximum))
        particle_count = capacity_hint or max(1, int(raw_particle_count))
        gravity_vector = Vector(settings.gravity)
        min_substeps = int(clamp(int(settings.timesteps_min), 1, 16))
        max_substeps = int(clamp(max(min_substeps, int(settings.timesteps_max)), 1, 32))
        simulation_fps = max(float(scene.render.fps) / max(float(scene.render.fps_base), 1e-12), 1e-6)
        pressure_iterations = 40
        pressure_tolerance = 1e-3
        particle_sheeting = False
        sheeting_strength = 0.5
        flip_ratio = clamp(float(settings.flip_ratio), 0, 1)
        physical_viscosity = float(settings.viscosity_base) * (10 ** -int(settings.viscosity_exponent)) if settings.use_diffusion else 0
        kinematic_viscosity = clamp(max(physical_viscosity, float(settings.viscosity_value) if settings.use_viscosity else 0), 0, 5)
        surface_tension = clamp(float(settings.surface_tension), 0, 5) if settings.use_diffusion else 0
        fractional_solids = 1 if settings.use_fractions else 0
        reseed_min = int(clamp(int(settings.particle_min), 1, 64))
        reseed_max = int(clamp(max(reseed_min, int(settings.particle_max)), 1, 96))
        cfl_number = clamp(float(settings.cfl_condition), 0, 10)

    return {
        "method": "FLIP",
        "particle_count": particle_count,
        "raw_particle_count": raw_particle_count,
        "capacity_hint": capacity_hint,
        "particle_size": particle_size,
        "raw_particle_size": raw_particle_size,
        "cell_size": cell_size,
        "cells": cells,
        "resolution": resolution,
        "raw_resolution": raw_resolution,
        "markers_per_cell": markers_per_cell,
        "physics": {
            "gravity": clamp(float(-gravity_vector.z), 0, 200),
            "flipRatio": flip_ratio,
            "kinematicViscosity": kinematic_viscosity,
            "surfaceTension": surface_tension,
            "minSubsteps": min_substeps,
            "maxSubsteps": max_substeps,
            "cflNumber": cfl_number,
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
            "fractionalSolids": fractional_solids,
            "movingSolidBoundaries": 1,
            "reseedParticles": 1,
            "reseedMinParticles": reseed_min,
            "reseedTargetParticles": markers_per_cell,
            "reseedMaxParticles": reseed_max,
            "reseedInterval": 5,
            "particleSheeting": 1 if particle_sheeting else 0,
            "sheetingStrength": sheeting_strength,
            "sheetingInterval": 5,
            "polygonSurface": 0,
            "viscosityIterations": 12,
            "maxSubDtMs": 1000 / (simulation_fps * min_substeps),
        },
    }


def extract_flows(scene, grid_position, derived=None):
    emitters = []
    pending_sinks = []
    object_ids = {}
    domain = find_domain(scene)
    domain_bounds = authored_domain_bounds(domain)
    domain_size = domain_bounds[1] - domain_bounds[0]
    grid_size = [float(domain_size.x), float(domain_size.z), float(domain_size.y)]
    derived = derived or derived_domain_values(scene, domain, grid_size)
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
        shape, transform = flow_shape_transform_for(obj, grid_position)
        ensure_flow_shape_thickness(shape, transform, derived["cell_size"])
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
            "sourcePresentation": bool(not obj.hide_render and obj.visible_get()),
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
        shape, transform = box_shape_transform_for(obj, grid_position)
        if source_kind == "MANTA":
            surface_margin = max(0.0, float(settings.surface_distance)) * derived["cell_size"]
            shape["size"] = [
                size + 2 * surface_margin / max(abs(transform["scale"][index]), 1e-6)
                for index, size in enumerate(shape["size"])
            ]
            enabled = bool(getattr(settings, "use_inflow", True))
        else:
            enabled = bool(settings.is_enabled and settings.remove_fluid)
        ensure_flow_shape_thickness(shape, transform, derived["cell_size"])
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


def targeted_flow_setup(scene, domain, grid_position, grid_size, derived=None):
    derived = derived or derived_domain_values(scene, domain, grid_size)
    emitters, sinks = extract_flows(scene, grid_position, derived)
    target_initial_particles = int(scene.blitefluid_target_initial_particles)
    if target_initial_particles > 0:
        previous_resolution = None
        for _ in range(3):
            target_resolution = target_flip_resolution(grid_size, emitters, derived["markers_per_cell"], target_initial_particles)
            if target_resolution == previous_resolution:
                break
            previous_resolution = target_resolution
            derived = derived_domain_values(scene, domain, grid_size, target_resolution)
            emitters, sinks = extract_flows(scene, grid_position, derived)

    initial_markers = clipped_initial_particle_count(emitters, grid_size, derived["cell_size"], derived["markers_per_cell"])
    has_inflow = any(emitter["behavior"] == "inflow" for emitter in emitters)
    inflow_particles = int(scene.blitefluid_target_inflow_particles) if has_inflow else 0
    if has_inflow or target_initial_particles > 0:
        particle_count = max(1, initial_markers + inflow_particles)
    else:
        particle_count = max(initial_markers, derived["capacity_hint"] or 1)
    return derived, emitters, sinks, initial_markers, inflow_particles, particle_count


def particle_target_preview_key(scene, domain):
    objects = []
    for obj in scene.objects:
        if not is_emitter_source_object(obj):
            continue
        settings = flow_settings(obj)
        objects.append(
            (
                obj.name,
                flip_object_type(obj),
                getattr(settings, "flow_behavior", ""),
                bool(getattr(settings, "use_plane_init", False)),
                tuple(round(value, 8) for row in authored_object_transform(obj) for value in row),
                tuple(round(value, 8) for vertex in obj.data.vertices for value in vertex.co),
            )
        )
    if is_flip_fluids_domain(domain):
        source_resolution = int(domain.flip_fluid.domain.simulation.resolution)
        marker_density = FLIP_MARKERS_PER_CELL
    else:
        settings = fluid_modifier(domain, "DOMAIN").domain_settings
        source_resolution = int(settings.resolution_max)
        marker_density = int(settings.particle_number)
    return (
        scene.as_pointer(),
        int(scene.blitefluid_target_initial_particles),
        int(scene.blitefluid_target_inflow_particles),
        source_resolution,
        marker_density,
        tuple(round(value, 8) for vertex in domain.data.vertices for value in vertex.co),
        tuple(round(value, 8) for row in authored_object_transform(domain) for value in row),
        tuple(objects),
    )


def particle_target_preview(scene):
    domain = find_domain(scene)
    key = particle_target_preview_key(scene, domain)
    cached = PARTICLE_TARGET_PREVIEW_CACHE.get(key)
    if cached is not None:
        return cached
    minimum, maximum = authored_domain_bounds(domain)
    center = (minimum + maximum) * 0.5
    size = maximum - minimum
    grid_position = blite_vec(center)
    grid_size = [float(size.x), float(size.z), float(size.y)]
    derived, _, _, initial_particles, inflow_particles, total_particles = targeted_flow_setup(scene, domain, grid_position, grid_size)
    preview = derived["resolution"], initial_particles, inflow_particles, total_particles
    if len(PARTICLE_TARGET_PREVIEW_CACHE) >= 8:
        PARTICLE_TARGET_PREVIEW_CACHE.pop(next(iter(PARTICLE_TARGET_PREVIEW_CACHE)))
    PARTICLE_TARGET_PREVIEW_CACHE[key] = preview
    return preview


def triangle_indices(triangle, base, reverse_winding):
    first, second, third = (base + index for index in triangle.vertices)
    return (first, third, second) if reverse_winding else (first, second, third)


def build_collision_bvhs(objects):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    records = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            transform = authored_object_transform(obj)
            vertices = [transform @ vertex.co for vertex in mesh.vertices]
            if not vertices or not mesh.loop_triangles:
                continue
            reverse_winding = transform.to_3x3().determinant() < 0
            triangles = [triangle_indices(triangle, 0, reverse_winding) for triangle in mesh.loop_triangles]
            minimum = Vector((min(value.x for value in vertices), min(value.y for value in vertices), min(value.z for value in vertices)))
            maximum = Vector((max(value.x for value in vertices), max(value.y for value in vertices), max(value.z for value in vertices)))
            records.append((BVHTree.FromPolygons(vertices, triangles, all_triangles=True), minimum, maximum))
        finally:
            evaluated.to_mesh_clear()
    return records


def build_local_collision_bvh(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = [vertex.co.copy() for vertex in mesh.vertices]
        if not vertices or not mesh.loop_triangles:
            raise ValueError(f"{obj.name}: animated collision mesh has no triangles")
        triangles = [triangle_indices(triangle, 0, False) for triangle in mesh.loop_triangles]
        minimum = Vector((min(value.x for value in vertices), min(value.y for value in vertices), min(value.z for value in vertices)))
        maximum = Vector((max(value.x for value in vertices), max(value.y for value in vertices), max(value.z for value in vertices)))
        return BVHTree.FromPolygons(vertices, triangles, all_triangles=True), minimum, maximum
    finally:
        evaluated.to_mesh_clear()


def point_aabb_distance_squared(point, minimum, maximum):
    return sum(max(minimum[index] - point[index], 0, point[index] - maximum[index]) ** 2 for index in range(3))


def point_inside_bvh(point, bvh, minimum, maximum, epsilon):
    if any(point[index] < minimum[index] - epsilon or point[index] > maximum[index] + epsilon for index in range(3)):
        return False
    direction = Vector((0.812381, 0.334219, 0.477913)).normalized()
    origin = point
    intersections = 0
    for _ in range(256):
        location, _, _, _ = bvh.ray_cast(origin, direction)
        if location is None:
            break
        intersections += 1
        origin = location + direction * epsilon
    return intersections % 2 == 1


def collision_signed_distance(point, records, empty_distance, epsilon):
    nearest_distance = empty_distance
    inside = False
    nearest_squared = nearest_distance * nearest_distance
    for bvh, minimum, maximum in records:
        bounds_distance_squared = point_aabb_distance_squared(point, minimum, maximum)
        within_bounds = bounds_distance_squared <= epsilon * epsilon
        if bounds_distance_squared <= nearest_squared or within_bounds:
            _, _, _, unsigned = bvh.find_nearest(point)
            if unsigned is not None and unsigned < nearest_distance:
                nearest_distance = unsigned
                nearest_squared = unsigned * unsigned
        if within_bounds and point_inside_bvh(point, bvh, minimum, maximum, epsilon):
            inside = True
    return -nearest_distance if inside else nearest_distance


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


def local_collision_metrics(obj, resolution):
    _, minimum, maximum = build_local_collision_bvh(obj)
    size = maximum - minimum
    grid_size = [float(size.x), float(size.z), float(size.y)]
    effective_resolution = max(2, int(resolution))
    padding_cells = min(2, max(0, (effective_resolution - 2) // 2))
    denominator = effective_resolution - 1 - 2 * padding_cells
    if denominator <= 0:
        padding_cells = 0
        denominator = effective_resolution - 1
    cell_size = max(grid_size) / max(1, denominator)
    dims = [max(2, int(math.ceil(axis / cell_size)) + 1 + 2 * padding_cells) for axis in grid_size]
    return cell_size, dims, math.prod(dims), padding_cells, minimum, maximum


def evaluated_collision_state(obj, frame):
    scene = bpy.context.scene
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    matrix = tuple(float(evaluated.matrix_world[row][column]) for row in range(4) for column in range(4))
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        return (
            len(mesh.loop_triangles),
            [(float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)) for vertex in mesh.vertices],
            matrix,
        )
    finally:
        evaluated.to_mesh_clear()


def classify_collision_animation(obj, scene, frame_start, frame_end):
    if frame_end <= frame_start:
        return "static"
    frames = sorted({round(frame_start + (frame_end - frame_start) * index / 8) for index in range(9)})
    previous_frame = scene.frame_current
    try:
        reference_triangles, reference_vertices, reference_matrix = evaluated_collision_state(obj, frames[0])
        transform_changed = False
        for frame in frames[1:]:
            triangles, vertices, matrix = evaluated_collision_state(obj, frame)
            if triangles != reference_triangles or len(vertices) != len(reference_vertices):
                return "topology"
            if any(
                abs(vertex[axis] - reference_vertices[index][axis]) > 1e-5
                for index, vertex in enumerate(vertices)
                for axis in range(3)
            ):
                return "deforming"
            transform_changed = transform_changed or any(abs(value - reference_matrix[index]) > 1e-6 for index, value in enumerate(matrix))
        return "rigid" if transform_changed else "static"
    finally:
        scene.frame_set(previous_frame)


def bake_collision(objects, grid_position, grid_size, resolution, window_manager):
    cell_size, dims, voxel_count = grid_metrics(grid_size, resolution)
    if voxel_count > MAX_SDF_VOXELS:
        raise ValueError(f"Collision grid has {voxel_count:,} voxels; lower SDF resolution")
    origin = [grid_position[index] - grid_size[index] * 0.5 for index in range(3)]
    empty_distance = max(grid_size)
    epsilon = max(1e-6, cell_size * 1e-5)
    collision_bvhs = build_collision_bvhs(objects)
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
                    point = blender_vec(point_blite)
                    distance = collision_signed_distance(point, collision_bvhs, empty_distance, epsilon)
                    struct.pack_into("<f", output, 64 + index * 4, float(distance))
                    index += 1
            window_manager.progress_update(z + 1)
    finally:
        window_manager.progress_end()
    return bytes(output)


def bake_local_collision(obj, resolution, window_manager):
    cell_size, dims, voxel_count, padding_cells, minimum, maximum = local_collision_metrics(obj, resolution)
    if voxel_count > MAX_SDF_VOXELS:
        raise ValueError(f"{obj.name}: animated collision grid has {voxel_count:,} voxels; lower its SDF resolution")
    bvh, _, _ = build_local_collision_bvh(obj)
    origin = [
        float(minimum.x) - padding_cells * cell_size,
        float(minimum.z) - padding_cells * cell_size,
        float(-maximum.y) - padding_cells * cell_size,
    ]
    empty_distance = max(float(maximum.x - minimum.x), float(maximum.y - minimum.y), float(maximum.z - minimum.z))
    epsilon = max(1e-6, cell_size * 1e-5)
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
    record = [(bvh, minimum, maximum)]
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
                    distance = collision_signed_distance(blender_vec(point_blite), record, empty_distance, epsilon)
                    struct.pack_into("<f", output, 64 + index * 4, float(distance))
                    index += 1
            window_manager.progress_update(z + 1)
    finally:
        window_manager.progress_end()
    return bytes(output), dims, voxel_count


def scene_objects(scene, domain, frame_start, frame_end):
    animated = animated_meshes(scene)
    animated_set = set(animated)
    collision_animation = {
        obj: classify_collision_animation(obj, scene, frame_start, frame_end)
        for obj in animated
        if is_collision_object(obj)
    }
    animated_collisions = [
        obj
        for obj, kind in collision_animation.items()
        if int(obj.blitefluid_animated_sdf_resolution) > 0 and kind == "rigid"
    ]
    unsupported_animated_collisions = [
        (obj, kind)
        for obj, kind in collision_animation.items()
        if int(obj.blitefluid_animated_sdf_resolution) > 0 and kind in {"deforming", "topology"}
    ]
    emitter_sources = [obj for obj in scene.objects if obj != domain and obj.type == "MESH" and is_emitter_source_object(obj)]
    presentation_objects = [
        obj
        for obj in scene.objects
        if obj != domain
        and (obj.type == "MESH" or (obj.type == "LIGHT" and obj.data.type in {"POINT", "SUN", "SPOT"}))
        and not obj.hide_render
        and obj.visible_get()
        and not (is_flip_fluids_domain(domain) and obj.name in FLIP_FLUIDS_GENERATED_OBJECTS)
    ]
    for obj in [*animated_collisions, *emitter_sources]:
        if obj not in presentation_objects:
            presentation_objects.append(obj)
        parent = obj.parent
        while parent is not None:
            if parent not in presentation_objects:
                presentation_objects.append(parent)
            parent = parent.parent
    visual_objects = [obj for obj in presentation_objects if obj.type == "MESH"]
    collision_objects = [
        obj
        for obj in scene.objects
        if obj != domain
        and obj.type == "MESH"
        and is_collision_object(obj)
        and (
            obj not in animated_set
            or (
                int(obj.blitefluid_animated_sdf_resolution) > 0
                and collision_animation.get(obj) == "static"
            )
        )
    ]
    return presentation_objects, visual_objects, collision_objects, animated, animated_collisions, unsupported_animated_collisions


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
            frame_start, frame_end = simulation_frame_range(scene, domain)
            summary.update(
                domain=domain,
                domain_bounds=domain_bounds,
                grid_size=grid_size,
                cell_size=cell_size,
                dims=dims,
                voxel_count=voxel_count,
                sdf_resolution=sdf_resolution,
                frame_start=frame_start,
                frame_end=frame_end,
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
    if not is_flip_fluids_domain(summary["domain"]):
        domain_settings = fluid_modifier(summary["domain"], "DOMAIN").domain_settings
        if abs(domain_settings.gravity.x) > 1e-5 or abs(domain_settings.gravity.y) > 1e-5 or domain_settings.gravity.z > 1e-5:
            warnings.append("Babylon Lite currently uses only the downward source-scene Z gravity component")
    unsupported_lights = [
        obj.name
        for obj in scene.objects
        if obj.type == "LIGHT" and obj.data.type not in {"POINT", "SUN", "SPOT"} and not obj.hide_render and obj.visible_get()
    ]
    if scene.blitefluid_export_lights and unsupported_lights:
        warnings.append(f"glTF cannot export non-punctual lights; skipped: {', '.join(unsupported_lights)}")

    presentation_objects, visual_objects, collision_objects, animated, animated_collisions, unsupported_animated_collisions = scene_objects(
        scene,
        summary["domain"],
        summary["frame_start"],
        summary["frame_end"],
    )
    summary.update(
        presentation_objects=presentation_objects,
        visual_objects=visual_objects,
        collision_objects=collision_objects,
        animated_meshes=animated,
        animated_collisions=animated_collisions,
        emitter_count=emitter_count,
        sink_count=sink_count,
    )
    if not visual_objects:
        errors.append("No visible presentation meshes are available for scene.glb")
    if not collision_objects and not animated_collisions:
        warnings.append("No liquid effectors or Babylon Lite collider meshes were found")
    if len(animated_collisions) > 16:
        errors.append("Babylon Lite supports at most 16 animated collision meshes")
    for obj, kind in unsupported_animated_collisions:
        if kind == "topology":
            errors.append(f"{obj.name}: deforming or topology-changing animated collisions are not supported")
        else:
            errors.append(f"{obj.name}: deforming animated collisions are not supported; use rigid object or parent transforms")
    animated_voxels = 0
    for obj in animated_collisions:
        try:
            _, dims, voxel_count, _, _, _ = local_collision_metrics(obj, int(obj.blitefluid_animated_sdf_resolution))
            animated_voxels += voxel_count
            if voxel_count > MAX_SDF_VOXELS:
                errors.append(f"{obj.name}: animated collision grid has {voxel_count:,} voxels; lower its SDF resolution")
            elif min(dims) < 2:
                errors.append(f"{obj.name}: animated collision grid is degenerate")
        except ValueError as error:
            errors.append(str(error))
    if summary.get("voxel_count", 0) + animated_voxels > MAX_SDF_VOXELS:
        errors.append("Combined static and animated collision grids exceed the 16M-voxel limit")
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
        simulation_fps = float(dprops.simulation.get_frame_rate())
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
        simulation_fps = float(scene.render.fps) / max(float(scene.render.fps_base), 1e-12)
        application = "Native fluid"
    frame_start, frame_end = simulation_frame_range(scene, domain)
    return {
        "application": application,
        "version": bpy.app.version_string,
        "settings": {
            "timeline": {
                "frameStart": frame_start,
                "frameEnd": frame_end,
                "fps": scene.render.fps,
                "fpsBase": scene.render.fps_base,
                "simulationFps": simulation_fps,
            },
            "export": {
                "separateFiles": scene.blitefluid_separate_files,
                "exportLights": scene.blitefluid_export_lights,
                "decimateMeshes": scene.blitefluid_decimate_meshes,
                "targetTriangles": scene.blitefluid_target_triangles,
                "initialState": scene.blitefluid_export_initial_state,
            },
            "collisionSdfResolution": scene.blitefluid_sdf_resolution,
            "targetInitialParticles": scene.blitefluid_target_initial_particles,
            "targetInflowParticles": scene.blitefluid_target_inflow_particles,
            "animatedCollisionSdfResolutions": {
                obj.name: int(obj.blitefluid_animated_sdf_resolution)
                for obj in animated_meshes(scene)
            },
            "domain": domain_settings,
            "flows": flows,
            "effectors": effectors,
        },
    }


def camera_preset(scene, domain):
    camera = scene.camera
    if camera is None or camera.type != "CAMERA" or camera.data.type != "PERSP":
        return None
    evaluated = camera.evaluated_get(bpy.context.evaluated_depsgraph_get())
    position = evaluated.matrix_world.translation.copy()
    forward = evaluated.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    if forward.length_squared <= 1e-12:
        return None
    forward.normalize()
    minimum, maximum = authored_domain_bounds(domain)
    domain_center = (minimum + maximum) * 0.5
    radius = max((domain_center - position).dot(forward), (maximum - minimum).length, 1.0)
    target = position + forward * radius
    runtime_position = blite_vec(position)
    runtime_target = blite_vec(target)
    offset = [runtime_position[axis] - runtime_target[axis] for axis in range(3)]
    runtime_radius = math.sqrt(sum(value * value for value in offset))
    frame = camera.data.view_frame(scene=scene)
    vertical_angles = [math.atan2(corner.y, -corner.z) for corner in frame]
    return {
        "alpha": math.atan2(offset[2], offset[0]),
        "beta": math.acos(clamp(offset[1] / runtime_radius, -1, 1)),
        "radius": runtime_radius,
        "target": runtime_target,
        "fov": clamp(max(vertical_angles) - min(vertical_angles), 1e-4, math.pi - 1e-4),
        "mirrorX": True,
    }


def default_preset(scene, grid_position, grid_size, emitters, sinks, derived, particle_count):
    domain = find_domain(scene)
    if is_flip_fluids_domain(domain):
        settings = domain.flip_fluid.domain
        foam_enabled = bool(getattr(settings.whitewater, "enable_whitewater_simulation", False))
        bubbles_enabled = foam_enabled
        foam_layer_depth = clamp(float(getattr(settings.whitewater, "foam_layer_depth", 0)), 0, 4)
        simulation_time_scale = clamp(float(settings.simulation.time_scale), 0.01, 100)
    else:
        settings = fluid_modifier(domain, "DOMAIN").domain_settings
        foam_enabled = bool(settings.use_spray_particles or settings.use_bubble_particles or settings.use_foam_particles)
        bubbles_enabled = bool(settings.use_bubble_particles)
        foam_layer_depth = 0
        simulation_time_scale = clamp(float(settings.time_scale), 0.01, 100)
    camera = camera_preset(scene, domain)
    preset = {
        "formatVersion": 16,
        "simulationSemantics": {
            "version": 1,
            "profile": "normalized-v1",
            "pbfPhysics": "scale-adjusted",
        },
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
        **({"camera": camera} if camera is not None else {}),
        "render": {
            "renderAsSpheres": False,
            "waterColor": fluid_render_color(scene, domain),
            "absorption": 1,
            "particleSize": 0.6,
            "refractionStrength": 0.1,
            "specularPower": 250,
            "reflectionExposure": 2,
            "reflectionContrast": 0.6,
            "waterReflectivity": 0.02,
            "surfaceDepthBlur": 18,
            "depthBlurEdgeThreshold": 0.05,
            "surfaceThicknessBlur": 6,
            "halfRendering": True,
            "thicknessDownscale": 8,
            "surfaceFilter": "narrowRange",
            "narrowRangeDelta": 10,
            "narrowRangeMu": 1,
            "anisotropicSurface": False,
            "anisoRadiusDamping": 0.2,
        },
        "foam": {
            "enableFoam": foam_enabled,
            "activeParticles": False,
            "trappedAirRate": 51,
            "waveCrestRate": 48,
            "foamLayerDepth": foam_layer_depth,
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


def evaluated_triangle_count(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        return len(mesh.loop_triangles)
    finally:
        evaluated.to_mesh_clear()


def add_export_decimation(objects, target_triangles, protected_objects=()):
    mesh_objects = [obj for obj in objects if obj.type == "MESH"]
    protected = set(protected_objects)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    triangle_counts = [(obj, evaluated_triangle_count(obj, depsgraph)) for obj in mesh_objects]
    total_triangles = sum(count for _, count in triangle_counts)
    if total_triangles <= target_triangles:
        return []
    fixed_triangles = sum(count for obj, count in triangle_counts if count <= MIN_DECIMATABLE_TRIANGLES or obj in protected)
    decimatable_triangles = total_triangles - fixed_triangles
    ratio = max(0, target_triangles - fixed_triangles) / max(1, decimatable_triangles)
    modifiers = []
    try:
        for obj, count in triangle_counts:
            if count <= MIN_DECIMATABLE_TRIANGLES or obj in protected:
                continue
            modifier = obj.modifiers.new("Babylon Lite export decimation", "DECIMATE")
            modifier.decimate_type = "COLLAPSE"
            modifier.ratio = ratio
            modifier.use_collapse_triangulate = True
            modifiers.append((obj, modifier))
    except Exception:
        for obj, modifier in modifiers:
            obj.modifiers.remove(modifier)
        raise
    bpy.context.view_layer.update()
    return modifiers


def glb_json(path):
    with open(path, "rb") as stream:
        data = stream.read()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("Blender produced an invalid GLB")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A or 20 + json_length > len(data):
        raise ValueError("Blender produced a GLB without a valid JSON chunk")
    return json.loads(data[20 : 20 + json_length].decode("utf-8").rstrip("\x00 "))


def verify_linked_nodes(path, animated_collisions, emitter_source_names):
    if not animated_collisions and not emitter_source_names:
        return
    document = glb_json(path)
    node_names = [node.get("name", "") for node in document.get("nodes", [])]
    for obj in animated_collisions:
        if node_names.count(obj.name) != 1:
            raise ValueError(f'{obj.name}: animated collision must resolve to exactly one glTF node')
    for name in emitter_source_names:
        if node_names.count(name) != 1:
            raise ValueError(f'{name}: emitter source must resolve to exactly one glTF node')
    if animated_collisions and not document.get("animations"):
        raise ValueError("Animated collision meshes were exported without glTF animation")


def checker_image_resolution(scale):
    required = max(64, int(math.ceil(max(1.0, scale))) * 4)
    resolution = 64
    while resolution < required and resolution < 2048:
        resolution *= 2
    return min(resolution, 2048)


def create_checker_image(name, checker):
    scale = max(1.0, float(checker.inputs["Scale"].default_value))
    color_a = checker.inputs["Color1"].default_value
    color_b = checker.inputs["Color2"].default_value
    resolution = checker_image_resolution(scale)
    pixels = array("f", [0.0]) * (resolution * resolution * 4)
    index = 0
    for y in range(resolution):
        tile_y = int(math.floor(((y + 0.5) / resolution) * scale))
        for x in range(resolution):
            tile_x = int(math.floor(((x + 0.5) / resolution) * scale))
            color = color_a if (tile_x + tile_y) % 2 == 0 else color_b
            pixels[index] = color[0]
            pixels[index + 1] = color[1]
            pixels[index + 2] = color[2]
            pixels[index + 3] = color[3]
            index += 4
    image = bpy.data.images.new(name, width=resolution, height=resolution, alpha=True)
    image.colorspace_settings.name = "sRGB"
    image.pixels.foreach_set(pixels)
    image.update()
    image.pack()
    return image


def add_export_checker_textures(objects):
    records = []
    materials = []
    seen = set()
    for obj in objects:
        if obj.type != "MESH":
            continue
        for material in obj.data.materials:
            if material is not None and material.as_pointer() not in seen:
                seen.add(material.as_pointer())
                materials.append(material)
    try:
        for material in materials:
            if not material.node_tree:
                continue
            for checker in [node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeTexChecker"]:
                color_output = checker.outputs.get("Color")
                if color_output is None or not color_output.is_linked or checker.inputs["Vector"].is_linked:
                    continue
                for link in list(color_output.links):
                    target = link.to_socket
                    if link.to_node.bl_idname not in {"ShaderNodeBsdfPrincipled", "ShaderNodeEeveeSpecular"} or target.name != "Base Color":
                        continue
                    image = create_checker_image(f"Babylon Lite {material.name} checker", checker)
                    texture = material.node_tree.nodes.new("ShaderNodeTexImage")
                    texture.name = "Babylon Lite export checker"
                    texture.image = image
                    texture.interpolation = "Closest"
                    texture.extension = "REPEAT"
                    old_from = link.from_socket
                    material.node_tree.links.remove(link)
                    material.node_tree.links.new(texture.outputs["Color"], target)
                    records.append((material, texture, image, old_from, target))
                    break
    except Exception:
        remove_export_checker_textures(records)
        raise
    return records


def remove_export_checker_textures(records):
    for material, texture, image, old_from, target in reversed(records):
        if material.node_tree:
            for link in list(target.links):
                if link.from_node == texture:
                    material.node_tree.links.remove(link)
            material.node_tree.links.new(old_from, target)
            material.node_tree.nodes.remove(texture)
        bpy.data.images.remove(image)


def export_glb(path, objects, export_lights, decimate, target_triangles, frame_start, frame_end, animated_collisions=(), emitter_source_names=()):
    if not objects:
        raise ValueError("No visible presentation meshes are available for scene.glb")
    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    previous_selected = list(bpy.context.selected_objects)
    previous_visibility = [(obj, obj.hide_render, obj.hide_viewport, obj.hide_get()) for obj in objects]
    previous_frame_start = bpy.context.scene.frame_start
    previous_frame_end = bpy.context.scene.frame_end
    previous_frame = bpy.context.scene.frame_current
    decimation_modifiers = add_export_decimation(objects, target_triangles, animated_collisions) if decimate else []
    checker_records = []
    try:
        checker_records = add_export_checker_textures(objects)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.hide_render = False
            obj.hide_viewport = False
            obj.hide_set(False)
            obj.select_set(True)
        view_layer.objects.active = objects[0]
        bpy.context.scene.frame_start = frame_start
        bpy.context.scene.frame_end = frame_end
        bpy.context.scene.frame_set(frame_start)
        export_options = dict(
            filepath=path,
            export_format="GLB",
            use_selection=True,
            export_apply=True,
            export_yup=True,
            export_lights=export_lights,
            export_import_convert_lighting_mode="COMPAT",
            export_animations=True,
            export_frame_range=True,
            export_frame_step=1,
            export_anim_slide_to_zero=True,
            export_force_sampling=True,
            export_bake_animation=True,
            export_animation_mode="SCENE",
        )
        available_options = {prop.identifier for prop in bpy.ops.export_scene.gltf.get_rna_type().properties}
        bpy.ops.export_scene.gltf(**{key: value for key, value in export_options.items() if key in available_options})
        verify_linked_nodes(path, animated_collisions, emitter_source_names)
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for obj, hide_render, hide_viewport, hidden in previous_visibility:
            obj.hide_render = hide_render
            obj.hide_viewport = hide_viewport
            obj.hide_set(hidden)
        for obj in previous_selected:
            if obj.name in view_layer.objects:
                obj.select_set(True)
        view_layer.objects.active = previous_active
        for obj, modifier in decimation_modifiers:
            obj.modifiers.remove(modifier)
        remove_export_checker_textures(checker_records)
        bpy.context.scene.frame_start = previous_frame_start
        bpy.context.scene.frame_end = previous_frame_end
        bpy.context.scene.frame_set(previous_frame)
        view_layer.update()


def ffp3_vector_count(path):
    with open(path, "rb") as stream:
        header = stream.read(16)
    if len(header) != 16:
        raise ValueError(f"{os.path.basename(path)} is truncated")
    surface, boundary, interior, id_limit = struct.unpack("<4I", header)
    count = surface + boundary + interior
    expected = 16 + id_limit * 12 + count * 12
    if os.path.getsize(path) != expected:
        raise ValueError(f"{os.path.basename(path)} has an invalid payload length")
    return count, id_limit


def baked_initial_state_paths(domain, frame):
    bakefiles = os.path.join(domain.flip_fluid.domain.cache.get_cache_abspath(), "bakefiles")
    suffix = str(frame).zfill(6) + ".ffp3"
    return os.path.join(bakefiles, "fluidparticles" + suffix), os.path.join(bakefiles, "fluidparticlesvelocity" + suffix)


def build_baked_initial_state(domain, frame):
    positions_path, velocities_path = baked_initial_state_paths(domain, frame)
    if not os.path.isfile(positions_path) or not os.path.isfile(velocities_path):
        raise ValueError(
            f"Exact initial state requires baked fluid particle positions and velocities at frame {frame}; "
            "enable this option before baking, then bake the simulation"
        )
    count, position_id_limit = ffp3_vector_count(positions_path)
    velocity_count, velocity_id_limit = ffp3_vector_count(velocities_path)
    if count != velocity_count:
        raise ValueError(f"Initial position/velocity counts differ at frame {frame}: {count:,} vs {velocity_count:,}")
    with open(positions_path, "rb") as stream:
        positions = stream.read()
    with open(velocities_path, "rb") as stream:
        velocities = stream.read()
    position_offset = 16 + position_id_limit * 12
    velocity_offset = 16 + velocity_id_limit * 12
    output = bytearray(INITIAL_STATE_HEADER_BYTES + count * 24)
    struct.pack_into("<3IiI3I", output, 0, INITIAL_STATE_MAGIC, 1, count, frame, 1, 0, 0, 0)
    for index in range(count):
        px, py, pz = struct.unpack_from("<3f", positions, position_offset + index * 12)
        vx, vy, vz = struct.unpack_from("<3f", velocities, velocity_offset + index * 12)
        struct.pack_into("<3f", output, INITIAL_STATE_HEADER_BYTES + index * 12, px, pz, -py)
        struct.pack_into("<3f", output, INITIAL_STATE_HEADER_BYTES + count * 12 + index * 12, vx, vz, -vy)
    return {"bytes": bytes(output), "count": count, "frame": frame}


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

    derived, emitters, sinks, _, _, particle_count = targeted_flow_setup(scene, domain, grid_position, grid_size, summary["derived"])
    emitter_source_names = [emitter["sourceNode"] for emitter in emitters if "sourceNode" in emitter]
    presentation_objects = summary["presentation_objects"]
    collision_objects = summary["collision_objects"]
    animated_objects = summary["animated_collisions"]
    frame_start = summary["frame_start"]
    frame_end = summary["frame_end"]
    initial_state = build_baked_initial_state(domain, frame_start) if scene.blitefluid_export_initial_state else None
    if initial_state is not None:
        particle_count = max(particle_count, initial_state["count"])
    previous_frame = scene.frame_current
    animated_payloads = []
    try:
        scene.frame_set(frame_start)
        collision = bake_collision(
            collision_objects,
            grid_position,
            grid_size,
            summary["sdf_resolution"],
            context.window_manager,
        )
        for index, obj in enumerate(animated_objects):
            resolution = int(obj.blitefluid_animated_sdf_resolution)
            sdf, dims, voxel_count = bake_local_collision(obj, resolution, context.window_manager)
            animated_payloads.append(
                {
                    "id": f"animated-collision-{index + 1:04d}",
                    "node": obj.name,
                    "space": "node-local",
                    "resolution": resolution,
                    "bakeFrame": frame_start,
                    "presentation": bool(not obj.hide_render and obj.visible_get()),
                    "enabled": True,
                    "trilinear": True,
                    "bytes": sdf,
                    "dims": dims,
                    "voxelCount": voxel_count,
                }
            )
    finally:
        scene.frame_set(previous_frame)
    preset = default_preset(scene, grid_position, grid_size, emitters, sinks, derived, particle_count)

    if scene.blitefluid_separate_files:
        base_path = os.path.splitext(filepath)[0]
        glb_path = base_path + ".glb"
        sdf_path = base_path + ".sdf"
        export_glb(
            glb_path,
            presentation_objects,
            scene.blitefluid_export_lights,
            scene.blitefluid_decimate_meshes,
            scene.blitefluid_target_triangles,
            frame_start,
            frame_end,
            animated_objects,
            emitter_source_names,
        )
        animated_entries = []
        sdf_container = bytearray(collision)
        byte_offset = len(collision)
        for payload in animated_payloads:
            byte_length = len(payload["bytes"])
            sdf_container.extend(payload["bytes"])
            animated_entries.append(
                {
                    **{key: value for key, value in payload.items() if key not in {"bytes", "dims", "voxelCount"}},
                    "sdf": os.path.basename(sdf_path),
                    "byteOffset": byte_offset,
                    "byteLength": byte_length,
                }
            )
            byte_offset += byte_length
        initial_state_entry = None
        if initial_state is not None:
            byte_length = len(initial_state["bytes"])
            sdf_container.extend(initial_state["bytes"])
            initial_state_entry = {
                "data": os.path.basename(sdf_path),
                "byteOffset": byte_offset,
                "byteLength": byte_length,
                "count": initial_state["count"],
                "frame": initial_state["frame"],
                "space": "world",
            }
            byte_offset += byte_length
        with open(sdf_path, "wb") as stream:
            stream.write(zlib.compress(sdf_container, level=6))
        for stale_path in glob.glob(base_path + ".animated-collision-*.sdf"):
            os.remove(stale_path)
        preset["scene"] = {
            "encoding": "external",
            "glb": os.path.basename(glb_path),
            "collision": os.path.basename(sdf_path),
            "sdfCompression": "zlib",
            "collisionEnabled": True,
            "collisionTrilinear": True,
            "collisionByteLength": len(collision),
            "anchorPosition": grid_position,
            **({"initialState": initial_state_entry} if initial_state_entry else {}),
            **({"animatedCollisions": animated_entries} if animated_entries else {}),
        }
    else:
        with tempfile.TemporaryDirectory(prefix="blitefluid-") as temporary:
            glb_path = os.path.join(temporary, "scene.glb")
            export_glb(
                glb_path,
                presentation_objects,
                scene.blitefluid_export_lights,
                scene.blitefluid_decimate_meshes,
                scene.blitefluid_target_triangles,
                frame_start,
                frame_end,
                animated_objects,
                emitter_source_names,
            )
            with open(glb_path, "rb") as stream:
                glb = stream.read()
        preset["scene"] = {
            "encoding": "base64",
            "glb": base64.b64encode(glb).decode("ascii"),
            "collision": base64.b64encode(zlib.compress(collision, level=6)).decode("ascii"),
            "sdfCompression": "zlib",
            "collisionEnabled": True,
            "collisionTrilinear": True,
            "anchorPosition": grid_position,
            **(
                {
                    "initialState": {
                        "data": base64.b64encode(zlib.compress(initial_state["bytes"], level=6)).decode("ascii"),
                        "count": initial_state["count"],
                        "frame": initial_state["frame"],
                        "space": "world",
                    }
                }
                if initial_state is not None
                else {}
            ),
            **(
                {
                    "animatedCollisions": [
                        {
                            **{key: value for key, value in payload.items() if key not in {"bytes", "dims", "voxelCount"}},
                            "sdf": base64.b64encode(zlib.compress(payload["bytes"], level=6)).decode("ascii"),
                        }
                        for payload in animated_payloads
                    ]
                }
                if animated_payloads
                else {}
            ),
        }

    with open(filepath, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(preset, stream, indent=2)
        stream.write("\n")
    return len(emitters), len(sinks), len(collision_objects), len(animated_payloads), summary["dims"]


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
            emitters, sinks, colliders, animated_colliders, dims = export_bundle(context, self.filepath)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Exported {emitters} emitters, {sinks} sinks, {colliders} static and {animated_colliders} animated colliders; "
            f"static SDF {dims[0]} x {dims[1]} x {dims[2]}",
        )
        return {"FINISHED"}


class BLITEFLUID_PT_export(bpy.types.Panel):
    bl_label = "Babylon Lite Fluid"
    bl_idname = "BLITEFLUID_PT_export"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"

    def draw(self, context):
        initial_count, inflow_count = flow_emitter_counts(context.scene)
        initial_row = self.layout.row()
        initial_row.enabled = initial_count > 0
        initial_row.prop(context.scene, "blitefluid_target_initial_particles")
        inflow_row = self.layout.row()
        inflow_row.enabled = inflow_count > 0
        inflow_row.prop(context.scene, "blitefluid_target_inflow_particles")
        try:
            resolution, initial_particles, inflow_particles, total_particles = particle_target_preview(context.scene)
            self.layout.label(text=f"Resolution Divisions: {resolution}")
            self.layout.label(
                text=f"Particles: {initial_particles:,} initial + {inflow_particles:,} inflow = {total_particles:,} total"
            )
        except (AttributeError, TypeError, ValueError):
            self.layout.label(text="Particle estimate unavailable until a valid liquid setup exists")
        try:
            domain = find_domain(context.scene)
            frame_start, frame_end = simulation_frame_range(context.scene, domain)
            self.layout.label(text=f"Export frame range: {frame_start} to {frame_end}")
        except (AttributeError, TypeError, ValueError):
            self.layout.label(text="Export frame range unavailable until a valid liquid domain exists")
        initial_state_row = self.layout.row()
        try:
            domain = find_domain(context.scene)
            initial_state_row.enabled = is_flip_fluids_domain(domain)
        except (AttributeError, TypeError, ValueError):
            initial_state_row.enabled = False
        initial_state_row.prop(context.scene, "blitefluid_export_initial_state")
        if context.scene.blitefluid_export_initial_state and initial_state_row.enabled:
            try:
                frame_start, _ = simulation_frame_range(context.scene, domain)
                positions_path, velocities_path = baked_initial_state_paths(domain, frame_start)
                position_count, _ = ffp3_vector_count(positions_path)
                velocity_count, _ = ffp3_vector_count(velocities_path)
                if position_count != velocity_count:
                    raise ValueError("position/velocity particle counts differ")
                self.layout.label(text=f"Initial state: frame {frame_start}, {position_count:,} markers")
            except (OSError, ValueError):
                self.layout.label(text="Initial state cache missing; enable before baking and rebake", icon="ERROR")
        self.layout.prop(context.scene, "blitefluid_sdf_resolution", text="Static collision SDF resolution")
        try:
            dims, texture_bytes = collision_texture_metrics(context.scene)
            self.layout.label(text=f"Texture: {dims[0]} x {dims[1]} x {dims[2]} R32Float, {texture_bytes / (1024 * 1024):.2f} MiB")
        except (AttributeError, TypeError, ValueError):
            self.layout.label(text="Texture size unavailable until a valid liquid domain exists")
        animated = animated_meshes(context.scene)
        if animated:
            box = self.layout.box()
            box.label(text="Animated mesh collisions")
            for obj in animated:
                row = box.row()
                row.label(text=obj.name)
                row.prop(obj, "blitefluid_animated_sdf_resolution", text="")
                if int(obj.blitefluid_animated_sdf_resolution) == 0:
                    box.label(text=f"{obj.name}: visual only; no fluid collision", icon="INFO")
        self.layout.prop(context.scene, "blitefluid_separate_files")
        self.layout.prop(context.scene, "blitefluid_export_lights")
        self.layout.prop(context.scene, "blitefluid_decimate_meshes")
        if context.scene.blitefluid_decimate_meshes:
            self.layout.prop(context.scene, "blitefluid_target_triangles")
        self.layout.operator(BLITEFLUID_OT_export.bl_idname, icon="EXPORT")


CLASSES = (BLITEFLUID_OT_export, BLITEFLUID_PT_export)


def menu_func_export(self, context):
    self.layout.operator(BLITEFLUID_OT_export.bl_idname, text="Babylon Lite Fluid JSON (.json)")


def update_export_initial_state(scene, context):
    if not scene.blitefluid_export_initial_state:
        return
    try:
        domain = find_domain(context.scene)
    except ValueError:
        return
    if not is_flip_fluids_domain(domain):
        return
    particles = domain.flip_fluid.domain.particles
    particles.enable_fluid_particle_output = True
    particles.fluid_particle_output_amount = 1.0
    particles.enable_fluid_particle_surface_output = True
    particles.enable_fluid_particle_boundary_output = True
    particles.enable_fluid_particle_interior_output = True
    particles.enable_fluid_particle_velocity_vector_attribute = True


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.types.Scene.blitefluid_target_initial_particles = IntProperty(
        name="Target initial-fluid particles",
        description="Approximate total markers seeded from all enabled volume initial-fluid objects; 0 keeps the source domain resolution",
        default=0,
        min=0,
    )
    bpy.types.Scene.blitefluid_target_inflow_particles = IntProperty(
        name="Target inflow particles",
        description="Dormant particle slots reserved for all inflow emitters in addition to the initial-fluid markers",
        default=80000,
        min=0,
    )
    bpy.types.Scene.blitefluid_sdf_resolution = IntProperty(
        name="Static collision SDF resolution",
        description="Grid points on the longest collision-SDF axis; independent from Mantaflow Resolution Divisions",
        default=64,
        min=8,
        max=2048,
    )
    bpy.types.Object.blitefluid_animated_sdf_resolution = IntProperty(
        name="Animated collision SDF resolution",
        description="0 disables fluid collision; positive values set the longest local SDF axis for this rigid animated mesh",
        default=0,
        min=0,
        max=2048,
    )
    bpy.types.Scene.blitefluid_separate_files = BoolProperty(
        name="Export separate files",
        description="Write sibling .glb and .sdf files and reference them from the JSON instead of embedding base64 data",
        default=False,
    )
    bpy.types.Scene.blitefluid_export_initial_state = BoolProperty(
        name="Export baked initial positions/velocities",
        description="Enable full FLIP marker and velocity cache output for baking, then export the first baked frame as the exact Babylon FLIP initial state",
        default=False,
        update=update_export_initial_state,
    )
    bpy.types.Scene.blitefluid_export_lights = BoolProperty(
        name="Export lights",
        description="Include visible point, sun, and spot lights in the scene GLB",
        default=True,
    )
    bpy.types.Scene.blitefluid_decimate_meshes = BoolProperty(
        name="Decimate exported meshes",
        description="Temporarily add Blender Decimate modifiers to target a total triangle count across exported presentation meshes",
        default=False,
    )
    bpy.types.Scene.blitefluid_target_triangles = IntProperty(
        name="Target total triangles",
        description="Approximate total triangle target across exported presentation meshes",
        default=100000,
        min=1,
    )


def unregister():
    del bpy.types.Scene.blitefluid_target_triangles
    del bpy.types.Scene.blitefluid_decimate_meshes
    del bpy.types.Scene.blitefluid_export_lights
    del bpy.types.Scene.blitefluid_export_initial_state
    del bpy.types.Scene.blitefluid_separate_files
    del bpy.types.Object.blitefluid_animated_sdf_resolution
    del bpy.types.Scene.blitefluid_sdf_resolution
    del bpy.types.Scene.blitefluid_target_inflow_particles
    del bpy.types.Scene.blitefluid_target_initial_particles
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
