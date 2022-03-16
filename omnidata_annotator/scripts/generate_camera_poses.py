
"""
  Name: generate_camera_poses.py

  Desc: Generate camera poses inside the mesh. Camera locations are generated using Poisson Disc 
        Sampling to cover the whole space.

"""

import os
import sys
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Euler
import scipy.stats as stats

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from load_settings import settings


def get_cell_coords(pt):
    """Get the coordinates of the cell that pt = (x,y,z) falls in."""

    return int(pt[0] // a), int(pt[1] // a), int(pt[2] // a)


def get_neighbours(coords):
    """Return the indexes of points in cells neighbouring cell at coords.

    For the cell at coords = (x,y), return the indexes of points in the cells
    with neighbouring coordinates illustrated below: ie those cells that could 
    contain points closer than r.

                                     ooo
                                    ooooo
                                    ooXoo
                                    ooooo
                                     ooo

    """

    neighbor_range = [-2, -1, 0, 1, 2]
    dxdydz = [(x, y, z) for x in neighbor_range for y in neighbor_range for z in neighbor_range]
    neighbours = []
    for dx, dy, dz in dxdydz:
        neighbour_coords = coords[0] + dx, coords[1] + dy, coords[2] + dz
        if not (0 <= neighbour_coords[0] < nx and
                0 <= neighbour_coords[1] < ny and
                0 <= neighbour_coords[2] < nz):
            # We're off the grid: no neighbours here.
            continue
        neighbour_cell = cells[neighbour_coords]
        if neighbour_cell is not None:
            # This cell is occupied: store this index of the contained point.
            neighbours.append(neighbour_cell)
    return neighbours


def point_valid(pt):
    """Is pt a valid point to emit as a sample?

    It must be no closer than r from any other point: check the cells in its
    immediate neighbourhood.

    """
    cell_coords = get_cell_coords(pt)
    for idx in get_neighbours(cell_coords):
        nearby_pt = samples[idx]
        # Squared distance between or candidate point, pt, and this nearby_pt.
        distance2 = (nearby_pt[0] - pt[0]) ** 2 + (nearby_pt[1] - pt[1]) ** 2 + (nearby_pt[2] - pt[2]) ** 2
        if distance2 < r ** 2:
            # The points are too close, so pt is not a candidate.
            return False
    # All points tested: if we're here, pt is valid
    return True


def get_point(k, refpt):
    """Try to find a candidate point relative to refpt to emit in the sample.

    We draw up to k points from the annulus of inner radius r, outer radius 2r
    around the reference point, refpt. If none of them are suitable (because
    they're too close to existing points in the sample), return False.
    Otherwise, return the pt.

    """
    i = 0
    while i < k:

        theta = np.random.uniform(0, 2 * np.pi)
        z0 = np.random.uniform(-1, 1)
        x0 = np.sqrt(1 - z0 ** 2) * np.cos(theta)
        y0 = np.sqrt(1 - z0 ** 2) * np.sin(theta)
        T = np.random.uniform(r ** 3, (2 * r) ** 3)
        R = np.cbrt(T)
        pt = refpt[0] + x0 * R, refpt[1] + y0 * R, refpt[2] + z0 * R
        if not (0 <= pt[0] < WIDTH and 0 <= pt[1] < HEIGHT and 0 <= pt[2] < DEPTH):
            # This point falls outside the domain, so try again.
            continue
        if point_valid(pt):
            return pt
        i += 1
    # We failed to find a suitable point in the vicinity of refpt.
    return False


def sample_camera_locations_building(model, bbox_corners, n_samples, distance):
    """
    Generate camera locations inside the mesh using Poisson Disc Sampling.

    Args:
        model: The building mesh
        n_samples: Choose up to n_samples points around each reference point as candidates for a new, in Poisson Disc algorithm
        distance: Minimum distance between cameras (samples)

    Returns:
        camera_locations: List of generated camera locations

    """
    # k : sample point
    # r : Minimum distance between samples

    global a, k, r
    global nx, ny, nz
    global samples, cells
    global WIDTH, HEIGHT, DEPTH

    # find mesh bounding box
    # bbox_corners = [model.matrix_world * Vector(corner) for corner in model.bound_box]
    
    x_range, y_range, z_range = set(), set(), set()
    for corner in bbox_corners:
        x_range.add(corner[0])
        y_range.add(corner[1])
        z_range.add(corner[2])
    x_range = sorted(list(x_range))
    y_range = sorted(list(y_range))
    z_range = sorted(list(z_range))

    print("Mesh bounding box:")
    print("X axis : ", x_range)
    print("Y axis : ", y_range)
    print("Z axis : ", z_range)

    # find building floor
    floors = find_building_floors(model, z_range)

    WIDTH, HEIGHT, DEPTH = x_range[-1] - x_range[0], y_range[-1] - y_range[0], z_range[-1] - z_range[0]
    k = n_samples
    r = distance

    # Generate camera locations using Poisson Disc Sampling

    # Cell side length
    a = r / np.sqrt(3)

    # Number of cells in the x- and y- and z-directions of the grid
    nx, ny, nz = int(WIDTH / a) + 1, int(HEIGHT / a) + 1, int(DEPTH / a) + 1

    # A list of coordinates in the grid of cells
    coords_list = [(ix, iy, iz) for ix in range(nx) for iy in range(ny) for iz in range(nz)]
    # Initilalize the dictionary of cells: each key is a cell's coordinates, the
    # corresponding value is the index of that cell's point's coordinates in the
    # samples list (or None if the cell is empty).
    cells = {coords: None for coords in coords_list}

    # Pick a random point to start with.
    pt = (np.random.uniform(0, WIDTH), np.random.uniform(0, HEIGHT), np.random.uniform(0, DEPTH))
    samples = [pt]
    # Our first sample is indexed at 0 in the samples list...
    cells[get_cell_coords(pt)] = 0
    # ... and it is active, in the sense that we're going to look for more points
    # in its neighbourhood.
    active = [0]

    nsamples = 1
    # As long as there are points in the active list, keep trying to find samples.
    while active:
        # choose a random "reference" point from the active list.
        idx = np.random.choice(active)
        refpt = samples[idx]
        # Try to pick a new point relative to the reference point.
        pt = get_point(k, refpt)
        if pt:
            # Point pt is valid: add it to the samples list and mark it as active
            samples.append(pt)
            nsamples += 1
            active.append(len(samples) - 1)
            cells[get_cell_coords(pt)] = len(samples) - 1
        else:
            # We had to give up looking for valid points near refpt, so remove it
            # from the list of "active" points.
            active.remove(idx)

    camera_locations = []
    for pt in samples:
        point = Vector((pt[0] + x_range[0], pt[1] + y_range[0], pt[2] + z_range[0]))

        # check if camera height is between min and max camera height specified in the setting
        flag = False
        for f in range(len(floors) - 1):
            floor_z_range = [floors[f] + settings.MIN_CAMERA_HEIGHT, floors[f] + settings.MAX_CAMERA_HEIGHT]
            if floor_z_range[0] < point[2] < floor_z_range[1]:
                flag = True
                break
        if not flag:
            continue

        # check if camera is inside the mesh
        if not camera_inside_mesh(point, model):
            print("Camera outside mesh...")
            continue

        # check closest distance to a point on mesh
        loc = model.closest_point_on_mesh(point)[1]
        distance_to_closest_point = np.sqrt(
            (loc[0] - point[0]) ** 2 + (loc[1] - point[1]) ** 2 + (loc[2] - point[2]) ** 2)
        if distance_to_closest_point < settings.MIN_CAMERA_DISTANCE_TO_MESH:
            print("Camera distance to mesh less than MIN_CAMERA_DISTANCE_TO_MESH...")
            continue

        camera_locations.append((point[0], point[1], point[2]))

    return camera_locations


def camera_inside_mesh(point, model):
    ''' 
    Check if generated camera is inside the mesh. 

    '''
    axes = [Vector((1, 0, 0)), Vector((0, 1, 0))]  # ignore the ceiling
    outside = False
    for axis in axes:
        orig = point
        count = 0
        while True:
            ray_hit, location, normal, index = model.ray_cast(orig, orig + axis * 10000.0)
            if index == -1: break
            count += 1
            orig = location + axis * 0.00001
        if count % 2 == 0:
            outside = True
            break
    return not outside


def sample_camera_quaternion(num_samples):
    ''' 
    Sample camera rotations.
    Yaw is sampled uniformly between -180 and 180 degrees.
    Roll is sampled from a Gaussian distribution with zero mean and standard deviation equal to MAX_CAMERA_ROLL
    Pitch will be determined in point generation process. 

    '''
    quaternion_samples = []
    yaw_samples = np.random.uniform(-180, 180, num_samples) * (np.pi / 180)
    roll_samples = stats.truncnorm(-1, 1, loc=0, scale=settings.MAX_CAMERA_ROLL).rvs(num_samples) * (np.pi / 180)

    for i in range(num_samples):
        roll = roll_samples[i]
        yaw = yaw_samples[i]
        pitch = 0.

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)

        q_w = cp * cr * cy + sp * sr * sy
        q_x = sp * cr * cy - cp * sr * sy
        q_y = cp * sr * cy + sp * cr * sy
        q_z = cp * cr * sy - sp * sr * cy

        quaternion_samples.append((q_w, q_x, q_y, q_z))
    return quaternion_samples

def find_building_floors(model, z_range):
    ''' 
    Find building floors based on mesh density along Z axis.

    '''
    height = z_range[-1] - z_range[0]

    bpy.context.scene.objects.active = model
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(model.data)

    z_values = [v.co.to_tuple()[2] for v in bm.verts]
    for face in bm.faces:
        z_values.append(face.calc_center_median()[2])
    z_values = sorted(z_values)

    density, division = np.histogram(z_values, bins=int(height // settings.FLOOR_THICKNESS), density=True)
    mean_density = density.mean()
    print("Z-axis density : {}, mean density = {}".format(density, mean_density))

    floors = [z_range[0]]
    for i in range(len(density)):
        # Find a new floor if mesh density is more than 1.5 * mean_density
        if density[i] > 1.5 * mean_density and \
            floors[-1] + settings.FLOOR_HEIGHT < division[i] < z_range[-1] - settings.FLOOR_HEIGHT:

            floors.append(division[i])

    floors.append(z_range[-1])

    print("Building floors : ", floors)

    bm.free()
    bpy.ops.object.mode_set(mode='OBJECT')

    return floors


def sample_camera_locations_object(model, n_samples):
    """
    Generate camera locations on a sphere surrounding the mesh.

    Args:
        model: The object mesh
        n_samples: Generate n_sample cameras on the sphere
       
    Returns:
        camera_locations: List of generated camera locations

    """


    # find mesh bounding box
    bbox_corners = [model.matrix_world * Vector(corner) for corner in model.bound_box]
    x_range, y_range, z_range = set(), set(), set()
    for corner in bbox_corners:
        x_range.add(corner[0])
        y_range.add(corner[1])
        z_range.add(corner[2])
    x_range = sorted(list(x_range))
    y_range = sorted(list(y_range))
    z_range = sorted(list(z_range))

    print("Mesh bounding box:")
    print("X axis : ", x_range)
    print("Y axis : ", y_range)
    print("Z axis : ", z_range)

    radius = settings.SPHERE_SCALING_FACTOR * 0.5 * \
        np.sqrt((x_range[-1]-x_range[0])**2 + \
                (y_range[-1]-y_range[0])**2 + \
                (z_range[-1]-z_range[0])**2)

    center = np.array([x_range[-1] + x_range[0], y_range[-1] + y_range[0], z_range[-1] + z_range[0]]) / 2

    print("radius = ", radius)
    vec = np.random.randn(n_samples, 3)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= radius
    
    camera_locations = []
    for loc in vec:
        loc += center
        camera_locations.append((loc[0], loc[1], loc[2]))

    return camera_locations


