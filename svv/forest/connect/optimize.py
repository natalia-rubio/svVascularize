import numpy
import numpy as np
from tqdm import trange

from geomdl import BSpline, utilities
from scipy import optimize
from scipy.interpolate import splprep, splev, CubicSpline
import pyvista as pv
from copy import deepcopy
from time import perf_counter
from svv.utils.spatial.c_distance import minimum_segment_distance
#from svv.forest.utils.angles import cost_angles
from svv.forest.connect.assign import assign_network, assign_network_vector

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
def test_vessels(size):
    vessels = numpy.random.random((size, 7))
    vessels[:, :6] *= 10
    return vessels


def test_lines(size):
    lines = numpy.zeros((size, 3))
    lines[:, 0] = numpy.linspace(0, 1, size) * 10
    lines[:, 1] = 0.5 * 10
    lines[:, 2] = 0.5 * 10
    return lines

class Spline(object):
    def __init__(self, pts, degree=3, cubic_hermite=True, weights=None, tangents=None):
        self.ctrlpts = pts
        self.degree = degree
        self.sample_size = None
        self.cubic_hermite = cubic_hermite
        if cubic_hermite:
            bc_num = 0
            if not isinstance(tangents, type(None)):
                if not isinstance(tangents[0], type(None)):
                    tangent_start = tangents[0]
                    bc_num += 1
                if not isinstance(tangents[1], type(None)):
                    tangent_end = tangents[1]
                    bc_num += 1
            else:
                tangent_start = None
                tangent_end = None
            if not isinstance(weights, type(None)):
                if not isinstance(weights[0], type(None)):
                    tangent_start = tangent_start * weights[0]
                else:
                    tangent_start = None
                if not isinstance(weights[1], type(None)):
                    tangent_end = tangent_end * weights[1]
                else:
                    tangent_end = None
            t = numpy.linspace(0, 1, pts.shape[0] - bc_num)
            tmp = numpy.zeros((pts.shape[0]-bc_num, 3))
            if bc_num == 2:
                tmp[0, :] = pts[0, :]
                tmp[-1, :] = pts[-1, :]
                tmp[1:-1, :] = pts[2:-2, :]
                self.spline_x = CubicSpline(t, tmp[:, 0], bc_type=((1, tangent_start[0]), (1, tangent_end[0])))
                self.spline_y = CubicSpline(t, tmp[:, 1], bc_type=((1, tangent_start[1]), (1, tangent_end[1])))
                self.spline_z = CubicSpline(t, tmp[:, 2], bc_type=((1, tangent_start[2]), (1, tangent_end[2])))
            elif bc_num == 1:
                if not isinstance(tangent_start, type(None)):
                    tmp[0, :] = pts[0, :]
                    tmp[1:, :] = pts[2:, :]
                    self.spline_x = CubicSpline(t, tmp[:, 0], bc_type=((1, tangent_start[0]), 'not-a-knot'))
                    self.spline_y = CubicSpline(t, tmp[:, 1], bc_type=((1, tangent_start[1]), 'not-a-knot'))
                    self.spline_z = CubicSpline(t, tmp[:, 2], bc_type=((1, tangent_start[2]), 'not-a-knot'))
                else:
                    tmp[-1, :] = pts[-1, :]
                    tmp[:-1, :] = pts[:-2, :]
                    self.spline_x = CubicSpline(t, tmp[:, 0], bc_type=('not-a-knot', (1, tangent_end[0])))
                    self.spline_y = CubicSpline(t, tmp[:, 1], bc_type=('not-a-knot', (1, tangent_end[1])))
                    self.spline_z = CubicSpline(t, tmp[:, 2], bc_type=('not-a-knot', (1, tangent_end[2])))
            else:
                tmp[:, :] = pts[:, :]
                self.spline_x = CubicSpline(t, tmp[:, 0])
                self.spline_y = CubicSpline(t, tmp[:, 1])
                self.spline_z = CubicSpline(t, tmp[:, 2])
        else:
            self.tck = splprep(pts.T, s=0, k=degree)
        self.evalpts = None

    def evaluate(self):
        t = numpy.linspace(0, 1, self.sample_size)
        if self.cubic_hermite:
            self.evalpts = numpy.vstack((self.spline_x(t), self.spline_y(t), self.spline_z(t))).T
        else:
            self.evalpts = numpy.array(splev(t, self.tck[0])).T
        return

    def __call__(self, t):
        if self.cubic_hermite:
            return numpy.vstack((self.spline_x(t), self.spline_y(t), self.spline_z(t))).T
        else:
            return numpy.array(splev(t, self.tck[0])).T

    def derivatives(self, order):
        t = numpy.linspace(0, 1, self.sample_size)
        der = None
        if self.cubic_hermite:
            dx = self.spline_x.derivative(order)
            dy = self.spline_y.derivative(order)
            dz = self.spline_z.derivative(order)
            der = numpy.vstack((dx(t), dy(t), dz(t)))
        else:
            der = numpy.array(splev(t, self.tck[0], der=order)).T
        return der

    def curvature(self, num):
        t_start = numpy.linspace(0, 0.1, num//4)[:-1]
        t = numpy.linspace(0.1, 0.9, num//2)
        t_end = numpy.linspace(0.9, 1, num//4)[1:]
        t = numpy.hstack((t_start, t, t_end))[1:-1]
        dx = self.spline_x.derivative(1)
        dy = self.spline_y.derivative(1)
        dz = self.spline_z.derivative(1)
        ddx = self.spline_x.derivative(2)
        ddy = self.spline_y.derivative(2)
        ddz = self.spline_z.derivative(2)
        curvature = (numpy.linalg.norm(numpy.cross(numpy.vstack((dx(t), dy(t), dz(t))), numpy.vstack((ddx(t), ddy(t), ddz(t))), axis=0), axis=0)
                                       / numpy.linalg.norm(numpy.vstack((dx(t), dy(t), dz(t))), axis=0)**3)
        return curvature


def spline_factory(p1, p2, p3, p4, clamp_first=True, clamp_second=True, ctlpts_func=None):
    def create_spline(data, p1_=p1, p2_=p2, p3_=p3, p4_=p4, c1=clamp_first, c2=clamp_second, ctlpts_func=ctlpts_func):
        """
        if c1 and c2:
            diff = data[1:-1].reshape(-1, 3)
            t = numpy.linspace(0, 1, diff.shape[0] + 2)
            pts = ctlpts_func(t)
            tmp0 = 0.5 * (numpy.tanh(4 * (data[0] - 0.5)) + 1)
            pts[0, :] = (p1_ * data[0] + p2_ * (1 - data[0]))
            pts[-1, :] = (p4_ * data[-1] + p3_ * (1 - data[-1]))
            pts[1:-1, :] += diff
            #ctr_pts = numpy.vstack((p1_, pts, p3_))
            ctr_pts = pts
        elif c1:
            diff = data[1:].reshape(-1, 3)
            t = numpy.linspace(0, 1, diff.shape[0] + 1)
            pts = ctlpts_func(t)
            pts[0, :] = (p1_ * 0.5 * (numpy.tanh(4 * (data[0] - 0.5)) + 1) +
                         pts[0, :] * (1 - 0.5 * (numpy.tanh(4 * (data[0] - 0.5)) + 1)))
            pts[1:, :] += diff
            #ctr_pts = numpy.vstack((p1_, pts))
            ctr_pts = pts
        elif c2:
            diff = data[:-1].reshape(-1, 3)
            t = numpy.linspace(0, 1, diff.shape[0] + 1)
            pts = ctlpts_func(t)
            pts[-1, :] = (p4_ * 0.5 * (numpy.tanh(4 * (data[-1] - 0.5)) + 1) +
                          pts[-1, :] * (1 - 0.5 * (numpy.tanh(4 * (data[-1] - 0.5)) + 1)))
            pts[:-1, :] += diff
            #ctr_pts = numpy.vstack((pts, p3_))
            ctr_pts = pts
        else:
            diff = data.reshape(-1, 3)
            t = numpy.linspace(0, 1, diff.shape[0])
            pts = ctlpts_func(t)
            ctr_pts = pts + diff
        """
        if c1 and c2:
            diff = data[2:-2].reshape(-1, 3)
            t = numpy.linspace(0.25, 0.75, diff.shape[0])
            pts = ctlpts_func(t)
            direction_start = (p2_ - p1_)
            direction_end = (p3_ - p4_)
            direction_start = direction_start / numpy.linalg.norm(direction_start)
            direction_end = direction_end / numpy.linalg.norm(direction_end)
            tmp_0 = 0.5 + 0.5 * 0.5 * (numpy.tanh(4 * (data[0] - 0.5)) + 1)
            tmp_1 = 0.5 + 0.5 * 0.5 * (numpy.tanh(4 * (data[-1] - 0.5)) + 1)
            tmp_0 = 0.5 + 0.5*abs(data[0])
            tmp_1 = 0.5 + 0.5*abs(data[-1])
            ctr_pt_start = p1_ * (1 - tmp_0) + tmp_0 * p2_
            ctr_pt_end = p3_ * (1 - tmp_1) + tmp_1 * p4_
            pts += diff
            tmp_2 = 0.5 + 0.5*0.5 * (numpy.tanh(4 * (data[1] - 0.5)) + 1)
            tmp_3 = 0.5 + 0.5*0.5 * (numpy.tanh(4 * (data[-2] - 0.5)) + 1)
            tmp_2 = abs(data[1])
            tmp_3 = abs(data[-2])
            #ctr_pt_start_vec = ctr_pt_start * (1 - tmp_2) + tmp_2 * p2_
            #ctr_pt_end_vec = ctr_pt_end * (1 - tmp_3) + tmp_3 * p4_
            #ctr_pts = numpy.vstack((ctr_pt_start, ctr_pt_start_vec, pts, ctr_pt_end_vec, ctr_pt_end))
            scale_1 = np.linalg.norm(p2_ - p1_)
            scale_2 = np.linalg.norm(p4_ - p3_)
            weights = [scale_1*tmp_2, scale_2*tmp_3]
            tangents = [direction_start, direction_end]
            ctr_pts = numpy.vstack((ctr_pt_start, ctr_pt_start + weights[0]*tangents[0], pts, ctr_pt_end - weights[1]*tangents[1], ctr_pt_end))
        elif c1:
            diff = data[2:].reshape(-1, 3)
            t = numpy.linspace(0.25, 0.75, diff.shape[0])
            pts = ctlpts_func(t)
            direction_start = (p2_ - p1_)
            direction_start = direction_start / numpy.linalg.norm(direction_start)
            tmp_0 = 0.5 + 0.5 * 0.5 * (numpy.tanh(4 * (data[0] - 0.5)) + 1)
            ctr_pt_start = p1_ * (1 - tmp_0) + tmp_0 * p2_
            pts += diff
            tmp_2 = 0.5 + 0.5*0.5 * (numpy.tanh(4 * (data[1] - 0.5)) + 1)
            pts += diff
            scale_1 = np.linalg.norm(p2_ - p1_)
            weights = [scale_1*tmp_2, None]
            tangents = [direction_start, None]
            ctr_pts = numpy.vstack((ctr_pt_start, ctr_pt_start + weights[0]*tangents[0], pts, p4_))
            weights = weights
            tangents = tangents
        elif c2:
            diff = data[:-2].reshape(-1, 3)
            t = numpy.linspace(0.25, 0.75, diff.shape[0])
            pts = ctlpts_func(t)
            direction_end = (p3_ - p4_)
            direction_end = direction_end / numpy.linalg.norm(direction_end)
            tmp_1 = 0.5 + 0.5 * 0.5 * (numpy.tanh(4 * (data[-1] - 0.5)) + 1)
            ctr_pt_end = p3_ * (1 - tmp_1) + tmp_1 * p4_
            pts += diff
            tmp_3 = 0.5 + 0.5*0.5 * (numpy.tanh(4 * (data[-2] - 0.5)) + 1)
            pts += diff
            scale_2 = np.linalg.norm(p4_ - p3_)
            weights = [None, scale_2*tmp_3]
            tangents = [None, direction_end]
            ctr_pts = numpy.vstack((p2_, pts, ctr_pt_end - weights[1]*tangents[1], ctr_pt_end))
            weights = weights
            tangents = tangents
        else:
            diff = data.reshape(-1, 3)
            t = numpy.linspace(0, 1, diff.shape[0]+2)[1:-1]
            pts = ctlpts_func(t)
            pts = pts + diff
            ctr_pts = numpy.vstack((p2_, pts, p4_))
            weights = None
            tangents = None
        #ctr_pts = ctr_pts.tolist()
        curve = Spline(ctr_pts, weights=weights, tangents=tangents)
        return curve
    return create_spline


def bezier_factory(p1, p2, p3, p4, clamp_first=True, clamp_second=True, ctlpts_func=None):
    def create_bezier(data, p1_=p1, p2_=p2, p3_=p3, p4_=p4, c1=clamp_first, c2=clamp_second, ctlpts_func=ctlpts_func):
        if c1 and c2:
            diff = data[2:-2].reshape(-1, 3)
            t = numpy.linspace(0.25, 0.75, diff.shape[0])
            pts = ctlpts_func(t)
            tmp_0 = 0.5 * (numpy.tanh(4 * (data[0] - 0.5)) + 1)
            tmp_1 = 0.5 * (numpy.tanh(4 * (data[-1] - 0.5)) + 1)
            ctr_pt_start = p1_ + (0.5 + 0.5 * tmp_0) * (p2_ - p1_)
            ctr_pt_end = p3_ + (0.5 + 0.5 * tmp_1) * (p4_ - p3_)
            pts += diff
            tmp_2 = 0.5 * (numpy.tanh(4 * (data[1] - 0.5)) + 1)
            tmp_3 = 0.5 * (numpy.tanh(4 * (data[-2] - 0.5)) + 1)
            ctr_pt_start_vec = ctr_pt_start + (0.1 + max(0.0, data[1])) * (p2_ - ctr_pt_start)
            ctr_pt_end_vec = ctr_pt_end + (0.1 + max(0.0, data[-2])) * (p4_ - ctr_pt_end)
            ctr_pts = numpy.vstack((ctr_pt_start, ctr_pt_start_vec, pts, ctr_pt_end_vec, ctr_pt_end))
        elif c1:
            diff = data[2:].reshape(-1, 3)
            t = numpy.linspace(0, 1, diff.shape[0] + 2)[1:-1]
            pts = ctlpts_func(t)
            ctr_pt_start = p1_ + data[0] * (p2_ - p1_)
            ctr_pt_start_vec = ctr_pt_start + data[1] * ((p2_ - p1_) / numpy.linalg.norm(p2_ - p1_))
            pts += diff
            ctr_pts = numpy.vstack((ctr_pt_start, ctr_pt_start_vec, pts, p4_))
        elif c2:
            diff = data[:-2].reshape(-1, 3)
            t = numpy.linspace(0, 1, diff.shape[0] + 2)[1:-1]
            pts = ctlpts_func(t)
            ctr_pt_end = p4_ + data[-1] * (p3_ - p4_)
            ctr_pt_end_vec = ctr_pt_end + data[-2] * (p4_ - p3_) / numpy.linalg.norm(p4_ - p3_)
            pts += diff
            ctr_pts = numpy.vstack((p2_, pts, ctr_pt_end_vec, ctr_pt_end))
        else:
            diff = data.reshape(-1, 3)
            t = numpy.linspace(0, 1, diff.shape[0]+2)[1:-1]
            pts = ctlpts_func(t)
            pts = pts + diff
            ctr_pts = numpy.vstack((p2_, pts, p4_))
        ctr_pts = ctr_pts.tolist()
        curve = BSpline.Curve()
        curve.degree = len(ctr_pts) - 1
        curve.ctrlpts = ctr_pts
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        return curve
    return create_bezier


def cost_bounds(bounds, pts, func=lambda x:max(0,x), boundary_func=None):
    bounds_score = 0.0
    if boundary_func is None:
        for i in range(pts.shape[0]):
            bounds_score += func(bounds[0, 0] - pts[i, 0])
            bounds_score += func(pts[i, 0] - bounds[0, 1])
            bounds_score += func(bounds[1, 0] - pts[i, 1])
            bounds_score += func(pts[i, 1] - bounds[1, 1])
            bounds_score += func(bounds[2, 0] - pts[i, 2])
            bounds_score += func(pts[i, 2] - bounds[2, 1])
    else:
        #print("Using boundary function")
        #for i in range(pts.shape[0]):
        #    bounds_score += func(boundary_func([pts[i, 0], pts[i, 1], pts[i, 2], bounds_k]))
        bounds_score = boundary_func(pts).flatten()
        bounds_score = bounds_score[bounds_score > 0]
        if len(bounds_score) > 0:
            bounds_score = numpy.max(bounds_score)
        else:
            bounds_score = 0.0
    return bounds_score


def collision_test(vessels, pts, radius, clearance):
    # this is not a perfect function and still needs to be optimized for accuracy in the line
    # segment distance calculation
    lines = numpy.zeros((pts.shape[0]-1, 6))
    lines[:, 0:3] = pts[:-1, :]
    lines[:, 3:6] = pts[1:, :]
    #line_centers = (lines[:, 0:3] + lines[:, 3:6]) / 2
    #vessel_centers = (vessels[:, 0:3] + vessels[:, 3:6]) / 2
    #vessel_lengths = numpy.linalg.norm(vessels[:, 3:6] - vessels[:, 0:3], axis=1).reshape(1, -1)
    #line_lengths = numpy.linalg.norm(lines[:, 3:6] - lines[:, 0:3], axis=1).reshape(-1, 1)
    #dists = numpy.linalg.norm(line_centers[:, numpy.newaxis, :] - vessel_centers[numpy.newaxis, :, :], axis=2)
    #dists = dists - (vessel_lengths + line_lengths) / 2
    #dists = numpy.min(dists, axis=1)
    dists = numpy.min(minimum_segment_distance(vessels, lines), axis=0)
    return numpy.sum(dists[dists < 2.0*radius + clearance])


def cost_new(data, curve_generator=None, r=None, collision_vessels=None, clearance=None, bounds=None, sample_size=None,
             boundary_func=None, length_threshold=None, length_bounds=None, start=None, end=None, angle_threshold=70,
             clamp_first=True, clamp_second=True):
    #start = perf_counter()
    curve = curve_generator(data)
    curve.sample_size = sample_size
    curve.evaluate()
    pts = numpy.array(curve.evalpts)
    #end = perf_counter()
    #print("Time to evaluate curve: ", end-start)
    #start = perf_counter()
    if isinstance(collision_vessels, type(None)):
        collision_cost = 0.0
    else:
        collision_cost = collision_test(collision_vessels, pts, r, clearance)
    #end = perf_counter()
    #print("Time to evaluate collisions: ", end-start)
    #start = perf_counter()

    if not isinstance(bounds, type(None)):
        bounds_cost = cost_bounds(bounds, pts)
    else:
        bounds_cost = 0.0
    if not isinstance(boundary_func, type(None)):
        bounds_cost = cost_bounds(bounds, pts, boundary_func=boundary_func)
    else:
        #print("no bounds")
        bounds_cost = 0.0
    #bounds_cost = 0.0
    #end = perf_counter()
    #print("Time to evaluate bounds: ", end-start)
    #start = perf_counter()
    u = numpy.linspace(0, 1, pts.shape[0])[1:-1]
    radius_of_curvature = []
    #derivatives_1 = []
    #derivatives_2 = []
    derivatives_1 = curve.derivatives(1).T
    first = derivatives_1[:-1, :]
    last = derivatives_1[1:, :]
    flips = np.linalg.norm(first*last, axis=1)
    flips = np.abs(np.sum(flips[flips < 0]))
    #derivatives_2 = curve.derivatives(2)[:, 1:-1]
    #derivatives_3 = curve.derivatives(3)[:, 1:-1]
    #for i in range(u.shape[0]):
    #    derivatives = curve.derivatives(u[i], 2)
    #    #derivatives_1.append(derivatives[1])
    #    #derivatives_2.append(derivatives[2])
    #    radius_of_curvature.append(1/(numpy.linalg.norm(numpy.cross(derivatives[1], derivatives[2]))/numpy.linalg.norm(derivatives[1])**3))
    #derivatives_1 = numpy.array(derivatives_1)
    #derivatives_2 = numpy.array(derivatives_2)
    #curvature = numpy.linalg.norm(numpy.cross(derivatives_1, derivatives_2, axis=0), axis=0)/numpy.linalg.norm(derivatives_1, axis=0)**3

    #torsion = numpy.linalg.norm(numpy.cross(derivatives_1, derivatives_2, axis=0) * derivatives_3, axis=0)/numpy.linalg.norm(numpy.cross(derivatives_1, derivatives_2, axis=0), axis=0)**2
    #radius_of_curvature.append(1/curvature)
    roc= 1/curve.curvature(4*sample_size)
    #radius_of_curvature = numpy.array(radius_of_curvature)
    if len(roc[roc < 2 * r]) == 0:
        curvature_cost = 0.0
    else:
        curvature_cost = (2*r - numpy.mean(roc[roc < 2 * r]))/(2*r)
    if clamp_first:
        vector_start = pts[0, :] - start
        vector_1 = pts[1, :] - pts[0, :]
        vector_start = vector_start / numpy.linalg.norm(vector_start)
        vector_1 = vector_1 / numpy.linalg.norm(vector_1)
        flip_1 = max(0.0, (numpy.arccos(numpy.dot(vector_start, vector_1)) * (180 / numpy.pi)) - angle_threshold)
        curvature_cost += np.tanh(flip_1)*0.1
    if clamp_second:
        vector_end = pts[-1, :] - end
        vector_2 = pts[-2, :] - pts[-1, :]
        vector_end = vector_end / numpy.linalg.norm(vector_end)
        vector_2 = vector_2 / numpy.linalg.norm(vector_2)
        flip_2 = max(0.0, (numpy.arccos(numpy.dot(vector_end, vector_2))*(180/numpy.pi)) - angle_threshold)
        curvature_cost += np.tanh(flip_2)*0.1
    #curvature_cost += flips
    #vectors = get_all_vectors(pts)
    #angles = get_all_angles(vectors)
    #angle_cost = cost_angles(angles-angle_threshold)
    #angle_cost = cost_angles(pts, angle_threshold)
    #end = perf_counter()
    #print("Time to evaluate curvature: ", end-start)
    #start = perf_counter()
    spline_length = numpy.sum(numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1))
    if clamp_first:
        spline_length += numpy.linalg.norm(pts[0, :] - start)
    if clamp_second:
        spline_length += numpy.linalg.norm(pts[-1, :] - end)
    #end = perf_counter()
    #print("Time to evaluate length: ", end-start)
    length_cost = max(0.0, spline_length - length_bounds[1]*length_threshold)
    lower_length_cost = max(0.0, length_bounds[0]*length_threshold - spline_length)
    #print("Collision Cost: {}\tBounds Cost: {}\tCurvature Cost: {}\tLength Cost: {}".format(collision_cost,bounds_cost, curvature_cost, length_cost))
    pareto = np.tanh((np.tanh(max(0.0, abs(collision_cost))) + np.tanh(max(0.0, bounds_cost)) + np.tanh(max(0.0, curvature_cost)))) # + np.tanh(length_cost)))
    pareto_1 = max(0.0, abs(collision_cost))
    pareto_2 = max(0.0, bounds_cost)
    pareto_3 = max(0.0, curvature_cost)
    pareto_4 = np.tanh(length_cost/length_threshold)
    return (pareto +
            np.heaviside(pareto_1, 0)*2 +
            np.heaviside(pareto_2, 0)*4 +
            np.heaviside(pareto_3, 0) +
            np.heaviside(-pareto, 1) * np.heaviside(pareto_4, 0)*(0.5+0.5*pareto_4) +
            np.heaviside(-pareto, 1) * np.heaviside(-pareto_4, 1) * (0.5*np.tanh(lower_length_cost/length_threshold)))


class Connection(object):
    def __init__(self):
        self.proximal_0 = None
        self.distal_0 = None
        self.proximal_1 = None
        self.distal_1 = None
        self.radius_0 = None
        self.radius_1 = None
        self.solver = None
        self.bounds = None
        self.boundary_func = None
        self.ctrlpt_function = None
        self.curve_function = None
        self.physical_clearance = 0.0
        self.collision_vessels = None
        self.plotting_vessels = None
        self.clamp_first = True
        self.clamp_second = True
        self.history = []

    def set_first_vessel(self, proximal, distal, radius):
        self.proximal_0 = proximal
        self.distal_0 = distal
        self.radius_0 = radius

    def set_second_vessel(self, proximal, distal, radius):
        self.proximal_1 = proximal
        self.distal_1 = distal
        self.radius_1 = radius

    def set_solver(self, method='basinhopping'):
        if method == 'basinhopping':
            self.solver = optimize.basinhopping
        elif method == 'shgo':
            self.solver = optimize.shgo
        elif method == 'differential_evolution':
            self.solver = optimize.differential_evolution
        else:
            raise NotImplementedError("Solver method not implemented")

    def set_bounds(self, bounds, boundary_func=None):
        self.bounds = bounds
        self.boundary_func = boundary_func

    def set_ctrl_function(self, *args):
        """
        Set the function that seeds the control points for the initial
        guess of the optimization. The function should accept a vector
        of values between 0 and 1 and return a set of control points
        for the Bezier curve.
        """
        if len(args) == 1:
            self.ctrlpt_function = args[0]
        else:
            path_pts = numpy.vstack((self.distal_0, self.distal_1))
            k = 1
            tck = splprep(path_pts.T, s=0, k=k)
            self.ctrlpt_function = lambda t_: numpy.array(splev(t_, tck[0])).T

    def set_physical_clearance(self, clearance):
        self.physical_clearance = clearance

    def set_collision_vessels(self, collision_vessels):
        self.collision_vessels = collision_vessels

    def solve(self, *args, sample_size=20, **kwargs):
        self.history_x = []
        self.history_f = []
        def callback(x, f, accept):
            self.history_x.append(x)
            self.history_f.append(f)
            if f == 0:
                return True
        length_bounds = kwargs.get('length_bounds', [0.5, 1.05])
        angle_threshold = kwargs.get('angle_threshold', 70.0)
        if len(args) == 0:
            number_free_points = 1
        else:
            number_free_points = args[0]
        #if isinstance(self.ctrlpt_function, type(None)):
        #    path_pts = numpy.vstack((self.distal_0, self.distal_1))
        #    k = 1
        #    tck = splprep(path_pts.T, s=0, k=k)
        #    self.ctrlpt_function = lambda t_: numpy.array(splev(t_, tck[0])).T
        u_fine = numpy.linspace(0, 1, 25)
        fine_pts = self.ctrlpt_function(u_fine)
        length_threshold = numpy.sum(numpy.linalg.norm(numpy.diff(fine_pts, axis=0), axis=1))
        init_spline_length = deepcopy(length_threshold)
        if self.clamp_first:
            length_threshold += numpy.linalg.norm(self.distal_0-self.proximal_0)
        if self.clamp_second:
            length_threshold += numpy.linalg.norm(self.distal_1-self.proximal_1)
        self.curve_function = spline_factory(self.proximal_0, self.distal_0, self.proximal_1, self.distal_1,
                                             clamp_first=self.clamp_first, clamp_second=self.clamp_second,
                                             ctlpts_func=self.ctrlpt_function)
        cost_function = lambda data: cost_new(data, curve_generator=self.curve_function, r=self.radius_0,
                                              collision_vessels=self.collision_vessels, clearance=self.physical_clearance,
                                              bounds=self.bounds, sample_size=sample_size, boundary_func=self.boundary_func,
                                              length_threshold=length_threshold, length_bounds=length_bounds, start=self.proximal_0,
                                              end=self.proximal_1, angle_threshold=angle_threshold, clamp_first=self.clamp_first,
                                              clamp_second=self.clamp_second)
        if self.clamp_first and self.clamp_second:
            xopt = numpy.zeros(4+number_free_points*3)
            xopt[0] = 0.5
            xopt[1] = 1.0
            xopt[-1] = 0.5
            xopt[-2] = 1.0
        elif self.clamp_first or self.clamp_second:
            xopt = numpy.zeros(2+number_free_points*3)
            if self.clamp_first:
                xopt[0] = 0.5
                xopt[1] = 1.0
            else:
                xopt[-1] = 0.5
                xopt[-2] = 1.0
        else:
            xopt = numpy.zeros(number_free_points*3)
        shortest_length = numpy.linalg.norm(self.distal_0-self.distal_1)
        #lb = numpy.ones(xopt.shape[0]) * (-shortest_length)
        #ub = numpy.ones(xopt.shape[0]) * (shortest_length)
        lb = [None] * xopt.shape[0]
        ub = [None] * xopt.shape[0]
        if self.clamp_first:
            lb[0] = 0
            lb[1] = 0
            ub[0] = 1
        if self.clamp_second:
            lb[-1] = 0
            lb[-2] = 0
            ub[-1] = 1
        solver_bounds = []
        for b in range(len(lb)):
            solver_bounds.append([lb[b], ub[b]])
        res = self.solver(cost_function, xopt, niter=500, callback=callback, stepsize=init_spline_length/(number_free_points+1),
                          minimizer_kwargs={'bounds': solver_bounds, 'method': 'L-BFGS-B', 'tol': 1e-5})
        self.xopt = res.x
        self.best = res.fun
        self.curve = self.curve_function(self.xopt)
        self.curve.sample_size = sample_size
        self.curve.evaluate()
        #res.elapsed_time = perf_counter() - start
        return res

    def build_vessels(self, seperate, build_meshes=True):
        pts = numpy.array(self.curve.evalpts)
        #pts = self.curve.evalpts
        if seperate:
            sep = (pts.shape[0]-1)//2
            vessels_1 = numpy.zeros((sep,7))
            vessels_2 = numpy.zeros((pts.shape[0]-1-sep,7))
            vessels_1_pts = pts[:sep+1,:]
            vessels_2_pts = numpy.flip(pts[sep:,:],axis=0)
            vessels_1[:,0:3] = vessels_1_pts[:-1,:]
            vessels_1[:,3:6] = vessels_1_pts[1:,:]
            vessels_2[:,0:3] = vessels_2_pts[:-1,:]
            vessels_2[:,3:6] = vessels_2_pts[1:,:]
            vessels_1[:,6] = self.radius_0
            vessels_2[:,6] = self.radius_1
        else:
            vessels_1 = numpy.zeros((pts.shape[0]-1,7))
            vessels_1[:,0:3] = pts[:-1,:]
            vessels_1[:,3:6] = pts[1:,:]
            vessels_1[:,6] = self.radius_0
            pts_reverse = numpy.flip(pts,axis=0)
            vessels_2 = numpy.zeros((pts.shape[0]-1,7))
            vessels_2[:,0:3] = pts_reverse[:-1,:]
            vessels_2[:,3:6] = pts_reverse[1:,:]
            vessels_2[:,6] = self.radius_1
        self.vessels_1 = vessels_1
        self.vessels_2 = vessels_2
        self.seperate = seperate
        self.vessel_1_meshes = []
        if build_meshes:
            for i in range(vessels_1.shape[0]):
                center = (self.vessels_1[i, 0:3] + self.vessels_1[i, 3:6]) / 2
                direction = self.vessels_1[i, 3:6] - self.vessels_1[i, 0:3]
                length = numpy.linalg.norm(direction)
                direction = direction / length
                radius = self.vessels_1[i, 6]
                cylinder = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
                self.vessel_1_meshes.append(cylinder)
            self.vessel_2_meshes = []
            for i in range(self.vessels_2.shape[0]):
                center = (self.vessels_2[i, 0:3] + self.vessels_2[i, 3:6]) / 2
                direction = self.vessels_2[i, 3:6] - self.vessels_2[i, 0:3]
                length = numpy.linalg.norm(direction)
                direction = direction / length
                radius = self.vessels_2[i, 6]
                cylinder = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
                self.vessel_2_meshes.append(cylinder)
        return vessels_1, vessels_2

    def show(self):
        plotter = pv.Plotter()
        if self.seperate:
            for i in range(self.vessels_1.shape[0]):
                center = (self.vessels_1[i,0:3]+self.vessels_1[i,3:6])/2
                direction = self.vessels_1[i,3:6] - self.vessels_1[i,0:3]
                length = numpy.linalg.norm(direction)
                direction = direction/length
                radius = self.vessels_1[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='red')
            for i in range(self.vessels_2.shape[0]):
                center = (self.vessels_2[i,0:3]+self.vessels_2[i,3:6])/2
                direction = self.vessels_2[i,3:6] - self.vessels_2[i,0:3]
                length = numpy.linalg.norm(direction)
                direction = direction/length
                radius = self.vessels_2[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='blue')
        else:
            for i in range(self.vessels_1.shape[0]):
                center = (self.vessels_1[i,0:3]+self.vessels_1[i,3:6])/2
                direction = self.vessels_1[i,3:6] - self.vessels_1[i,0:3]
                length = numpy.linalg.norm(direction)
                direction = direction/length
                radius = self.vessels_1[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='red')
        if self.plotting_vessels is not None:
            for i in range(self.plotting_vessels.shape[0]):
                center = (self.plotting_vessels[i,0:3]+self.plotting_vessels[i,3:6])/2
                direction = self.plotting_vessels[i,3:6] - self.plotting_vessels[i,0:3]
                length = numpy.linalg.norm(direction)
                direction = direction/length
                radius = self.plotting_vessels[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='black')
        if np.all(self.proximal_0 != self.distal_0):
            center = (self.proximal_0+self.distal_0)/2
            direction = (self.distal_0-self.proximal_0)
            length = numpy.linalg.norm(direction)
            direction = direction/length
            radius = self.radius_0
            cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
            plotter.add_mesh(cylinder,color='yellow')
        if np.all(self.proximal_1 != self.distal_1):
            center = (self.proximal_1+self.distal_1)/2
            direction = (self.distal_1-self.proximal_1)
            length = numpy.linalg.norm(direction)
            direction = direction/length
            radius = self.radius_1
            cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
            plotter.add_mesh(cylinder,color='yellow')
        return plotter

    def show_history(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0.1, bottom=0.25)
        line, = ax.plot([], [], [], 'b-', label='Connection History')
        spline = self.curve_function(self.history_x[0])
        spline.sample_size = 20
        spline.evaluate()
        his = spline.evalpts
        for x in self.history_x[1:]:
            spline = self.curve_function(x)
            spline.sample_size = 20
            spline.evaluate()
            his = np.vstack((his, spline.evalpts))
        x_min = np.min(his[:, 0])
        x_max = np.max(his[:, 0])
        y_min = np.min(his[:, 1])
        y_max = np.max(his[:, 1])
        z_min = np.min(his[:, 2])
        z_max = np.max(his[:, 2])

        def update_plot(idx):
            spline = self.curve_function(self.history_x[idx])
            spline.sample_size = 20
            spline.evaluate()
            upstream_1 = numpy.vstack([self.proximal_0, spline.evalpts[0, :]])
            upstream_2 = numpy.vstack([spline.evalpts[-1, :], self.proximal_1])
            ax.clear()
            ax.plot(spline.ctrlpts[:, 0], spline.ctrlpts[:, 1], spline.ctrlpts[:, 2], 'yo--', label='Control Points')
            ax.plot(spline.evalpts[:, 0], spline.evalpts[:, 1], spline.evalpts[:, 2], 'g-', label='Connection')
            ax.plot(upstream_1[:, 0], upstream_1[:, 1], upstream_1[:, 2], 'r-', label='Tree 1')
            ax.plot(upstream_2[:, 0], upstream_2[:, 1], upstream_2[:, 2], 'b-', label='Tree 2')
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            ax.legend()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            if (self.history_f[idx] // 7) == 1:
                collision = True
                bounds = True
                curvature = True
            elif (self.history_f[idx] // 6) == 1:
                collision = True
                bounds = True
                curvature = False
            elif (self.history_f[idx] // 5) == 1:
                collision = False
                bounds = True
                curvature = True
            elif (self.history_f[idx] // 4) == 1:
                collision = False
                bounds = True
                curvature = False
            elif (self.history_f[idx] // 3) == 1:
                collision = True
                bounds = False
                curvature = True
            elif (self.history_f[idx] // 2) == 1:
                collision = True
                bounds = False
                curvature = False
            elif (self.history_f[idx] // 1) == 1:
                collision = False
                bounds = False
                curvature = True
            elif self.history_f[idx] < 1:
                collision = False
                bounds = False
                curvature = False
            else:
                raise ValueError("Invalid cost function")
            title = 'Connection History at Iteration {}, Cost: {} \n Collision: {} ,Boundary: {}, Curvature: {}'.format(idx, self.history_f[idx], collision, bounds, curvature)
            ax.set_title(title)
            fig.canvas.draw_idle()

        ax_slider = fig.add_axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Control Set', 0, len(self.history_x) - 1, valinit=0, valfmt='%0.0f')
        def update(val):
            idx = int(slider.val)
            update_plot(idx)

        slider.on_changed(update)
        update_plot(0)
        plt.show()

class SimpleConnection(object):
    def __init__(self, forest, network_id, tree_0, tree_1, idx, jdx, ctrl_function=None, clamp_first=True,
                 clamp_second=True, point_0=None, point_1=None):
        self.forest = forest
        conn = Connection()
        vessel_0 = forest.networks[network_id][tree_0].data[idx, :]
        vessel_1 = forest.networks[network_id][tree_1].data[jdx, :]
        if not isinstance(point_0, type(None)):
            conn.set_first_vessel(point_0[0:3], point_0[3:6], vessel_0[21])
        else:
            conn.set_first_vessel(vessel_0[0:3], vessel_0[3:6], vessel_0[21])
        if not isinstance(point_1, type(None)):
            conn.set_second_vessel(point_1[0:3], point_1[0:3], vessel_1[21])
        else:
            conn.set_second_vessel(vessel_1[0:3], vessel_1[3:6], vessel_1[21])
        collision_vessels = []
        tree_0_idx = numpy.arange(forest.networks[network_id][tree_0].data.shape[0], dtype=int).tolist()
        if not numpy.isnan(vessel_0[17]):
            parent = int(vessel_0[17])
            daughter_0 = int(forest.networks[network_id][tree_0].data[parent, 15])
            daughter_1 = int(forest.networks[network_id][tree_0].data[parent, 16])
            tree_0_idx.remove(parent)
            tree_0_idx.remove(daughter_0)
            tree_0_idx.remove(daughter_1)
        if not numpy.isnan(vessel_0[15]):
            daughter_2 = int(vessel_1[15])
            tree_0_idx.remove(daughter_2)
        if not numpy.isnan(vessel_0[16]):
            daughter_3 = int(vessel_1[16])
            tree_0_idx.remove(daughter_3)
        tree_0_idx = numpy.array(tree_0_idx).astype(int)
        tmp = numpy.zeros((tree_0_idx.shape[0], 7))
        tmp[:, 0:3] = forest.networks[network_id][tree_0].data[tree_0_idx, 0:3]
        tmp[:, 3:6] = forest.networks[network_id][tree_0].data[tree_0_idx, 3:6]
        tmp[:, 6] = forest.networks[network_id][tree_0].data[tree_0_idx, 21]
        collision_vessels.append(tmp)
        tree_1_idx = numpy.arange(forest.networks[network_id][tree_1].data.shape[0], dtype=int).tolist()
        if not numpy.isnan(vessel_1[17]):
            parent = int(vessel_1[17])
            daughter_0 = int(forest.networks[network_id][tree_1].data[parent, 15])
            daughter_1 = int(forest.networks[network_id][tree_1].data[parent, 16])
            tree_1_idx.remove(parent)
            tree_1_idx.remove(daughter_0)
            tree_1_idx.remove(daughter_1)
        if not numpy.isnan(vessel_1[15]):
            daughter_2 = int(vessel_1[15])
            tree_1_idx.remove(daughter_2)
        if not numpy.isnan(vessel_1[16]):
            daughter_3 = int(vessel_1[16])
            tree_1_idx.remove(daughter_3)
        tree_1_idx = numpy.array(tree_1_idx).astype(int)
        tmp = numpy.zeros((tree_1_idx.shape[0], 7))
        tmp[:, 0:3] = forest.networks[network_id][tree_1].data[tree_1_idx, 0:3]
        tmp[:, 3:6] = forest.networks[network_id][tree_1].data[tree_1_idx, 3:6]
        tmp[:, 6] = forest.networks[network_id][tree_1].data[tree_1_idx, 21]
        collision_vessels.append(tmp)
        for i in range(forest.n_networks):
            for j in range(forest.n_trees_per_network[i]):
                if i == network_id and (j == tree_0 or j == tree_1):
                    continue
                tmp = numpy.zeros((forest.networks[i][j].data.shape[0], 7))
                tmp[:, 0:3] = forest.networks[i][j].data[:, 0:3]
                tmp[:, 3:6] = forest.networks[i][j].data[:, 3:6]
                tmp[:, 6] = forest.networks[i][j].data[:, 21]
                collision_vessels.append(tmp)
        collision_vessels = numpy.vstack(collision_vessels)
        if collision_vessels.shape[0] > 0:
            conn.set_collision_vessels(collision_vessels)
        conn.set_solver()
        conn.set_physical_clearance(self.forest.physical_clearance)
        bounds = numpy.zeros((3, 2))
        bounds[:, 0] = numpy.min(forest.domain.points, axis=0).T
        bounds[:, 1] = numpy.max(forest.domain.points, axis=0).T
        bounds_func = forest.domain.__call__
        conn.set_bounds(bounds, boundary_func=bounds_func)
        if not isinstance(ctrl_function, type(None)):
            conn.set_ctrl_function(ctrl_function)
        conn.clamp_first = clamp_first
        conn.clamp_second = clamp_second
        self.connection = conn

    def solve(self, *args, **kwargs):
        return self.connection.solve(*args, **kwargs)

    def show(self):
        plotter = self.forest.show(plot_domain=True, return_plotter=True)
        self.connection.build_vessels(True)
        for mesh in self.connection.vessel_1_meshes:
            plotter.add_mesh(mesh, color='red')
        for mesh in self.connection.vessel_2_meshes:
            plotter.add_mesh(mesh, color='blue')
        return plotter

    def show_init(self):
        plotter = self.forest.show(plot_domain=True, return_plotter=True)
        if not isinstance(self.connection.xopt, type(None)):
            xopt = self.connection.xopt.copy()
        self.connection.xopt = np.zeros_like(xopt)
        self.connection.xopt[0] = 0.5
        self.connection.xopt[1] = 1.0
        self.connection.xopt[-1] = 0.5
        self.connection.xopt[-2] = 1.0
        original_curve = deepcopy(self.connection.curve)
        original_sample_size = deepcopy(original_curve.sample_size)
        self.connection.curve = self.connection.curve_function(self.connection.xopt)
        self.connection.curve.sample_size = original_sample_size
        self.connection.curve.evaluate()
        self.connection.build_vessels(True)
        for mesh in self.connection.vessel_1_meshes:
            plotter.add_mesh(mesh, color='red')
        for mesh in self.connection.vessel_2_meshes:
            plotter.add_mesh(mesh, color='blue')
        self.connection.xopt = xopt
        self.connection.curve = original_curve
        self.connection.curve.sample_size = original_sample_size
        self.connection.curve.evaluate()
        self.connection.build_vessels(True)
        return plotter

class TreeConnections(object):
    def __init__(self, forest, network_id):
        self.forest = forest
        self.network_id = network_id
        self.assignments, self.connections = assign_network(forest, network_id)
        self.vessel_connections = []
        self.connection_collisions = []
        self.connecting_vessels = []
        self.network = []
        for i in range(forest.n_trees_per_network[network_id]):
            self.network.append(deepcopy(forest.networks[network_id][i].data))

    def solve(self, *args, **kwargs):
        self.vessel_connections = []
        self.connection_collisions = []
        self.connecting_vessels = []
        tree_0 = 0
        tree_1 = 1
        tree_connections = []
        midpoints = []
        for j in trange(len(self.connections[0]), desc=f"Tree {tree_0} to Tree {tree_1}", leave=False):
            self.connecting_vessels.append([])
            self.connecting_vessels.append([])
            conn = SimpleConnection(self.forest, self.network_id, tree_0, tree_1,
                                    self.assignments[tree_0][j], self.assignments[tree_1][j],
                                    ctrl_function=self.connections[0][j],
                                    clamp_first=True, clamp_second=True)
            if len(self.connection_collisions) > 0:
                collisions = numpy.vstack(self.connection_collisions)
                if not isinstance(conn.connection.collision_vessels, type(None)):
                    collisions = numpy.vstack((collisions, conn.connection.collision_vessels))
                conn.connection.set_collision_vessels(collisions)
            conn.solve(*args, **kwargs)
            conn.connection.build_vessels(True, build_meshes=False)
            self.connection_collisions.append(conn.connection.vessels_1)
            self.connection_collisions.append(conn.connection.vessels_2)
            self.connecting_vessels[tree_0].append(conn.connection.vessels_1)
            self.connecting_vessels[tree_1].append(conn.connection.vessels_2)
            tree_connections.append(conn)
            midpoints.append(conn.connection.vessels_1[-1, 3:6])
        midpoints = numpy.array(midpoints)
        self.vessel_connections.append(tree_connections)
        if self.forest.n_trees_per_network[self.network_id] > 2:
            remaining_assignments, remaining_connections = assign_network_vector(self.forest, self.network_id, midpoints)
            self.assignments.extend(remaining_assignments)
            self.connections.extend(remaining_connections)
            for n in range(2, self.forest.n_trees_per_network[self.network_id]):
                tree_0 = 0
                tree_n = n
                tree_connections = []
                self.connecting_vessels.append([])
                for j in trange(len(remaining_connections[n-2]), desc=f"Tree {tree_0} to Tree {tree_n}", leave=False):
                    conn = SimpleConnection(self.forest, self.network_id, tree_n, tree_0,
                                            remaining_assignments[n-2][j], self.assignments[tree_0][j],
                                            ctrl_function=remaining_connections[n-2][j],
                                            clamp_first=True, clamp_second=False, point_1=midpoints[j])
                    #if len(self.connection_collisions) > 0:
                    #    collisions = numpy.vstack(self.connection_collisions)
                    #    if not isinstance(conn.connection.collision_vessels, type(None)):
                    #        collisions = numpy.vstack((collisions, conn.connection.collision_vessels))
                    #    conn.connection.set_collision_vessels(collisions)
                    collisions = []
                    for i in range(len(self.connecting_vessels)):
                        for c in range(len(self.connecting_vessels[i])):
                            if c == j:
                                collisions.append(self.connecting_vessels[i][c][:-1, :])
                            else:
                                collisions.append(self.connecting_vessels[i][c])
                    collisions = numpy.vstack(collisions)
                    #conn.connection.set_collision_vessels(collisions)
                    conn.solve(*args, **kwargs)
                    conn.connection.build_vessels(False, build_meshes=False)
                    self.connection_collisions.append(conn.connection.vessels_1)
                    self.connecting_vessels[tree_n].append(conn.connection.vessels_1)
                    tree_connections.append(conn)
                self.vessel_connections.append(tree_connections)
        """
        for i in range(len(self.connections)):
            tree_0 = 0
            tree_1 = i + 1
            if i == 0:
                # Account for the split bewteen "arterial/venous" of first connection
                self.connecting_vessels.append([])
                self.connecting_vessels.append([])
            else:
                self.connecting_vessels.append([])
            for j in trange(len(self.connections[0]), desc=f"Tree {tree_0} to Tree {tree_1}"):
                if i == 0:
                    conn = SimpleConnection(self.forest, self.network_id, tree_0, tree_1,
                                            self.assignments[tree_0][j], self.assignments[tree_1][j])
                else:
                    conn = SimpleConnection(self.forest, self.network_id, tree_0, tree_1,
                                            self.assignments[tree_0][j], self.assignments[tree_1][j],
                                            ctrl_function=self.connections[i][j], clamp_first=True,
                                            clamp_second=False)
                if len(self.connection_collisions) > 0:
                    collisions = numpy.vstack(self.connection_collisions)
                    if not isinstance(conn.connection.collision_vessels, type(None)):
                        collisions = numpy.vstack((collisions, conn.connection.collision_vessels))
                    conn.connection.set_collision_vessels(collisions)
                conn.solve(*args, **kwargs)
                if i == 0:
                    conn.connection.build_vessels(True, build_meshes=False)
                    self.connection_collisions.append(conn.connection.vessels_1)
                    self.connection_collisions.append(conn.connection.vessels_2)
                    self.connecting_vessels[tree_0].append(conn.connection.vessels_1)
                    self.connecting_vessels[tree_1].append(conn.connection.vessels_2)
                else:
                    conn.connection.build_vessels(False, build_meshes=False)
                    self.connection_collisions.append(conn.connection.vessels_2)
                    self.connecting_vessels[tree_1].append(conn.connection.vessels_2)
                self.vessel_connections.append(conn)
        """

    def show(self, **kwargs):
        colors = kwargs.get('colors', ['red', 'blue', 'green', 'yellow', 'purple',
                                       'orange', 'cyan', 'magenta', 'white', 'black'])
        plotter = pv.Plotter(**kwargs)
        count = 0
        for i in range(len(self.network)):
            for j in range(self.network[i].shape[0]):
                if j in self.assignments[i]:
                    conn_id = self.assignments[i].index(j)
                    center = (self.network[i][j, 0:3] + self.connecting_vessels[i][conn_id][0, 0:3]) / 2
                    length = numpy.linalg.norm(self.connecting_vessels[i][conn_id][0, 0:3] - self.network[i][j, 0:3])
                    direction = (self.connecting_vessels[i][conn_id][0, 0:3] - self.network[i][j, 0:3]) / length
                    radius = self.network[i][j, 21]
                    vessel = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
                else:
                    center = (self.network[i][j, 0:3] + self.network[i][j, 3:6]) / 2
                    direction = self.network[i].get('w_basis', j)
                    radius = self.network[i].get('radius', j)
                    length = self.network[i].get('length', j)
                    vessel = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
                plotter.add_mesh(vessel, color=colors[count % len(colors)], opacity=0.25)
            count += 1
        count = 0
        for conn in range(len(self.connecting_vessels)):
            for i in range(len(self.connecting_vessels[conn])):
                for j in range(self.connecting_vessels[conn][i].shape[0]):
                    center = (self.connecting_vessels[conn][i][j, 0:3] + self.connecting_vessels[conn][i][j, 3:6]) / 2
                    direction = self.connecting_vessels[conn][i][j, 3:6] - self.connecting_vessels[conn][i][j, 0:3]
                    length = numpy.linalg.norm(direction)
                    direction = direction / length
                    radius = self.connecting_vessels[conn][i][j, 6]
                    cylinder = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
                    plotter.add_mesh(cylinder, color=colors[count % len(colors)])
            count += 1
        plotter.add_mesh(self.forest.domain.boundary, color='grey', opacity=0.15)
        return plotter


class ForestConnections(object):
    def __init__(self, forest):
        self.forest = forest
        self.network_connections = []
        for i in range(forest.n_networks):
            self.network_connections.append(TreeConnections(forest, i))

    def solve(self, network_id, *args, **kwargs):
        for i in range(len(self.network_connections)):
            self.network_connections[i].solve(*args, **kwargs)

    def show(self, **kwargs):
        plotter = pv.Plotter(**kwargs)
        for i in range(len(self.network_connections)):
            self.network_connections[i].show(plotter=plotter)
        return plotter