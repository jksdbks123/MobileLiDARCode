from scipy.optimize import linear_sum_assignment
# Import necessary libraries for proessing .las data
import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
import os
import geopandas as gpd
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
# Load image
from skimage import filters, feature
import cv2
from shapely.geometry import LineString
from tqdm import tqdm
from DDBSCAN import Raster_DBSCAN

def shift_without_wrap(array, shift):
    dx, dy = shift
    result = array.copy()

    if dx > 0:
        result[dx:, :] = array[:-dx, :]
        result[:dx, :] = 0
    elif dx < 0:
        result[:dx, :] = array[-dx:, :]
        result[dx:, :] = 0

    result_temp = result.copy()

    if dy > 0:
        result_temp[:, dy:] = result[:, :-dy]
        result_temp[:, :dy] = 0
    elif dy < 0:
        result_temp[:, :dy] = result[:, -dy:]
        result_temp[:, dy:] = 0

    return result_temp

def get_link_list(Set_C,dis_C,C,cos_thresh):
    # Set_C list of spline points
    # dis_C list of distances to the midpoint

    linked = []
    for cur_ind in range(len(Set_C)-1):
        A = compute_affinity(Set_C[cur_ind],Set_C[cur_ind+1],dis_C[cur_ind],dis_C[cur_ind+1],C[cur_ind],cos_thresh)
        row_idx, col_idx = linear_sum_assignment(A)
        row_idx_,col_idx_ = [],[]
        for r,c in zip(row_idx,col_idx):
            if A[r,c] > 1:
                continue
            row_idx_.append(r)
            col_idx_.append(c)
        # Store both indices to capture the linkage
        links = list(zip(row_idx_, col_idx_))
        linked.append(links)
    return linked

def compute_affinity(point_set_cur, point_set_next,dis_set_cur,dis_set_next, C_vec, thresh):
    n = len(point_set_cur)
    m = len(point_set_next)
    A = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            # Metric 1: Difference in distance to sliding start point
            d_diff = np.abs(dis_set_cur[i] - dis_set_next[j]) 
            
            # Metric 2: Cosine similarity with C
            vector = point_set_next[j] - point_set_cur[i]
            cos_sim = np.sum(C_vec*vector)/(np.sqrt(np.sum(vector**2))*np.sqrt(np.sum(C_vec**2)))
            # If cosine similarity is below threshold, set the distance to infinity
            if cos_sim < thresh:
                d_diff = 10000
            A[i, j] = d_diff
    return A

def get_seperated_spline_poly(sets,linked,len_thred):
    splines = []  # List to store individual splines
    # Traverse the linked list
    for set_idx in range(len(sets)):
        # Check existing splines if they can be extended with the current set
        for spline in splines:
            last_point = spline[-1]
            next_point = [link[1] for link in linked[last_point[0]] if link[0] == last_point[1]]
            if next_point:
                spline.append((set_idx, next_point[0]))
        
        # Check if any point in the current set can initiate a new spline
        for idx in range(len(sets[set_idx])):
            point = (set_idx, idx)
            if not any(point in spline for spline in splines):
                splines.append([point])
    spline_polys = []
    for spline_inds in splines:
        spline_poly = []
        for spline_ind in spline_inds:
            spline_poly.append(sets[spline_ind[0]][spline_ind[1]])
        spline_polys.append(spline_poly)
    spline_polys_ = [s for s in spline_polys if len(s) > len_thred]
    return spline_polys_

def point_on_line(a, b, p):
    ap = p - a
    ab = b - a
    result = ((ap * ab).sum(axis = 1) / np.dot(ab, ab)).reshape(-1,1) * ab + a
    return result



def get_spline_poly(raster_cropped,section_heading_vecs,total_spline_coords,total_dis_splines2intersects_list,min_x,min_y,cos_thresh = 0.5,grid_size = 0.05):
    C = section_heading_vecs
    Set_C = total_spline_coords
    dis_C = total_dis_splines2intersects_list
    linked = get_link_list(Set_C,dis_C,C,cos_thresh)
    spline_polys = get_seperated_spline_poly(Set_C,linked,len_thred=1)
    identified_polys_layer = np.zeros(raster_cropped[0].shape, dtype=np.uint8)
    for i in range(len(spline_polys)):
        for j in range(len(spline_polys[i])-1):
            pt1_meter = spline_polys[i][j]
            pt2_meter = spline_polys[i][j+1]
            # convert to pixel unit
            pt1_pixel = ((pt1_meter[0] - min_x)/grid_size, (pt1_meter[1] - min_y)/grid_size)
            pt2_pixel = ((pt2_meter[0] - min_x)/grid_size, (pt2_meter[1] - min_y)/grid_size)
            # convert to int
            pt1_pixel = (int(pt1_pixel[0]),int(pt1_pixel[1]))
            pt2_pixel = (int(pt2_pixel[0]),int(pt2_pixel[1]))
            cv2.line(identified_polys_layer, pt1_pixel,pt2_pixel, color= i + 1, thickness=10,lineType = cv2.LINE_4)
    # add one dimension to make it a 3D array
    identified_polys_layer = np.expand_dims(identified_polys_layer,axis = 0)
    # add the identified_polys_layer to the raster_cropped
    raster_cropped_ = np.concatenate([raster_cropped,identified_polys_layer],axis = 0)

    return raster_cropped_, spline_polys


def get_conv_map(edges,perp_masks,route_clip_image,sub_section_centerline_points,conv_thred = 50):
    total_conv_map = np.zeros(edges.shape, dtype=np.uint8)
    total_route_image = np.zeros(edges.shape, dtype=np.uint8)
    sub_high_conv_images = []
    sub_route_clip_images = []
    sub_edge_bboxs = []
    sub_route_bboxs = []
    valid_centerline_points = []
    for i,temp_mask in tqdm(enumerate(perp_masks)):
        temp_mask = perp_masks[i]
        res = generate_convmap(edges,route_clip_image,temp_mask,conv_thred)
        if res is not None:
            sub_high_conv_image,sub_route_clip_image,sub_edge_bbox,sub_route_bbox = res
            sub_high_conv_images.append(sub_high_conv_image)
            sub_route_clip_images.append(sub_route_clip_image)
            sub_edge_bboxs.append(sub_edge_bbox)
            sub_route_bboxs.append(sub_route_bbox)
            valid_centerline_points.append(sub_section_centerline_points[i])  
            total_conv_map[sub_edge_bbox[1]:sub_edge_bbox[3], sub_edge_bbox[0]:sub_edge_bbox[2]] += sub_high_conv_image
            total_route_image[sub_route_bbox[1]:sub_route_bbox[3], sub_route_bbox[0]:sub_route_bbox[2]] += sub_route_clip_image.astype(np.uint8)
    return total_conv_map, total_route_image, sub_high_conv_images, sub_route_clip_images, sub_edge_bboxs, sub_route_bboxs, valid_centerline_points



def get_sections(route_clip, intensity_image, step_meter,grid_size,min_x,min_y,max_x,max_y):
    
    # Apply Gaussian smoothing
    smoothed = filters.gaussian(intensity_image, sigma=3)
    # Determine high threshold using Otsu's method
    high_threshold = filters.threshold_otsu(smoothed)
    # Set low threshold to be half the high threshold
    low_threshold = 0.5 * high_threshold
    # Apply Canny edge detection
    edges = feature.canny(smoothed, low_threshold=low_threshold, high_threshold=high_threshold)
    route_clip_coords = np.array(route_clip.get_coordinates())
    route_clip_image = np.zeros(edges.shape, dtype=np.uint8)
    perp_masks = []
    sub_section_centerline_points = [] # terminals of the sub-sections
    # Create a LineString from the route_clip_coords
    route_line = LineString(route_clip_coords)
    # convert step to pixel unit
    step_pixel = step_meter / grid_size
    conv_range = max(edges.shape[0], edges.shape[1]) # range in pixel unit
    interpolated_points_meter = []

    distance = 0
    while distance <= route_line.length:
        point = route_line.interpolate(distance)
        interpolated_points_meter.append((point.x, point.y))
        distance += step_meter
    for i in range(1,len(interpolated_points_meter)):
        pt1_meter = interpolated_points_meter[i - 1]
        pt2_meter = interpolated_points_meter[i]
    # convert to pixel unit
        pt1_pixel = ((pt1_meter[0] - min_x)/grid_size, (pt1_meter[1] - min_y)/grid_size)
        pt2_pixel = ((pt2_meter[0] - min_x)/grid_size, (pt2_meter[1] - min_y)/grid_size)
        cv2.line(route_clip_image, (int(pt1_pixel[0]), int(pt1_pixel[1])), (int(pt2_pixel[0]), int(pt2_pixel[1])), color=1, thickness=3,lineType = cv2.LINE_4)
    # Calculate the slope of the line segment
        if pt2_pixel[0] == pt1_pixel[0]:  # Avoid division by zero
            slope = np.inf
        else:
            slope = (pt2_pixel[1] - pt1_pixel[1]) / (pt2_pixel[0] - pt1_pixel[0])
    # Calculate the slope of the perpendicular line
        if slope == 0:  # Avoid division by zero
            perp_slope = np.inf
        else:
            perp_slope = -1 / slope

        mid_pt_pixel = ((pt1_pixel[0] + pt2_pixel[0]) // 2, (pt1_pixel[1] + pt2_pixel[1]) // 2)
        sub_section_centerline_points.append((pt1_pixel, pt2_pixel))
        perp_pt1_pixel = (int(mid_pt_pixel[0] - conv_range), int(mid_pt_pixel[1] - conv_range * perp_slope))
        perp_pt2_pixel = (int(mid_pt_pixel[0] + conv_range), int(mid_pt_pixel[1] + conv_range * perp_slope))

        mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.line(mask, perp_pt1_pixel, perp_pt2_pixel, color=1, thickness=int(step_pixel) + 5,lineType = cv2.LINE_4)
        perp_masks.append(mask)
    # n x 2 x 2
    sub_section_centerline_points = np.array(sub_section_centerline_points)

    return sub_section_centerline_points,perp_masks,interpolated_points_meter,edges,route_clip_image


# make a copy of the edge image
def generate_convmap(edge_image,route_clip_image,temp_mask,conv_thred):
    edges_flipped_copy = edge_image.copy()
    route_clip_image_copy = route_clip_image.copy()
# set the edge image to 0 where the mask is 0
    edges_flipped_copy[temp_mask==0] = 0
    edges_flipped_copy = edges_flipped_copy.astype(np.uint8)
# do same thing for roue_clip_image
    route_clip_image_copy[temp_mask==0] = 0

# get bounding box of the non-zero part of route_clip_image_copy
    non_zero_indices = np.nonzero(route_clip_image_copy)
    min_x = np.min(non_zero_indices[1])
    max_x = np.max(non_zero_indices[1])
    min_y = np.min(non_zero_indices[0])
    max_y = np.max(non_zero_indices[0])
    route_bbox = [min_x, min_y, max_x, max_y]
    route_clip_image_zoom = route_clip_image_copy[min_y:max_y, min_x:max_x]
    non_zero_indices = np.nonzero(edges_flipped_copy)
    # if non_zero_indices is empty, return a empty image
    if len(non_zero_indices[0]) == 0:
        return None
    min_x = np.min(non_zero_indices[1])
    max_x = np.max(non_zero_indices[1])
    min_y = np.min(non_zero_indices[0])
    max_y = np.max(non_zero_indices[0])
    if (max_y - min_y == 0) or (max_x - min_x == 0):
        return None
    edge_bbox = [min_x, min_y, max_x, max_y]
    edges_flipped_iamge_zoom = edges_flipped_copy[min_y:max_y, min_x:max_x]
# do convolution between edge_flipped_zoom and route_clip_image_zoom
    conv_map = cv2.filter2D(edges_flipped_iamge_zoom, -1, route_clip_image_zoom)
    high_conv_map = (conv_map>conv_thred).astype(np.uint8)
    # generate centerline from high conv area

    return high_conv_map,route_clip_image_zoom,edge_bbox,route_bbox

def get_spline_coords(sub_high_conv_images,sub_edge_bboxs,valid_centerline_points,grid_size,min_x,min_y,max_x,max_y,band_width = 0.5,peak_height = 0.1,peak_interval = 10):
    total_spline_coords = []
    total_dis_splines2intersects_list = []
    section_heading_vecs = []
    auxilary_proj_dis_list = []
    TotalCoords_meter_list = []
    mid_pt_list = []
    for i in tqdm(range(len(valid_centerline_points))):
        # convert the coordinates of the valid_centerline_points to meter coordinate system
        pt1_pixel,pt2_pixel = valid_centerline_points[i]
        pt1_pixel,pt2_pixel = np.array(pt1_pixel),np.array(pt2_pixel)
        pt1_meter = np.array([pt1_pixel[0] * grid_size + min_x,pt1_pixel[1] * grid_size + min_y])
        pt2_meter = np.array([pt2_pixel[0] * grid_size + min_x,pt2_pixel[1] * grid_size + min_y])
        # calculate unit vector of the line segment
        unit_vec = pt2_meter - pt1_meter
        heading_vec_unit = unit_vec/np.sqrt(np.sum(unit_vec**2))
        # calculate the perpendicular vector
        perp_vec = np.array([heading_vec_unit[1],-heading_vec_unit[0]])
        vec_flux_line_unit_1 = perp_vec
        # get the coordinates of the high conv area in the sub_high_conv_images
        TotalCoords_pixel = np.argwhere(sub_high_conv_images[i] == 1)
        TotalCoords_pixel[:,1] += sub_edge_bboxs[i][0]
        TotalCoords_pixel[:,0] += sub_edge_bboxs[i][1]
        # convert TotalCoords_pixel to TotalCoords_meter
        TotalCoords_meter = np.zeros(TotalCoords_pixel.shape) 
        TotalCoords_meter[:,0] = TotalCoords_pixel[:,0] * grid_size + min_y
        TotalCoords_meter[:,1] = TotalCoords_pixel[:,1] * grid_size + min_x
        # swap the columns of TotalCoords_meter
        TotalCoords_meter = TotalCoords_meter[:,[1,0]]
        projected_coords_meter = point_on_line(pt1_meter,pt2_meter,TotalCoords_meter) # projection of high conv points to the line segment
        auxilary_proj_vecs = TotalCoords_meter - projected_coords_meter 
        auxilary_proj_vec_mods = np.sqrt(np.sum((auxilary_proj_vecs)**2,axis = 1))
        # exlude 0 according to the auxilary_proj_vec_mods
        if len(auxilary_proj_vec_mods) == 0:
            continue
        auxilary_proj_vecs = auxilary_proj_vecs[auxilary_proj_vec_mods!=0]
        auxilary_proj_vec_mods = auxilary_proj_vec_mods[auxilary_proj_vec_mods!=0]
        
        auxilary_dirs = np.sum((auxilary_proj_vecs * vec_flux_line_unit_1),axis = 1)/auxilary_proj_vec_mods
        auxilary_proj_dis = auxilary_dirs * auxilary_proj_vec_mods
        mid_pt = (pt2_meter + pt1_meter) / 2
        
        maximum_dis = int(auxilary_proj_dis.max())
        minimum_dis = int(auxilary_proj_dis.min())
        if maximum_dis == minimum_dis:
            continue
        
        kde = KernelDensity(kernel='gaussian', bandwidth=band_width).fit(auxilary_proj_dis.reshape(-1,1))
        log_likelihood = kde.score_samples(np.arange(minimum_dis,maximum_dis,0.1).reshape(-1,1))
        density_curve = np.exp(log_likelihood)
        peaks,properties = find_peaks(density_curve,height=peak_height,width=peak_interval)
        road_center_dis2intersect = np.arange(minimum_dis,maximum_dis,0.1)[peaks]
        
        spline_coords = mid_pt + vec_flux_line_unit_1 * road_center_dis2intersect.reshape(-1,1)
        spline_coords = spline_coords[np.argsort(road_center_dis2intersect)]# this is the coordinates of in the sub_high_conv_images[i] that are on the road centerline
        # record the coordinates of the road centerline in the total_conv_map

        total_dis_splines2intersects_list.append(road_center_dis2intersect)
        total_spline_coords.append(spline_coords)
        mid_pt_list.append(mid_pt)
        auxilary_proj_dis_list.append(auxilary_proj_dis)
        TotalCoords_meter_list.append(TotalCoords_meter)
        section_heading_vecs.append(heading_vec_unit)
        
    return total_spline_coords,total_dis_splines2intersects_list,section_heading_vecs,auxilary_proj_dis_list,TotalCoords_meter_list,mid_pt_list
    
def get_raster(las_path,ransac,grid_size = 0.05):
    las = laspy.read(las_path)
    xyzi = np.vstack([las.x, las.y, las.z,las.intensity]).transpose()
# use ransac to get the ground coefficients, check how much is the acutal slope
    ransac.fit(xyzi[:,:2], xyzi[:,2])
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    ground_points = xyzi[inlier_mask]
# Get the bounds of the grid
    min_x, min_y = ground_points.min(axis=0)[:2]
    max_x, max_y = ground_points.max(axis=0)[:2]

# Create a grid of X and Y coordinates
# define grid size, x refers to the columns, y refers to the rows
    x = np.arange(min_x, max_x, grid_size)
    y = np.arange(min_y, max_y, grid_size)
    X, Y = np.meshgrid(x, y)

# Interpolate the elevation and intensity values of the ground points to the grid, using linear interpolation
    elevation = griddata(ground_points[:, :2], ground_points[:, 2], (X, Y), method='nearest', fill_value=np.nan)
    intensity = griddata(ground_points[:, :2], ground_points[:, 3], (X, Y), method='nearest', fill_value=np.nan)

    x = np.arange(min_x, max_x + grid_size, grid_size)
    y = np.arange(min_y, max_y + grid_size, grid_size)
    density, _, _ = np.histogram2d(ground_points[:,1], ground_points[:,0], bins=(y, x))
    raster = np.stack((intensity, elevation ,density))
# crop the all raster (all layers) according to the density values that are larger than 50, and show the cropped raster

    raster_cropped = raster.copy()
    raster_cropped[:, density < 1] = 0

    return raster_cropped, (min_x, max_x, min_y, max_y),las.vlrs[0].string

