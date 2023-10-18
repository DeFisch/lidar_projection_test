import open3d
import numpy as np
import json
import pickle
import torch
import io
import colorsys
import matplotlib.pyplot as plt
import cv2
import re
import math

data_root_path = "/Users/daniel/Documents/code/python/AutoDriveTasks/data/dataset_Mcity_1_Jun_8_23_KITTI_no_labels/"

pcd_src_path = ""
pcd_bin_path = data_root_path + "velodyne/000074.bin"
img_path = data_root_path + "image_2/000074.jpg"
calib_path = data_root_path + "calib.txt"
calib_json_path = data_root_path + "calib.json"
# gt_label_path = '/home/zhur123/AutoDrive/SUSTechPOINTS/data/dataset_Mar11_1_post/label/000015.json'
# pred_label_path = '/home/zhur123/AutoDrive/Data/inferences/pred_dicts_015_syn_ft.pkl'
score_thresh = 0.1119

GT_COLOR = [0, 1, 0]
PRED_COLOR = [1, 0, 0]

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue  
            key, value = re.split(':| ', line, 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def lidar_pts_to_img(orig_img, pts_velo, img_width, img_height):
    xs = []
    ys = []
    value = []

    with open(calib_json_path, 'r') as f:
        data = json.load(f)
    #data = read_calib_file(calib_path)
    # Extract the intrinsic and extrinsic matrices
    calib = dict()
    calib['ad_transform_mat'] = np.array(data['extrinsic']).reshape(4,4)
    calib['ad_projection_mat'] = np.array(data['intrinsic']).reshape(3,3)
    # calib['ad_transform_mat'] = np.vstack((np.array(data['Tr_velo_to_cam']).reshape(3,4),np.array([0,0,0,1])))
    # calib['ad_projection_mat'] = np.array(data['P2'])[:9].reshape(3,3)
    # calib['distortion'] = np.array(data['distortion']).reshape(5, order='F')
    
    # pts_velo_homogeneous = np.hstack((pts_velo, np.ones((pts_velo.shape[0], 1))))

    # get the 4x4 transformation matrix
    transform_mat = calib['ad_transform_mat']
    
    # Lidar points in camera frame
    pts_cam_frame = transform_mat @ np.transpose(pts_velo)
    # remove reflective values
    pts_cam_frame = np.delete(pts_cam_frame, 3, 0)

    # [x, y, z, w] -> [x, y, w]
    # then project lidar points from camera frame onto image plane
    points_2d = calib['ad_projection_mat'] @ pts_cam_frame

    points_2d = np.transpose(points_2d)
    dist = points_2d[:, 2]
    points_2d = points_2d[:, :2] / points_2d[:, 2:]
    
    # Scale up the points based on the image dimensions
    # points_2d_scaled = np.round(points_2d * [img_width, img_height]).astype(int)


    for i,point in enumerate(points_2d):
        if True:
            x = point[0]
            y = point[1]
            if x >= 0 and y >= 0 and x <= img_width and y <= img_height:
                xs.append(int(x))
                ys.append(int(y))
                value.append(dist[i] * 10)
          

    # show the original image with the lidar points
    plt.imshow(orig_img)
    plt.scatter(xs, ys, s=1, c=value, cmap='autumn', alpha=0.5)
    plt.show()
    
    # # Image plane 2D to Camera frame 3D            
    # inv_projection = np.linalg.inv(np.transpose(calib['ad_projection_mat']))
    # cam_coords = inv_projection @ pts_image_plane
    
    # # Camera frame to lidar sensor frame
    # cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
    # inv_transform = np.linalg.inv(np.transpose(calib['ad_transform_mat']))
    # velo_points = inv_transform @ cam_coords
    # velo_points = velo_points[:3, :]

    # # visualize the pcd    
    # pcd = open3d.geometry.PointCloud()
    # points = np.transpose(velo_points)
    # pcd.points = open3d.utility.Vector3dVector(points)
    # open3d.visualization.draw_geometries([pcd],
    #                                 front=[-0.9945, 0.03873, 0.0970],  
    #                                 lookat=[38.4120, 0.6139, 0.48500],
    #                                 up=[0.095457, -0.0421, 0.99453],
    #                                 zoom=0.33799
    #                                 )

def visualize(pcd_path, calib_path=None, gt_label_path=None, pred_label_path=None):

    # Plot pcd from bin
    points = np.fromfile(pcd_bin_path, dtype=np.float32).reshape(-1,4)

    img = cv2.cvtColor(cv2.imread(img_path)[:,:,:3],cv2.COLOR_BGR2RGB)
    h,w,_ = img.shape
    lidar_pts_to_img(img, points, w, h)

def rotate_pcd(pcd):
    # Save points into np array
    points = np.asarray(pcd.points)

    # Rotate points 90 degrees clockwise by x = y, y = -x
    rotated_npy = np.array([points[:,1], -points[:,0], points[:,2]]).T

    # Save rotated points back to pcd
    rotated_points = open3d.utility.Vector3dVector(rotated_npy)
    pcd.points = rotated_points

    # Visualize rotated point cloud
    # open3d.visualization.draw_geometries([pcd])

def main():    
    visualize(pcd_src_path, calib_path, None, None)
    # visualize(pcd_path)

if __name__ == "__main__":
    main()
