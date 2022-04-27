import json
import argparse 

import tqdm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# calculating overlapping fovs
REMAIN_VOXELS = 500

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--transform", help="transforms.json")
parser.add_argument("-m", "--mask", help="selected images", required=False)
parser.add_argument("-r", "--resolution", type=int, default=256)
parser.add_argument("-o", "--output", help="output transformation", default="transforms.json")
args = parser.parse_args()


def gen_cube_points(start, end, res):
    x_ = np.linspace(start, end, res)
    y_ = np.linspace(start, end, res)
    z_ = np.linspace(start, end, res)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='xy')
    x = x.reshape([-1, 1])
    y = y.reshape([-1, 1])
    z = z.reshape([-1, 1])
    xyz = np.concatenate([x, y, z], axis=-1)
    return xyz


def draw_ofov_img(ofov, cams, path, draw_unit_cube=True):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], c='b', marker='x', label="cameras")
    ax.scatter(ofov[:, 0], ofov[:, 1], ofov[:, 2], c='g', marker='o', label="overlapping FOVs")
    
    if draw_unit_cube:

        unit_cube = np.array([
            [0,0,0],
            [0,0,1],
            [0,1,0],
            [1,0,0],
            [1,1,0],
            [0,1,1],
            [1,0,1],
            [1,1,1],
        ])

        ax.scatter(unit_cube[:, 0], unit_cube[:, 1], unit_cube[:, 2], c='r', marker='^', label="Unit Cube")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()
    ax.grid(True)

    plt.savefig(path)


if __name__ == "__main__":

    if args.mask is not None:
        f = open(args.mask, 'r')
        images = [line.strip() for line in f.readlines()]
        f.close()

    with open(args.transform, 'r') as f:
        trans_data = json.load(f)

    K = np.zeros([3, 3])
    K[0][0] = trans_data["fl_x"]
    K[0][2] = trans_data["cx"]
    K[1][1] = trans_data["fl_y"]
    K[1][2] = trans_data["cy"]
    K[2][2] = 1
    
    K = K @ np.eye(3, 4)

    if args.mask is not None:
        Ts = [ frame['transform_matrix'] for frame in trans_data["frames"] if frame["file_path"] in images]
    else:
        Ts = [ frame['transform_matrix'] for frame in trans_data["frames"]]

    a = trans_data["aabb_scale"]
    w, h = trans_data["w"], trans_data["h"]
    res = args.resolution

    xyz = gen_cube_points(-a/2, a/2, res)

    colors = np.ones([res**3, 3]) * 1.0
    homo = np.concatenate([xyz, np.ones([res**3, 1])], axis=-1)
    ofovs = np.zeros([res**3], dtype=np.float32) 

    Cs = []
    for T in tqdm.tqdm(Ts):
        T = np.linalg.inv(T)
        t = T[:-1,-1]
        R = T[:-1,:-1]
        C = - R.T @ t
        Cs.append(C)
        
        proj = (K @ T @ homo.T).T
        loc_fovs = np.ones([res**3], dtype=bool) 
        proj = proj[:, :-1] / proj[:, -1:]
        loc_fovs = np.logical_and(loc_fovs, proj[:, 1]<h)
        loc_fovs = np.logical_and(proj[:, 0]<w, loc_fovs)
        loc_fovs = np.logical_and(loc_fovs, proj[:, 1]>0)
        loc_fovs = np.logical_and(proj[:, 0]>0, loc_fovs)
        
        ofovs = ofovs + loc_fovs.astype(np.float32) / len(Ts)
    
    cameras = np.array(Cs)

    least_percentile = min(99, 100 * (1 - REMAIN_VOXELS / args.resolution ** 3))
    ofovs = ofovs > np.percentile(ofovs, least_percentile)
    ofov_xyzs = xyz[ofovs]

    draw_ofov_img(ofov_xyzs, cameras, "beforeTransform.png")

    
    ofov_scale = 1.0 / max(np.percentile(ofov_xyzs, 99, axis=0) - np.percentile(ofov_xyzs, 1, axis=0))
    ofov_xyzs = ofov_xyzs * ofov_scale

    ofov_center = ofov_xyzs.mean(axis=0)
    ofov_translate =  np.array([0.5, 0.5, 0.5]) - ofov_center
    ofov_xyzs += ofov_translate

    print("Result:", ofov_scale, ofov_translate)

    draw_ofov_img(ofov_xyzs, cameras, "afterTransform.png")

    trans_data["scale"] = ofov_scale
    trans_data["offset"] = ofov_translate.tolist()

    if args.mask is not None:
        trans_data["frames"] = [ frame for frame in trans_data["frames"] if frame["file_path"] in images]

    with open(args.output, 'w+') as f:
        json.dump(trans_data, f, indent=4)