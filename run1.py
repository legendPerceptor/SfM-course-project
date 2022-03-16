from pathlib import Path
import json
import numpy as np
from numpy.linalg import inv, svd, det, norm
import matplotlib.pyplot as plt
from types import SimpleNamespace
from PIL import Image

import cv2

import open3d as o3d
from vis import vis_3d, o3d_pc, draw_camera
from mpl_interactive import Visualizer as TwoViewVis

from sfm import (
    t_and_R_from_essential,
    t_and_R_from_pose_pair,
    essential_from_t_and_R,
    F_from_K_and_E,
    E_from_K_and_F,
    t_and_R_from_essential,
    disambiguate_four_chirality_by_triangulation,
    triangulate,
    normalized_eight_point_algorithm,
    pose_pair_from_t_and_R,
    eight_point_algorithm,
    _throw_outliers,
    bundle_adjustment,
    align_B_to_A
)

from render import (
    compute_intrinsics, compute_extrinsics, as_homogeneous, homogenize
)

DATA_ROOT = Path("./data")

def pnp_calibration(pts_2d, pts_3d):
    # your code here
    n = pts_2d.shape[0]
    A = np.zeros((2 * n, 12))
    for i in range(n):
        x, y = pts_2d[i, 0], pts_2d[i, 1]
        X, Y, Z = pts_3d[i, 0], pts_3d[i, 1], pts_3d[i, 2]
        A[2 * i, :] = [X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x]
        A[2 * i + 1, :] = [0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y]
    U, S, V = np.linalg.svd(A)
    P = V[-1, :].reshape(3, 4)
    P = P / P[2, 3]
    return P


def load_sfm_data():
    root = DATA_ROOT / "sfm"
    poses = np.load(root / "camera_poses.npy")
    visibility = np.load(root / "multiview_visibility.npy")
    pts_3d = np.load(root / "pts_3d.npy")
    with open(root / "camera_intrinsics.json", "r") as f:
        intr = json.load(f)
    img_w, img_h, fov = intr['img_w'], intr['img_h'], intr['vertical_fov']

    data = SimpleNamespace(
        img_w=img_w, img_h=img_h, fov=fov,
        poses=poses, visibility=visibility, pts_3d=pts_3d
    )
    return data


def read_view_image(i):
    fname = DATA_ROOT / "sfm" / f"view_{i}.png"
    img = np.array(Image.open(fname))
    return img


def common_visible_points(data, view_indices):
    """view_indices: a list of view indices e.g. [0, 1, 4]
    """
    total_points = data.visibility.shape[1]
    print("Total points:", total_points)
    print("visibiliy shape:", data.visibility[0].shape)
    print("visibility 0 1:", data.visibility[0][0], data.visibility[0][1])
    mask = np.ones(total_points, dtype=bool) # your code here: use data.visibility
    print("mask shape:", mask.shape)
    for index in view_indices:
        mask = mask & data.visibility[index]
    common_pts = data.pts_3d[mask]
    return common_pts


def project(pts_3d, K, pose):
    P = K @ inv(pose)
    x1 = homogenize(pts_3d @ P.T)
    return x1


class Engine():
    # def __init__(self):
    #     data = load_sfm_data()
    #     self.view1, self.view2 = 1, 2
    #     self.poses = data.poses
    #     pose1 = data.poses[self.view1]
    #     pose2 = data.poses[self.view2]
    #     pts_3d = common_visible_points(data, [self.view1, self.view2])

    #     img_w, img_h = data.img_w, data.img_h
    #     fov = data.fov
    #     K = compute_intrinsics(img_w / img_h, fov, img_h)

    #     self.data = data
    #     self.K = K
    #     self.pose1 = pose1
    #     self.pose2 = pose2
    #     self.pts_3d = pts_3d
    #     self.img_w, self.img_h = img_w, img_h


    def __init__(self, img_w, img_h, fov):
        self.K = np.array([[960, -0.02, img_w / 2, 0], [0, 960, img_h / 2, 0], [0, 0, 1, 0]])

        # self.K = compute_intrinsics(img_w / img_h, fov, img_h)
        print("THE K",self.K)
        self.img_w = img_w
        self.img_h = img_h
        self.fov = fov



    def q2(self):
        pts_3d = self.pts_3d
        K = self.K
        pose1, pose2 = self.pose1, self.pose2
        img_w, img_h = self.img_w, self.img_h

        fname = str(DATA_ROOT / "optimus_prime.obj")
        mesh = o3d.io.read_triangle_mesh(fname, False)

        vis_3d(
            1500, 1500, mesh, o3d_pc(pts_3d),
            draw_camera(K, pose1, img_w, img_h, scale=10),
            draw_camera(K, pose2, img_w, img_h, scale=10),
        )

    def q4(self):
        pts_3d = self.pts_3d
        K = self.K
        pose1, pose2 = self.pose1, self.pose2

        img1 = read_view_image(self.view1)
        img2 = read_view_image(self.view2)

        x1s = project(pts_3d, K, pose1)
        x2s = project(pts_3d, K, pose2)

        t, R = t_and_R_from_pose_pair(pose1, pose2)
        print("t", t)
        print("R", R)
        K = K[:3, :3]
        E = essential_from_t_and_R(t, R)
        F = F_from_K_and_E(K, E)
        print("F", F)
        # now let's test the the epipolary geometry
        on_epi_line = np.einsum("np, pq, nq -> n", x2s, F, x1s)
        assert np.allclose(on_epi_line, 0)  # alright

        # epipole on img2
        U, S, V = svd(F.T)
        c1_on_img2 = V[-1, :]
        c1_on_img2 = homogenize(c1_on_img2)
        assert np.allclose(c1_on_img2.T @ F, 0)

        # epipole on img1
        U, S, V = svd(F)
        c2_on_img1 = V[-1, :]
        c2_on_img1 = homogenize(c2_on_img1)
        assert np.allclose(F @ c2_on_img1, 0)

        print(c1_on_img2)
        print(c2_on_img1)

        vis = TwoViewVis()
        vis.vis(img1, img2, F)

    def filter_pts(self, p1, p2, pts):
        nv1 = inv(p1) @ pts.T
        mask1 = nv1[2, :] < 0
        print("mask1 shape", mask1.shape)
        nv2 = inv(p2) @ pts.T
        mask2 = nv2[2, :] < 0
        print("mask2 shape", mask2.shape)
        mask = np.bitwise_and(mask1, mask2)
        print("mask shape", mask.shape)
        pts_result = pts[mask, :]
        print("pts shape after", pts_result.shape)
        # pts_1 = pts[:, :3]
        # mask3 = (np.abs(pts_1) > 100).any(axis=1)
        # mask = np.bitwise_and(mask, ~mask3)
        return pts, pts_result, mask

    def q10(self):
        
        p1, p2, pts = self.sfm_pipeline(draw_config=True, final_vis=True)
        pts_old, pts, mask = self.filter_pts(p1, p2, pts)

        
        # self.visualize_final_view(all_pts, camera_poses)
        

        all_pts = pts.copy()
        camera_poses = [p1, p2]
        for i in range(2, len(self.poses)-1):
            common_pts_3d = common_visible_points(self.data, [i-1, i, i+1])
            common_pts_1i_i = common_visible_points(self.data, [i-1, i])
            common_pts_i_i1 = common_visible_points(self.data, [i, i+1])
            x2s_prev = project(common_pts_1i_i, self.K, self.poses[i])
            x2s_prev = x2s_prev[mask, :]
            x1s_new = project(common_pts_i_i1, self.K, self.poses[i])
            x2s_new = project(common_pts_i_i1, self.K, self.poses[i+1])
            x2s_pnp_cheat = project(common_pts_3d, self.K, self.poses[i+1])
            x2s_pnp = []
            common_pts_for_pnp = []
            for j in range(x2s_prev.shape[0]):
                for k in range(x1s_new.shape[0]):
                    if np.array_equal(x2s_prev[j, :], x1s_new[k, :]):
                        x2s_pnp.append(x2s_new[k, :])
                        common_pts_for_pnp.append(pts[j])
            
            x2s_pnp = np.array(x2s_pnp)
            print("x2spnp shpa", x2s_pnp.shape)
            common_pts_for_pnp = np.array(common_pts_for_pnp)
            print("common pts for pnp", common_pts_for_pnp.shape)
            # P = pnp_calibration(x2s_pnp_cheat, common_pts_3d)
            P = pnp_calibration(x2s_pnp, common_pts_for_pnp)
            R_t = np.linalg.inv(self.K[:3, :3]) @ P # 3* 4
            R = R_t[:, :3]
            t = R_t[:, 3]
            p1_rec, p2_rec = pose_pair_from_t_and_R(t, R)
            
            x1s = project(common_pts_i_i1, self.K, self.poses[i])
            x2s = project(common_pts_i_i1, self.K, self.poses[i+1])
            p1, p2, pts = self.sfm_addcamera(p1=p2, p2=p2_rec, x1s=x1s, x2s=x2s, P=P)
            camera_poses.append(p2)
            print("inside forloop pts", pts.shape)
            pts_old, pts, mask = self.filter_pts(p1, p2, pts)
            # pts = _throw_outliers(pts)
            print("throw outliers pts", pts.shape)
            all_pts = np.vstack((all_pts, pts))
        
        all_pts = _throw_outliers(all_pts)
        self.visualize_final_view(all_pts, camera_poses)




    def q12(self):
        self.sfm_pipeline(use_noise=True, use_BA=True, final_vis=True, draw_config=True)
    

    def sfm_pipeline_for_project(self, x1s, x2s, x1s_all, x2s_all, F = None, use_BA=False, draw_config=False, final_vis=False):
        K = self.K
        full_K = K
        K = K[:3, :3]
        img_w, img_h = self.img_w, self.img_h
        if F is None:
            F = normalized_eight_point_algorithm(x1s, x2s, img_w, img_h)
        F1, mask = cv2.findFundamentalMat(x1s,x2s,cv2.FM_8POINT)
        F2 = normalized_eight_point_algorithm(x1s, x2s, img_w, img_h)
        print("F:", F)
        print("F1", F1)
        print("F2", F2)
        E = E_from_K_and_F(K, F2)
        print("E:", E)
        four_hypothesis = t_and_R_from_essential(E)
        p1, p2, t, R = disambiguate_four_chirality_by_triangulation(four_hypothesis, x1s, x2s, full_K, draw_config=draw_config)
        P1 = np.dot(full_K, inv(p1))
        P2 = np.dot(full_K, inv(p2))
        pred_pts = triangulate(P1[:3, :], x1s_all, P2[:3, :], x2s_all)
        # filter pts
        print("pred_pts 22222:", pred_pts[:10])
        if use_BA:
            p1, p2, pred_pts = bundle_adjustment(x1s_all, x2s_all, full_K, p1, p2, pred_pts)
        if final_vis:
            red = (1, 0, 0)
            green = (0, 1, 0)
            blue = (0, 0, 1)

            vis_3d(
                1500, 1500,
                # o3d_pc(pts_3d, red),
                o3d_pc(pred_pts, green),
                # draw_camera(K, pose1, img_w, img_h, 10, red),
                # draw_camera(K, pose2, img_w, img_h, 10, red),
                draw_camera(K, p1, img_w, img_h, 10, blue),
                draw_camera(K, p2, img_w, img_h, 10, blue),
            )
        return p1, p2, pred_pts
    
    def sfm_addcamera(self, p1, p2, x1s, x2s, P = None):
        P1 = np.dot(self.K, inv(p1))
        P2 = np.dot(self.K, inv(p2))
        print("sfm P1", P1)
        print("sfm P2", P2)
        print("sfm P", P)
        # print("self.K", self.K)
        # p2 = inv(inv(self.K) @ P2)
        pred_pts = triangulate(P1[:3, :], x1s, P2[:3, :], x2s)
        p1, p2, pred_pts = bundle_adjustment(x1s, x2s, self.K, p1, p2, pred_pts)
        return p1, p2, pred_pts

    def sfm_pipeline(self, use_noise=False, use_BA=False, draw_config=False, final_vis=False):
        pts_3d = self.pts_3d
        K = self.K
        img_w, img_h = self.img_w, self.img_h
        pose1, pose2 = self.pose1, self.pose2

        x1s = project(pts_3d, K, pose1)
        x2s = project(pts_3d, K, pose2)

        if use_noise:
            x1s, x2s = corruption_pipeline(x1s, x2s)

        full_K = K
        K = K[:3, :3]
        
        # your code here;
        # p1, p2, pred_pts ...
        F = normalized_eight_point_algorithm(x1s, x2s, img_w, img_h)
        F1 = cv2.findFundamentalMat(x1s,x2s,cv2.FM_8POINT)
        print("K", K)
        print("cv2 F", F)
        # F = eight_point_algorithm(x1s, x2s)
        print("F", F)
        E = E_from_K_and_F(K, F)
        print("E", E)
        four_hypothesis = t_and_R_from_essential(E)
        # print("four_hypothesis", four_hypothesis)
        p1, p2, t, R = disambiguate_four_chirality_by_triangulation(four_hypothesis, x1s, x2s, full_K, draw_config=draw_config)
        P1 = np.dot(full_K, inv(p1))
        P2 = np.dot(full_K, inv(p2))
        # print("E", E)
        # print("p2", p2)
        # print("p1", p1)
        pred_pts = triangulate(P1[:3, :], x1s, P2[:3, :], x2s)
        # pred_pts = triangulate(np.dot(full_K, p1)[:3, :], x1s, np.dot(full_K, p2)[:3, :], x2s)
        # print("full K", full_K)
        

        if use_BA:
            p1, p2, pred_pts = bundle_adjustment(x1s, x2s, full_K, p1, p2, pred_pts)
        # print("pred_pts.shape", pred_pts.shape)
        # print("pts3d.shape", pts_3d.shape)
        pred_pts, p1, p2 = align_B_to_A(pred_pts, p1, p2, pts_3d)
        # diff = pts_3d - pred_pts
        # print(diff)
        # distance = (diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2) / len(pts_3d)
        # total_distance = np.sum(distance)
        # print("total_distance after BA", total_distance)

        if not use_noise:
            assert np.allclose(pred_pts, pts_3d)

        if final_vis:
            red = (1, 0, 0)
            green = (0, 1, 0)
            blue = (0, 0, 1)

            vis_3d(
                1500, 1500,
                o3d_pc(pts_3d, red),
                o3d_pc(pred_pts, green),
                draw_camera(K, pose1, img_w, img_h, 10, red),
                draw_camera(K, pose2, img_w, img_h, 10, red),
                draw_camera(K, p1, img_w, img_h, 10, blue),
                draw_camera(K, p2, img_w, img_h, 10, blue),
            )
        return p1, p2, pred_pts
        
    
    def visualize_final_view(self, pts, cameraposes):
        red = (1, 0, 0)
        green = (0, 1, 0)
        draw_cameras = [draw_camera(self.K, pose, self.img_w, self.img_h) for pose in cameraposes]
        vis_3d(1500, 1500, o3d_pc(pts), *draw_cameras)

    # these are part of the prep materials; not part of the problems
    def teaser(self):
        K = self.K

        n = len(self.data.poses)
        img_w, img_h = self.img_w, self.img_h

        fname = str(DATA_ROOT / "optimus_prime.obj")
        mesh = o3d.io.read_triangle_mesh(fname, False)

        cams = [
            draw_camera(K, self.data.poses[i], img_w, img_h, scale=10)
            for i in range(n)
        ]

        vis_3d(1500, 1500, mesh, o3d_pc(self.data.pts_3d), *cams)

    def show_visib(self):
        plt.imshow(self.data.visibility, aspect="auto", interpolation='nearest')
        plt.xlabel("3D points")
        plt.ylabel("Camera view")
        plt.yticks(np.arange(10, dtype=int))
        plt.show()


def corruption_pipeline(x1s, x2s):
    s = 1
    noise = s * np.random.randn(*x1s.shape)
    noise[:, -1] = 0  # cannot add noise to the 1! fatal error

    x1s = x1s + noise
    x2s = x2s + noise

    x1s = flip_correspondence(x1s, 0.02)
    # round to integer
    x1s = np.rint(x1s)
    x2s = np.rint(x2s)
    return x1s, x2s


def flip_correspondence(x1s, perc):
    n = x1s.shape[0]
    num_wrong = int(n * perc)
    chosen = np.random.choice(n, size=num_wrong, replace=False)
    x1s[chosen] = x1s[np.random.permutation(chosen)]
    return x1s


def main():
    engine = Engine()
    # run it one at a time

    # engine.q2()
    # engine.q4()
    engine.q10()
    # engine.q12()


if __name__ == "__main__":
    main()
