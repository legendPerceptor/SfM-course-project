import numpy as np
from numpy.linalg import inv, svd, det, norm

from render import (
    as_homogeneous,
    homogenize
)

import torch
import torch.optim as optim
from lietorch import SE3
from scipy.spatial.transform import Rotation as rot
from vis import vis_3d, o3d_pc, draw_camera


def skew(xs):
    # xs: [n, 3]
    # your code here in case you need it
    # should create skew symmetric matrix
    # return mats
    pass


def project(pts_3d, K, pose):
    P = K @ inv(pose)
    x1 = homogenize(pts_3d @ P.T)
    return x1


def pose_to_se3_embed(pose):
    R, t = pose[:3, :3], pose[:3, -1]
    tau = t
    phi = rot.from_matrix(R).as_quat()  # convert to quaternion
    embed = np.concatenate([tau, phi], axis=0)
    embed = torch.as_tensor(embed)
    return embed


def as_torch_tensor(*args):
    return [torch.as_tensor(elem) for elem in args]


def torch_project(pts_3d, K, se3_pose):
    P = K @ se3_pose.inv().matrix()
    x1 = pts_3d @ P.T
    x1 = homogenize(x1)
    return x1


def bundle_adjustment(x1s, x2s, full_K, p1, p2, pred_pts):
    embed1 = pose_to_se3_embed(p1)
    embed2 = pose_to_se3_embed(p2)

    embed1.requires_grad_(True)
    embed2.requires_grad_(True)

    x1s, x2s, full_K, pred_pts = \
        as_torch_tensor(x1s, x2s, full_K, pred_pts)

    pred_pts.requires_grad_(True)

    lr = 1e-3
    # optimizer = optim.SGD([embed1, embed2, pred_pts], lr=lr, momentum=0.9)
    optimizer = optim.Adam([embed1, embed2, pred_pts], lr=lr)

    n_steps = 10000
    for i in range(n_steps):
        optimizer.zero_grad()

        p1 = SE3.InitFromVec(embed1)
        p2 = SE3.InitFromVec(embed2)

        x1_hat = torch_project(pred_pts, full_K, p1)
        x2_hat = torch_project(pred_pts, full_K, p2)
        err1 = torch.norm((x1_hat - x1s), dim=1)
        err1 = err1.mean()
        err2 = torch.norm((x2_hat - x2s), dim=1)
        err2 = err2.mean()
        err = (err1 + err2) / 2

        err.backward()
        optimizer.step()

        if (i % (n_steps // 10)) == 0:
            print(f"step {i}, err: {err.item()}")

    p1 = SE3.InitFromVec(embed1).matrix().detach().numpy()
    p2 = SE3.InitFromVec(embed2).matrix().detach().numpy()
    pred_pts = pred_pts.detach().numpy()
    return p1, p2, pred_pts


def eight_point_algorithm(x1s, x2s):
    # estimate the fundamental matrix
    # your code here
    # print("x1s shape: ",x1s.shape)
    A = np.zeros((x1s.shape[0], 9))
    for i in range(x1s.shape[0]):
        # print("x1s[i].shape", x1s[i].shape)
        # print("x1s[i].T", x1s[i].T.shape)
        a = x2s[i].reshape((3, 1))
        b = x1s[i].reshape((1, 3))
        tmp = np.dot(a, b)
        # print("tmp: ", tmp)
        A[i] = tmp.reshape(9)
    U, s, V_t = svd(A)
    

    # cond number is the ratio of top rank / last rank singular values
    # when you solve Ax = b, you take s[0] / s[-1]. But in vision,
    # we convert the problem above into the form Ax = 0. The nullspace is reserved for the solution.
    # This is called a "homogeneous system of equations" in linear algebra. This might be the reason
    # why homogeneous coordiantes are called homogeneous.
    # hence s[0] / s[-2].
    cond = s[0] / s[-2]
    print(f"condition number {cond}")

    F = V_t[-1].reshape(3, 3)
    F = F / F[-1, -1]
    F = enforce_rank_2(F)
    return F


def enforce_rank_2(F):
    # your code here
    U, s, V_t = svd(F)
    s[-1] = 0
    F = U @ np.diag(s) @ V_t
    return F


def normalized_eight_point_algorithm(x1s, x2s, img_w, img_h):
    # your code here
    A = np.array([[2/img_w, 0, -1], [0, 2/img_h, -1], [0, 0, 1]])
    x1s = x1s @ A.T
    x2s = x2s @ A.T
    F = eight_point_algorithm(x1s, x2s)
    # your code here
    F = A.T @ F @ A
    return F


def triangulate(P1, x1s, P2, x2s):
    # x1s: [n, 3]
    assert x1s.shape == x2s.shape
    n = len(x1s)
    # you can follow this and write it in a vectorized way, or you can do it
    # row by row, entry by entry
    # x1s = skew(x1s)  # [n, 3, 3 ]
    # x2s = skew(x2s)
    
    pts = np.zeros((n, 4))
    for i in range(n):
        x1si_cross = np.array([[0, -x1s[i, 2], x1s[i, 1]], [x1s[i, 2], 0, -x1s[i, 0]], [-x1s[i, 1], x1s[i, 0], 0]])
        m1 = np.dot(x1si_cross, P1)
        # print("P1 shape", P1.shape)
        # print("P2 shape", P2.shape)
        # print("m1 shape", m1.shape)
        x2si_cross = np.array([[0, -x2s[i, 2], x2s[i, 1]], [x2s[i, 2], 0, -x2s[i, 0]], [-x2s[i, 1], x2s[i, 0], 0]])
        m2 = np.dot(x2si_cross, P2)
        # print("m2 shape", m2.shape)
        tmp = np.vstack([m1, m2])
        # print("tmp shape", tmp.shape)
        U, s, V_t = np.linalg.svd(tmp)
        # print("Vt shape", V_t.shape)
        cur = V_t[-1, :].reshape(4)
        cur = cur / cur[-1]
        # print("cur shape", cur.shape)
        # print("cur", cur)
        # print("cur[:3] shape", cur[:3].shape)
        pts[i] = cur
    # pts = None
    return pts


def t_and_R_from_pose_pair(p1, p2):
    """the R and t that transforms points from pose 1's local frame to pose 2's local frame
    """
    # your code
    invp2 = inv(p2)
    print("invp2", invp2)
    print("p1", p1)
    p1_to_p2 = np.dot(invp2, p1)
    print("p1_to_p2", p1_to_p2)
    t = p1_to_p2[:3, -1]
    R = p1_to_p2[:3, :3]
    return t, R


def pose_pair_from_t_and_R(t, R):
    """since we only have their relative orientation, the first pose
    is fixed to be identity
    """
    # p1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    p1 = np.eye(4)
    # your code here
    print("R.shape", R.shape)
    print("t.shape", t.shape)
    # inv(p2) p1 X =  X'
    # T inv(p1) = inv(p2) p1
    T = np.hstack([R, t.reshape((3,1))])
    T = np.vstack([T, np.array([0, 0, 0, 1])])
    # print("stack R,t", p2)
    p2 = inv(T @ inv(p1))
    print("stack p2", p2)
    return p1, p2


def essential_from_t_and_R(t, R):
    # your code here
    t_cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = np.dot(t_cross, R)
    return E


def t_and_R_from_essential(E):
    """this has even more ambiguity. there are 4 compatible (t, R) configurations
    out of which only 1 places all points in front of both cameras

    That the rank-deficiency in E induces 2 valid R is subtle...
    """
    # your code here; get t
    # get t from left-null space of E
    U, s, V_t = svd(E.T)
    # t = U[:,-1]
    t = V_t[-1, :]
    # t_mat = skew(t.reshape(1, -1))[0]
    t_cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])

    # now solve procrustes to get back R
    U, s, V_t = svd(np.dot(t_cross.T, E))
    # your code here
    R = U @ V_t
    # your code here

    # makes sure R has det 1, and that we have 2 possible Rs
    R1 = R * det(R)
    U[:, 2] = -U[:, 2]
    R = U @ V_t
    R2 = R * det(R)

    four_hypothesis = [
        [ t, R1],
        [-t, R1],
        [ t, R2],
        [-t, R2],
    ]
    return four_hypothesis


def disambiguate_four_chirality_by_triangulation(four, x1s, x2s, full_K, draw_config=False):
    # note that our camera is pointing towards its negative z axis
    num_infront = np.array([0, 0, 0, 0])
    four_pose_pairs = []

    for i, (t, R) in enumerate(four):
        p1, p2 = pose_pair_from_t_and_R(t, R)
        P1 = np.dot(full_K, inv(p1))
        P2 = np.dot(full_K, inv(p2))
        pts = triangulate(P1[:3, :], x1s, P2[:3, :], x2s)
        # how many points in front of camera 1?
        nv1 = inv(p1) @ pts.T
        nv1 = np.count_nonzero(nv1[2, :] < 0)
        # how many points in front of camera 2? 
        # tmp = np.hstack([pts, np.ones((pts.shape[0], 1))])
        # print("tmp 4*n shape", tmp.shape)
        nv2 = inv(p2) @ pts.T
        nv2 = np.count_nonzero(nv2[2, :] < 0)
        print("nv1", nv1)
        print("nv2", nv2)
        num_infront[i] = nv1 + nv2
        four_pose_pairs.append((p1, p2))
        if draw_config:
            vis_3d(
                1500, 1500, o3d_pc(_throw_outliers(pts)),
                draw_camera(full_K, p1, 1600, 1200),
                draw_camera(full_K, p2, 1600, 1200),
            )

    i = np.argmax(num_infront)
    print("chosen i", i)
    t, R = four[i]
    p1, p2 = four_pose_pairs[i]
    return p1, p2, t, R


def F_from_K_and_E(K, E):
    # your code
    return inv(K.T) @ E @ inv(K)


def E_from_K_and_F(K, F):
    # your code
    return K.T @ F @ K


def _throw_outliers(pts):
    pts_1 = pts[:, :3]
    mask = (np.abs(pts_1) > 100).any(axis=1)
    return pts[~mask]


def align_B_to_A(B, p1, p2, A):
    # B, A: [n, 3]
    assert B.shape == A.shape
    A = A[:, :3]
    B = B[:, :3]
    p1 = p1.copy()
    p2 = p2.copy()

    a_centroid = A.mean(axis=0)
    b_centroid = B.mean(axis=0)

    A = A - a_centroid
    B = B - b_centroid
    p1[:3, -1] -= b_centroid
    p2[:3, -1] -= b_centroid

    centroid = np.array([0, 0, 0])
    # root mean squre from centroid
    scale_a = (norm((A - centroid), axis=1) ** 2).mean(axis=0) ** 0.5
    scale_b = (norm((B - centroid), axis=1) ** 2).mean(axis=0) ** 0.5
    rms_ratio = scale_a / scale_b

    B = B * rms_ratio
    p1[:3, -1] *= rms_ratio
    p2[:3, -1] *= rms_ratio

    U, s, V_t = svd(B.T @ A)
    R = U @ V_t
    assert np.allclose(det(R), 1), "not special orthogonal matrix"
    new_B = B @ R  # note that here there's no need to transpose R... lol... this is subtle
    p1[:3] = R.T @ p1[:3]
    p2[:3] = R.T @ p2[:3]

    new_B = new_B + a_centroid
    new_B = as_homogeneous(new_B)
    p1[:3, -1] += a_centroid
    p2[:3, -1] += a_centroid
    return new_B, p1, p2
