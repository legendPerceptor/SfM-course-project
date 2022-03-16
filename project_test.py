import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
from render import camera_pose
from vis import vis_3d, o3d_pc, draw_camera
from sfm import pose_pair_from_t_and_R


def plot_imageset(images, figsize=(16, 12), nrows_ncols=(4,4)):
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols, axes_pad=0.1)

    for ax, im in zip(grid, images):
        ax.set_axis_off()
        if len(im.shape) < 3:
            ax.imshow(im, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    plt.show()


multi_view_image_paths = sorted(glob.glob("data/online/House1280/images/*.jpg"))
multi_view_images = [cv2.imread(path) for path in multi_view_image_paths]
plot_imageset(multi_view_images)

# Convert to gray scale
gray_multi_view_images = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in multi_view_images]
# plot_imageset(gray_multi_view_images)

# Detect Features
sift = cv2.SIFT_create()
keypoints = []
descriptors = []
for im in gray_multi_view_images:
    keypoint, descriptor = sift.detectAndCompute(im, None)
    keypoints.append(keypoint)
    descriptors.append(descriptor)
results = []
for i, cur_img in enumerate(gray_multi_view_images):
    p_cur = cv2.drawKeypoints(cur_img, keypoints[i], None, color=(0,255,0))
    results.append(p_cur)
# plot_imageset(results)



# Match descriptors
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

all_matches = []
match_imgs = []
n = len(gray_multi_view_images)
for i in range(n-1):
    j = i + 1
    matches = bf.match(descriptors[i], descriptors[j])
    # matches = sorted(matches, key=lambda x: x.distance)
    result = cv2.drawMatches(gray_multi_view_images[i], keypoints[i], gray_multi_view_images[j], keypoints[j], matches, None, flags=2)
    match_imgs.append(result)
    all_matches.append(matches)

# plot_imageset(match_imgs)

# def getHomographyImg(matches, keypoints, pano_imgs, i, j):
#     good = matches

#     src_pts = np.float32([ keypoints[i][m.queryIdx].pt for m in good ])
#     dst_pts = np.float32([ keypoints[j][m.trainIdx].pt for m in good ])
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#     print(len(matchesMask), len(src_pts))
#     draw_params = dict(#matchColor = (0,255,0), # draw matches in green color
#                     singlePointColor = None,
#                     matchesMask = matchesMask, # draw only inliers
#                     flags = 2)
#     img3 = cv2.drawMatches(pano_imgs[i],keypoints[i],pano_imgs[j],keypoints[j],good,None,**draw_params)
#     return img3, M, matchesMask


# imgs_after_RANSAC = []
# H_matrices = []
# for i, match in enumerate(all_matches):
#     img, M, mask = getHomographyImg(match, keypoints, gray_multi_view_images, i, (i+1)%n)
#     imgs_after_RANSAC.append(img)
#     H_matrices.append(M)
# plot_imageset(imgs_after_RANSAC, figsize=(20,20))

from run1 import Engine

img_w = 1280
img_h = 720
fov = 83
eng = Engine(img_w, img_h, fov)



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



def compute_intrinsics(aspect_ratio, fov, img_height_in_pix):
    # aspect ratio is  w / h
    intrinsic = np.dot(compute_normalized_to_img_trans(aspect_ratio, img_height_in_pix), compute_proj_to_normalized(aspect_ratio, fov))
    # your code here
    return intrinsic


def compute_proj_to_normalized(aspect, fov):
    # compared to standard OpenGL NDC intrinsic,
    # this skips the 3rd row treatment on z. hence the name partial_ndc
    # it's incomplete, but enough for now

    # note that fov is in degrees. you need it in rad
    # note also that your depth is negative. where should you put a -1 somewhere?

    fov_radian = fov * np.pi / 180
    my_tan = np.tan(fov_radian / 2)
    # 3D point [x, y, z, 1] is projected
    partial_ndc_intrinsic = np.array([[0.5 / my_tan, 0 ,0, 0],
                [0, 0.5 * aspect/ my_tan, 0, 0],
                [0, 0, -1, 0]])
    return partial_ndc_intrinsic


def compute_normalized_to_img_trans(aspect, img_height_in_pix):
    img_h = img_height_in_pix
    img_w = img_height_in_pix * aspect

    # your code here
    ndc_to_img = np.array([[img_w/2, 0, img_w/2-0.5], [0, -img_h/2, img_h/2-0.5], [0 ,0, 1]])
    return ndc_to_img




def recover_3d_from_one_match(match, i, j):
    src_pts = np.float32([ keypoints[i][m.queryIdx].pt for m in match ])
    dst_pts = np.float32([ keypoints[j][m.trainIdx].pt for m in match ])
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    # print(matchesMask)
    src_pts = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))
    dst_pts = np.hstack((dst_pts, np.ones((dst_pts.shape[0], 1))))
    print(src_pts.shape, dst_pts.shape)
    src_cleared = []
    dst_cleared = []
    for i in range(len(matchesMask)):
        if matchesMask[i] == 1:
            src_cleared.append(src_pts[i])
            dst_cleared.append(dst_pts[i])
    src_cleared = np.array(src_cleared)
    dst_cleared = np.array(dst_cleared)
    # print(src_cleared)
    print(src_cleared.shape, dst_cleared.shape)
    p1, p2, pts =  eng.sfm_pipeline_for_project(src_cleared, dst_cleared, src_cleared, dst_cleared, use_BA = True, draw_config=True, final_vis=False)
    return p1, p2, pts, dst_cleared, matchesMask



p1, p2, pts, last_2d_pts, last_mask = recover_3d_from_one_match(all_matches[0], 0, 1)
all_pts = pts.copy()
camera_poses = [p1, p2]

# for i in range(2, len(all_matches)):
#     src_pts = np.float32([ keypoints[i][m.queryIdx].pt for m in all_matches[i] ])
#     dst_pts = np.float32([ keypoints[i-1][m.trainIdx].pt for m in all_matches[i] ])
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()

    
#     src_pts = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))
#     dst_pts = np.hstack((dst_pts, np.ones((dst_pts.shape[0], 1))))
#     src_cleared = []
#     dst_cleared = []
#     for j in range(len(matchesMask)):
#         if matchesMask[i] == 1:
#             src_cleared.append(src_pts[i])
#             dst_cleared.append(dst_pts[i])
#     src_cleared = np.array(src_cleared)
#     dst_cleared = np.array(dst_cleared)

#     if(len(dst_cleared) < 10):
#         print("not enough matches found when i =", i)
#         continue

#     matching_previous_2d = []
#     matching_previous_pts = []

#     for j, last2dpt in enumerate(last_2d_pts):
#         for k, cur2dpt in enumerate(dst_cleared):
#             if np.array_equal(cur2dpt, last2dpt):
#                 matching_previous_2d.append(cur2dpt)
#                 matching_previous_pts.append(pts[j])
    
#     if(len(matching_previous_2d) < 10):
#         print("not enough common pts found when i =", i)
#         continue

#     print(len(matching_previous_2d))
#     matching_previous_2d = np.array(matching_previous_2d)
#     matching_previous_pts = np.array(matching_previous_pts)

#     P = pnp_calibration(matching_previous_2d, matching_previous_pts)
#     # p2 = cv2.solvePnP(matching_previous_pts, matching_previous_2d, eng.K[:3, :3], 
#     R_t = np.linalg.inv(eng.K[:3, :3]) @ P # 3* 4
#     R = R_t[:, :3]
#     t = R_t[:, 3]
#     p1_rec, p2_rec = pose_pair_from_t_and_R(t, R)
#     # registered the new camera
#     # compute the new 3D points
#     # x1s = src_cleared
#     # x2s = dst_cleared
#     p1, p2, pts = eng.sfm_addcamera(p1=p2, p2=p2_rec, x1s=src_cleared, x2s=dst_cleared)
#     camera_poses.append(p2)
#     last_2d_pts = dst_cleared
#     last_mask = matchesMask
#     all_pts = np.vstack((all_pts, pts))

from sfm import pose_pair_from_t_and_R



for i in range(1, 3):
    src_pts = np.float32([ keypoints[i][m.queryIdx].pt for m in all_matches[i] ])
    dst_pts = np.float32([ keypoints[i+1][m.trainIdx].pt for m in all_matches[i] ])
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    
    src_pts = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))
    dst_pts = np.hstack((dst_pts, np.ones((dst_pts.shape[0], 1))))
    src_cleared = []
    dst_cleared = []
    for j in range(len(matchesMask)):
        if matchesMask[j] == 1:
            src_cleared.append(src_pts[j])
            dst_cleared.append(dst_pts[j])
    src_cleared = np.array(src_cleared)
    dst_cleared = np.array(dst_cleared)

    matching_previous_2d = []
    matching_previous_pts = []

    for j, last2dpt in enumerate(last_2d_pts):
        for k, cur2dpt in enumerate(src_cleared):
            if np.array_equal(cur2dpt, last2dpt):
                matching_previous_2d.append(cur2dpt)
                matching_previous_pts.append(pts[j])

    print(len(matching_previous_2d))
    matching_previous_2d = np.array(matching_previous_2d)
    matching_previous_pts = np.array(matching_previous_pts)
    if(len(matching_previous_2d) < 10):
        print("not enough common pts found when i =", i)
        continue
    P = pnp_calibration(matching_previous_2d, matching_previous_pts)
    # p2 = cv2.solvePnP(matching_previous_pts, matching_previous_2d, eng.K[:3, :3], 
    R_t = np.linalg.inv(eng.K[:3, :3]) @ P # 3* 4
    R = R_t[:, :3]
    t = R_t[:, 3]
    p1_rec, p2_rec = pose_pair_from_t_and_R(t, R)
     # 3 * 4
    # registered the new camera
    # compute the new 3D points
    # x1s = src_cleared
    # x2s = dst_cleared
    p1, p2, pts = eng.sfm_addcamera(p1=p2, p2=p2_rec, x1s=src_cleared, x2s=dst_cleared, P=P)
    camera_poses.append(p2)
    last_2d_pts = dst_cleared
    last_mask = matchesMask
    all_pts = np.vstack((all_pts, pts))

 

 
eng.visualize_final_view(all_pts, camera_poses)

    


