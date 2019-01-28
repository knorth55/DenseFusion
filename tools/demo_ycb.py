import _init_paths  # NOQA

import argparse
import copy
import numpy as np
import numpy.ma as ma
from PIL import Image
import scipy.io as scio
import random

import open3d
from open3d import PinholeCameraIntrinsic
import torch
from torch.autograd import Variable
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms

from lib.network import PoseNet
from lib.network import PoseRefineNet
from lib.transformations import quaternion_from_matrix
from lib.transformations import quaternion_matrix

# global param
norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [
    -1, 40, 80, 120, 160, 200, 240, 280, 320,
    360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0
num_obj = 21
img_width = 640
img_height = 480
num_points = 1000
iteration = 2
bs = 1
dataset_config_dir = 'datasets/ycb/dataset_config'
ycb_toolbox_dir = 'YCB_Video_toolbox'
result_wo_refine_dir = \
    'experiments/eval_result/ycb/Densefusion_wo_refine_result'
result_refine_dir = \
    'experiments/eval_result/ycb/Densefusion_iterative_result'
trained_models_dir = 'trained_models/ycb'
# global param


def get_bbox(posecnn_rois, idx):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_height:
        delt = rmax - img_height
        rmax = img_height
        rmin -= delt
    if cmax > img_width:
        delt = cmax - img_width
        cmax = img_width
        cmin -= delt
    return rmin, rmax, cmin, cmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, help='dataset root dir')
    parser.add_argument('--model', type=str, help='resume PoseNet model')
    parser.add_argument(
        '--refine_model', type=str, help='resume PoseRefineNet model')
    args = parser.parse_args()

    estimator = PoseNet(num_points=num_points, num_obj=num_obj)
    estimator.cuda()
    estimator.load_state_dict(
        torch.load('{0}/{1}'.format(trained_models_dir, args.model)))
    estimator.eval()

    refiner = PoseRefineNet(num_points=num_points, num_obj=num_obj)
    refiner.cuda()
    refiner.load_state_dict(
        torch.load('{0}/{1}'.format(trained_models_dir, args.refine_model)))
    refiner.eval()

    testlist = []
    input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        testlist.append(input_line)
    input_file.close()
    # print(len(testlist))

    # class_file = open('{0}/classes.txt'.format(dataset_config_dir))
    # class_id = 1
    # cld = {}
    # while 1:
    #     class_input = class_file.readline()
    #     if not class_input:
    #         break
    #     class_input = class_input[:-1]
    #
    #     input_file = open(
    #         '{0}/models/{1}/points.xyz'
    #         .format(args.dataset_root, class_input))
    #     cld[class_id] = []
    #     while 1:
    #         input_line = input_file.readline()
    #         if not input_line:
    #             break
    #         input_line = input_line[:-1]
    #         input_line = input_line.split(' ')
    #         cld[class_id].append(
    #             [float(input_line[0]),
    #              float(input_line[1]),
    #              float(input_line[2])])
    #     input_file.close()
    #     cld[class_id] = np.array(cld[class_id])
    #     class_id += 1

    for now in random.sample(list(range(0, 2949)), 2949):
        imgpath = '{0}/{1}-color.png'.format(args.dataset_root, testlist[now])
        depthpath = '{0}/{1}-depth.png'.format(
            args.dataset_root, testlist[now])
        img = Image.open(imgpath)
        depth = np.array(Image.open(depthpath))
        posecnn_meta = scio.loadmat(
            '{0}/results_PoseCNN_RSS2018/{1}.mat'
            .format(ycb_toolbox_dir, '%06d' % now))
        label = np.array(posecnn_meta['labels'])
        posecnn_rois = np.array(posecnn_meta['rois'])

        lst = posecnn_rois[:, 1:2].flatten()
        my_result_wo_refine = []
        my_result = []

        for idx in range(len(lst)):
            itemid = lst[idx]
            try:
                rmin, rmax, cmin, cmax = get_bbox(posecnn_rois, idx)

                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
                mask = mask_label * mask_depth

                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                if len(choose) > num_points:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:num_points] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(
                        choose, (0, num_points - len(choose)), 'wrap')

                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()
                depth_masked = depth_masked[choose][:, np.newaxis]
                depth_masked = depth_masked.astype(np.float32)
                xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()
                xmap_masked = xmap_masked[choose][:, np.newaxis]
                xmap_masked = xmap_masked.astype(np.float32)
                ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()
                ymap_masked = ymap_masked[choose][:, np.newaxis]
                ymap_masked = ymap_masked.astype(np.float32)
                choose = np.array([choose])

                pt2 = depth_masked / cam_scale
                pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                img_masked = np.array(img)[:, :, :3]
                img_masked = np.transpose(img_masked, (2, 0, 1))
                img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                cloud = torch.from_numpy(cloud.astype(np.float32))
                choose = torch.LongTensor(choose.astype(np.int32))
                img_masked = norm(
                    torch.from_numpy(img_masked.astype(np.float32)))
                index = torch.LongTensor([itemid - 1])

                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                img_masked = Variable(img_masked).cuda()
                index = Variable(index).cuda()

                cloud = cloud.view(1, num_points, 3)
                img_masked = img_masked.view(
                    1, 3, img_masked.size()[1], img_masked.size()[2])

                pred_r, pred_t, pred_c, emb = estimator(
                    img_masked, cloud, choose, index)
                pred_r = pred_r / torch.norm(
                    pred_r, dim=2).view(1, num_points, 1)

                pred_c = pred_c.view(bs, num_points)
                how_max, which_max = torch.max(pred_c, 1)
                pred_t = pred_t.view(bs * num_points, 1, 3)
                points = cloud.view(bs * num_points, 1, 3)

                my_r = pred_r[0][which_max[0]].view(-1)
                my_r = my_r.cpu().data.numpy()
                my_t = (points + pred_t)[which_max[0]].view(-1)
                my_t = my_t.cpu().data.numpy()
                my_pred = np.append(my_r, my_t)
                my_result_wo_refine.append(my_pred.tolist())

                for ite in range(0, iteration):
                    T = Variable(torch.from_numpy(my_t.astype(np.float32)))
                    T = T.cuda().view(1, 3).repeat(num_points, 1)
                    T = T.contiguous().view(1, num_points, 3)
                    my_mat = quaternion_matrix(my_r)
                    R = Variable(
                        torch.from_numpy(my_mat[:3, :3].astype(np.float32)))
                    R = R.cuda().view(1, 3, 3)
                    my_mat[0][3] = my_t[0]
                    my_mat[1][3] = my_t[1]
                    my_mat[2][3] = my_t[2]

                    new_cloud = torch.bmm((cloud - T), R).contiguous()
                    pred_r, pred_t = refiner(new_cloud, emb, index)
                    pred_r = pred_r.view(1, 1, -1)
                    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                    my_r_2 = pred_r.view(-1).cpu().data.numpy()
                    my_t_2 = pred_t.view(-1).cpu().data.numpy()
                    my_mat_2 = quaternion_matrix(my_r_2)

                    my_mat_2[0][3] = my_t_2[0]
                    my_mat_2[1][3] = my_t_2[1]
                    my_mat_2[2][3] = my_t_2[2]

                    my_mat_final = np.dot(my_mat, my_mat_2)
                    my_r_final = copy.deepcopy(my_mat_final)
                    my_r_final[0][3] = 0
                    my_r_final[1][3] = 0
                    my_r_final[2][3] = 0
                    my_r_final = quaternion_from_matrix(my_r_final, True)
                    my_t_final = np.array(
                        [my_mat_final[0][3],
                         my_mat_final[1][3],
                         my_mat_final[2][3]])

                    my_pred = np.append(my_r_final, my_t_final)
                    my_r = my_r_final
                    my_t = my_t_final
                my_result.append(my_pred.tolist())
            except ZeroDivisionError:
                print('PoseCNN Detector Lost {0} at No.{1} keyframe'
                      .format(itemid, now))
                my_result_wo_refine.append([0.0 for i in range(7)])
                my_result.append([0.0 for i in range(7)])
        # scio.savemat(
        #     '{0}/{1}.mat'.format(result_wo_refine_dir, '%04d' % now),
        #     {'poses': my_result_wo_refine})
        # scio.savemat(
        #     '{0}/{1}.mat'.format(result_refine_dir, '%04d' % now),
        #     {'poses': my_result})

        # visualization
        intrinsic = PinholeCameraIntrinsic(
            img_width, img_height, cam_fx, cam_fy, cam_cx, cam_cy)
        vis_rgb = open3d.read_image(imgpath)
        vis_depth = open3d.read_image(depthpath)
        vis_rgbd = open3d.create_rgbd_image_from_color_and_depth(
            vis_rgb, vis_depth, depth_scale=cam_scale,
            convert_rgb_to_intensity=False)
        vis_pcd = open3d.create_point_cloud_from_rgbd_image(
            vis_rgbd, intrinsic)
        vis_pcd.transform(
            [[1, 0, 0, 0],
             [0, -1, 0, 0],
             [0, 0, -1, 0],
             [0, 0, 0, 1]]
        )
        pcds = [vis_pcd]

        with open('{0}/classes.txt'.format(dataset_config_dir)) as f:
            label_names = f.read().split('\n')[:-1]

        for lbl, pose in zip(lst, my_result):
            lbl = int(lbl - 1)
            obj_pcdpath = '{0}/models/{1}/points.xyz'.format(
                args.dataset_root, label_names[lbl])
            obj_pcd = open3d.read_point_cloud(obj_pcdpath)
            rq = quaternion_matrix(pose[:4])
            rq[0][3] = pose[4]
            rq[1][3] = pose[5]
            rq[2][3] = pose[6]
            obj_pcd.transform(rq)
            obj_pcd.transform(
                [[1, 0, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, -1, 0],
                 [0, 0, 0, 1]]
            )
            pcds.append(obj_pcd)
        open3d.draw_geometries(pcds)
        # print(my_result_wo_refine)
        # print(my_result)
        print("Finish No.{0} keyframe".format(now))


if __name__ == '__main__':
    main()
