import numpy as np

import open3d as o3d 
import torch
import math

from pvrcnn.core import cfg, Preprocessor
from pvrcnn.detector import PV_RCNN, Second
from pvrcnn.ops import nms_rotated, box_iou_rotated
from pvrcnn.core import cfg, AnchorGenerator


def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])
    def rotx(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0],
                         [0, c, s],
                         [0, -s, c]])
    def rotz(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, s, 0],
                         [-s, c, 0],
                         [0, 0, 1]])

    R = rotz(math.pi/2 - heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d

## 解析JSON文件获取box信息
def get_box_annotation(boxes):
    with open(label_file, encoding='utf-8') as f:
        res = f.read()
        result = json.loads(res)
    file_name = result["fileName"]
    print(" label file name:", file_name)
    boxes = result["elem"]
    return boxes

def box2lineset(box):
    box_center = box[:3]
    # box_size = box[3, 5, 4]
    box_size = np.array([box[3], box[5], box[4]], dtype=np.float)
    box_yaw = box[6]
    corners_3d = get_3d_box(box_size, box_yaw, box_center)
    lines = [[0,1], [0,3], [1,2], [2,3], [4,5], [4,7], [5,6], [6,7],
             [0,4], [1,5], [2,6], [3,7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_3d)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # print(corners_3d)
    return line_set

def custom_draw_geometry_with_key_callback(pcd):
    def change_point_size(vis):
        opt = vis.get_render_option()
        print(opt.point_size)
        opt.point_size = 2.3
        return False
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False
    def change_field_of_view(vis):
        opt = vis.get_view_control()
        opt.set_constant_z_far(1000)
        # print(opt.get_field_of_view())
        # opt.change_field_of_view(0.6)
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_image("depth.png")
        # plt.imshow(np.asarray(depth))
        # plt.show()
        return False
    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False
    key_to_callback = {}

    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = change_point_size
    key_to_callback[ord("J")] = change_field_of_view
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks(pcd, key_to_callback)

def inference(out, anchors, cfg):
    cls_map, reg_map = out['P_cls'].squeeze(0), out['P_reg'].squeeze(0)
    score_map = cls_map.sigmoid()
    top_scores, class_idx = score_map.view(cfg.NUM_CLASSES, -1).max(0)
    top_scores, anchor_idx = top_scores.topk(k=500)
    class_idx = class_idx[anchor_idx]
    top_anchors = anchors.view(cfg.NUM_CLASSES, -1, cfg.BOX_DOF)[class_idx, anchor_idx]
    top_boxes = reg_map.reshape(cfg.NUM_CLASSES, -1, cfg.BOX_DOF)[class_idx, anchor_idx]

    P_xyz, P_wlh, P_yaw = top_boxes.split([3, 3, 1], dim=1)
    A_xyz, A_wlh, A_yaw = top_anchors.split([3, 3, 1], dim=1)

    A_wl, A_h = A_wlh.split([2, 1], -1)
    A_norm = A_wl.norm(dim=-1, keepdim=True).expand(-1, 2)
    A_norm = torch.cat((A_norm, A_h), dim=-1)

    top_boxes = torch.cat((
        (P_xyz * A_norm + A_xyz),
        (torch.exp(P_wlh) * A_wlh),
        (P_yaw + A_yaw)), dim=1
    )

    nms_idx = nms_rotated(top_boxes[:, [0, 1, 3, 4, 6]], top_scores, iou_threshold=0.005)
    top_boxes = top_boxes[nms_idx]
    top_scores = top_scores[nms_idx]
    top_classes = class_idx[nms_idx]
    return top_boxes, top_scores, top_classes

def viz_detections(points, boxes):
    boxes = boxes.cpu().numpy()
    bev_map = Drawer(points, [boxes]).image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bev_map.transpose(1, 0, 2)[::-1])
    ax.set_axis_off()
    fig.tight_layout()
    plt.show()

def load_ckpt(fpath, model, optimizer):
    if not osp.isfile(fpath):
        return 0
    ckpt = torch.load(fpath)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']
    return epoch

def get_model(cfg):
    cfg.merge_from_file('./configs/UDI/all_class_lite.yaml')
    anchors = AnchorGenerator(cfg).anchors
    preprocessor = Preprocessor(cfg)
    model = Second(cfg).cuda().eval()
    ckpt = torch.load('../pvrcnn/ckpts/epoch_12.pth')['state_dict']
    model.load_state_dict(ckpt, strict=True)
    return model, preprocessor, anchors


def main():
    model, preprocessor, anchors = get_model(cfg)
    fpath = '../data/kitti/training/velodyne_reduced/000032.bin'
    points = np.fromfile(fpath, np.float32).reshape(-1, 4)
    with torch.no_grad():
        item = preprocessor(dict(points=[points], anchors=anchors))
        for key in ['points', 'features', 'coordinates', 'occupancy', 'anchors']:
            item[key] = item[key].cuda()
        boxes, batch_idx, class_idx, scores = model.inference(item)
    viz_detections(points, boxes)


if __name__ == '__main__':
    # cfg.merge_from_file('./configs/UDI/all_class_lite.yaml')
    main()
