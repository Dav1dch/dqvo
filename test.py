import numpy as np
from superpoint import SuperPoint
import cv2

def rotationMatrixToQuaternion3(m):
    #q0 = qw
    t = np.matrix.trace(m)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if(t > 0):
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5/t
        q[0] = (m[2,1] - m[1,2]) * t
        q[1] = (m[0,2] - m[2,0]) * t
        q[2] = (m[1,0] - m[0,1]) * t

    else:
        i = 0
        if (m[1,1] > m[0,0]):
            i = 1
        if (m[2,2] > m[i,i]):
            i = 2
        j = (i+1)%3
        k = (j+1)%3

        t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k,j] - m[j,k]) * t
        q[j] = (m[j,i] + m[i,j]) * t
        q[k] = (m[k,i] + m[i,k]) * t

    return q


def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, )
    :param q2: (4, )
    :return:
    """

    # d = abs(np.dot(q1, q2))
    d = abs(q2 @ q1.T)
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta

import torch

def main():
    img = cv2.imread("/home/david/datasets/kitti/color/000000.png", cv2.IMREAD_GRAYSCALE) / 255
    img = torch.from_numpy(img).cuda().unsqueeze(0).type(torch.float32).unsqueeze(0)
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints':100 
        }
    }
    print(img.shape)
    sp = SuperPoint(config['superpoint']).cuda()
    pred = sp(img)
    print(len(pred["keypoints"][0]))
    # t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
    # q_criterion = quaternion_angular_error
    # poses = np.loadtxt("./data/poses/00.txt")
    # pose1 = poses[51].reshape(3, 4)
    # pose1 = np.array(list(pose1[:, 3]) + list(rotationMatrixToQuaternion3(pose1[:3, :3])))
    # pose2 = poses[86].reshape(3, 4)
    # pose2 = np.array(list(pose2[:, 3]) + list(rotationMatrixToQuaternion3(pose2[:3, :3])))
    # pose3 = poses[106].reshape(3, 4)
    # pose3 = np.array(list(pose3[:, 3]) + list(rotationMatrixToQuaternion3(pose3[:3, :3])))
    # print(t_criterion(pose1[:3], pose2[:3]))
    # print(q_criterion(pose1[3:], pose2[3:]))
    #
    # print(t_criterion(pose2[3], pose3[3]))
    # print(q_criterion(pose2[3:], pose3[3:]))

    return

if __name__ == "__main__":
    main()


