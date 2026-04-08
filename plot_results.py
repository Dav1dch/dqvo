import glob
import os
import pickle
import queue

import matplotlib.pyplot as plt
import numpy as np

from datasets.kitti import KITTI
from datasets.utils import euler_to_rotation, dual_quaternion_to_pose

import matplotlib

matplotlib.use("Agg")


def save_trajectory(poses, sequence, save_dir):
    """
    Save predicted poses in .txt file
    Args:
        poses {ndarray}: list with all 4x4 pose matrix
        sequence {str}: sequence of KITTI dataset
        save_dir {str}: path to save pose
    """
    # create directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output_filename = os.path.join(save_dir, "{}.txt".format(sequence))
    with open(output_filename, "w") as f:
        for pose in poses:
            pose = pose.flatten()[:12]
            line = " ".join([str(x) for x in pose]) + "\n"
            f.write(line)


def post_processing(pred_poses, args):

    if args["window_size"] == 2:
        pred_poses = pred_poses.squeeze(1)
        return np.asarray(pred_poses)

    num_batchs = pred_poses.shape[0]

    # get poses in overlaped frames
    q = queue.Queue(args["window_size"] - 1)  # The max size is 5.
    idx = 0
    poses = []

    while not q.full():
        q.put(pred_poses[idx, :, :])
        idx = idx + 1

    while idx < num_batchs:
        # process first full queue
        if idx == (args["window_size"] - 1):
            poses.append(q.queue[0][0, :])

            # implemented for specific case window_size = 3 and overlap = 2
            avg_pose = (q.queue[0][1, :] + q.queue[1][0, :]) / 2
            poses.append(avg_pose)

            if args["window_size"] == 4:
                # implemented for specific case window_size = 4 and overlap = 3
                avg_pose = (q.queue[0][2, :] + q.queue[1][1, :] + q.queue[2][0, :]) / 3
                poses.append(avg_pose)

        elif idx < (num_batchs - 1):
            if args["window_size"] == 3:
                # implemented for specific case window_size = 3 and overlap = 2
                avg_pose = (q.queue[0][1, :] + q.queue[1][0, :]) / 2
                poses.append(avg_pose)

            elif args["window_size"] == 4:
                # implemented for specific case window_size = 4 and overlap = 3
                avg_pose = (q.queue[0][2, :] + q.queue[1][1, :] + q.queue[2][0, :]) / 3
                poses.append(avg_pose)

        # process last full queue (idx == num_batchs-1)
        else:
            if args["window_size"] == 3:
                # implemented for specific case window_size = 3 and overlap = 2
                poses.append(q.queue[1][1, :])

            elif args["window_size"] == 4:
                # implemented for specific case window_size = 4 and overlap = 2
                avg_pose = (q.queue[1][2, :] + q.queue[2][1, :]) / 2
                poses.append(avg_pose)
                poses.append(q.queue[2][2, :])

            idx = idx + 1

        # update queue
        if idx < (num_batchs - 1):
            idx = idx + 1
            first = q.get()  # dequeue first element
            q.put(pred_poses[idx, :, :])

    return np.asarray(poses)


def recover_trajectory_and_poses(poses, norm, use_dual_quaternion=False):

    if use_dual_quaternion:
        return recover_trajectory_and_poses_dual_quaternion(poses, norm)
    else:
        return recover_trajectory_and_poses_euler(poses, norm)


def recover_trajectory_and_poses_euler(poses, norm):

    predicted_poses = []
    # recover predicted trajectory
    predicted_trajectory = []
    T = np.eye(4)  # Initialize T at the start

    for i in range(len(poses) - 1):
        angles = poses[i, :3]
        t = poses[i, 3:]

        # undo normalization
        mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])
        # mean_t = np.array(
        #     [0.0013561882415943965, 0.0020733994642250808, -0.00014500151678821502]
        # )
        # std_t = np.array([0.17138581278908738, 0.18442089670966283, 0.097886895944611])
        # mean_angles = np.array(
        #     [-1.27889350370224e-05, -0.00016774727133543198, -8.898331117560475e-05]
        # )
        # std_angles = np.array(
        #     [0.018866636090010627, 0.013954030128821779, 0.012936541020268352]
        # )
        [x, y, z] = angles

        if norm:
            [x, y, z] = np.multiply(angles, std_angles) + mean_angles
            t = np.multiply(t, std_t) + mean_t
        R = np.asarray(euler_to_rotation(x, y, z, seq="zyx"))

        T_r = np.concatenate(
            (
                np.concatenate([R, np.reshape(t, (3, 1))], axis=1),
                [[0.0, 0.0, 0.0, 1.0]],
            ),
            axis=0,
        )
        T_abs = np.dot(T, T_r)
        T = T_abs

        predicted_poses.append(T)
        predicted_trajectory.append(T_abs[:3, 3])

    return predicted_poses, predicted_trajectory


def recover_trajectory_and_poses_dual_quaternion(poses, norm):
    """
    Recover trajectory and poses from dual quaternion representation.

    Args:
        poses: array of shape (N, 7) where each row is [qw, qx, qy, qz, tx, ty, tz]
        norm: whether to undo normalization

    Returns:
        predicted_poses: list of 4x4 transformation matrices
        predicted_trajectory: list of translation vectors
    """
    predicted_poses = []
    predicted_trajectory = []

    # Undo normalization for translation
    mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
    std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

    T = np.eye(4)  # Initialize T at the start

    for i in range(len(poses) - 1):
        # Extract dual quaternion components
        dq = poses[i, :]  # [qw, qx, qy, qz, tx, ty, tz]

        # Extract rotation quaternion and translation
        q = dq[:4]  # rotation quaternion
        t = dq[4:]  # translation

        # Undo normalization for translation
        if norm:
            t = np.multiply(t, std_t) + mean_t

        # Create dual quaternion with normalized translation
        dq_normalized = np.concatenate([q, t])

        # Convert dual quaternion to 4x4 transformation matrix
        T_r = dual_quaternion_to_pose(dq_normalized)

        # Compose transformation
        T_abs = np.dot(T, T_r)
        T = T_abs

        predicted_poses.append(T)
        predicted_trajectory.append(T_abs[:3, 3])

    return predicted_poses, predicted_trajectory


if __name__ == "__main__":

    ckpt_path = "checkpoints/Exp53_tri/"
    ckpt_name = "checkpoint_last"
    sequences = ["01", "03", "04", "05", "06", "07", "10"]
    # sequences = ['00',"01",'02', "03", "04", "05", "06", "07", '08', '09',"10"]
    # sequences = ["03", "07"]
    # sequences = ["00"]
    # sequences = ["06", "07", "09", "10"]

    # read hyperparameters and configuration
    with open(os.path.join(ckpt_path, "args.pkl"), "rb") as f:
        args = pickle.load(f)
    f.close()
    args["window_size"] = 3
    args["overlap"] = 2

    ckpt_path = os.path.join(ckpt_path, ckpt_name)
    args["checkpoint_path"] = ckpt_path

    # plot trajectory and ground truth
    for sequence in sequences:
        # read ground test data and predicted poses
        pred_path = os.path.join(
            args["checkpoint_path"], "pred_poses_{}.npy".format(sequence)
        )
        # pred_TS_path = "/home/david/Codes/OTSformer-fp/checkpoints/TS/checkpoint_best/pred_poses_{}.npy".format(
        #     sequence
        # )
        pred_poses = np.load(pred_path)
        # pred_TS_poses = np.load(pred_TS_path)

        # post processing and recover trajectory
        poses = post_processing(pred_poses, args)
        # poses_TS = post_processing(pred_TS_poses, args)

        # Check if we're using dual quaternion representation
        use_dual_quaternion = args.get("use_dual_quaternion", False)
        pred_poses, pred_trajectory = recover_trajectory_and_poses(
            poses, True, use_dual_quaternion
        )
        # pred_TS_poses, pred_TS_trajectory = recover_trajectory_and_poses(poses_TS, True)

        save_trajectory(
            pred_poses,
            sequence,
            save_dir=os.path.join(args["checkpoint_path"], "pred_poses"),
        )

        # get ground truth trajectories
        test_data = KITTI(sequences=[sequence], window_size=args["window_size"])
        gt_poses = test_data.windowed_data.loc[
            test_data.windowed_data["sequence"] == sequence, [3, 7, 11]
        ]

        plt.figure()
        pred_trajectory = np.asarray(pred_trajectory)
        plt.plot(
            [x[0] for x in pred_trajectory], [z[2] for z in pred_trajectory], "b"
        )  # plot estimated trajectory
        # plt.plot(
        #     [x[0] for x in pred_TS_trajectory], [z[2] for z in pred_TS_trajectory], "g"
        # )  # plot ground truth trajectory
        plt.plot(
            [x[0] for x in gt_poses.values], [z[2] for z in gt_poses.values], "r"
        )  # plot ground truth trajectory
        plt.grid()
        plt.title("VO - Seq {}".format(sequence))
        plt.xlabel("Translation in x direction [m]")
        plt.ylabel("Translation in z direction [m]")
        plt.legend(["OURS", "TSFormer-VO", "ground truth"])

        # create checkpoints folder
        if not os.path.exists(os.path.join(args["checkpoint_path"], "plots")):
            os.makedirs(os.path.join(args["checkpoint_path"], "plots"))
        plt.savefig(
            os.path.join(
                args["checkpoint_path"], "plots", "pred_traj_{}.png".format(sequence)
            )
        )
