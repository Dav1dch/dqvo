import functools

import cv2
import numpy as np
import torch

from scipy.spatial.transform import Rotation


def convert_tartan_pose_to_kitti(x, y, z, qx, qy, qz, qw):
    """
    Convert TartanAir pose to KITTI-format 3x4 transformation matrix.

    Args:
        x, y, z: Position coordinates
        qw, qx, qy, qz: Orientation as unit quaternion (scalar first)

    Returns:
        np.ndarray: 3x4 transformation matrix in KITTI format
    """
    # Construct rotation matrix from quaternion
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # Scipy uses [x,y,z,w]

    # Extract rotation matrix components
    r00, r01, r02 = R[0]
    r10, r11, r12 = R[1]
    r20, r21, r22 = R[2]

    # Build KITTI 3x4 matrix:
    #   Remap axes:
    #     KITTI X (forward)  <- Tartan Z
    #     KITTI Y (left)     <- Tartan X (negated)
    #     KITTI Z (up)       <- Tartan Y (negated)
    kitti_pose = np.array(
        [
            # [r22, -r20, -r21, z],  # X-row
            # [-r02, r00, r01, -x],  # Y-row
            # [-r12, r10, r11, -y],  # Z-row
            [r22, -r20, -r21, x],  # X-row
            [-r02, r00, r01, y],  # Y-row
            [-r12, r10, r11, z],  # Z-row
        ]
    )

    return kitti_pose


def extract_fp(cvimage1, cvimage2, max_points=100):
    """
    Extract matching feature points using ORB.
    
    Args:
        cvimage1: First grayscale image
        cvimage2: Second grayscale image
        max_points: Maximum number of feature matches to extract
        
    Returns:
        pt1: Feature points in first image [N, 2] (pixel coordinates)
        pt2: Feature points in second image [N, 2] (pixel coordinates)
    """
    orb = cv2.ORB_create(nfeatures=500)
    
    k1, d1 = orb.detectAndCompute(cvimage1, None)
    k2, d2 = orb.detectAndCompute(cvimage2, None)
    
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return np.array([]), np.array([])
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    good_matches = good_matches[:max_points]
    
    if len(good_matches) < 4:
        return np.array([]), np.array([])
    
    pt1 = np.array([k1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    pt2 = np.array([k2[m.trainIdx].pt for m in good_matches], dtype=np.float32)
    
    return pt1, pt2


def rotation_to_euler(M, cy_thresh=None, seq="zyx"):
    """
    Taken From: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
    Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
    threshold below which to give up on straightforward arctan for
    estimating x rotation.  If None (default), estimate from
    precision of input.
    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
    Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
    [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
    [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
    [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
    z = atan2(-r12, r11)
    y = asin(r13)
    x = atan2(-r23, r33)
    for x,y,z order
    y = asin(-r31)
    x = atan2(r32, r33)
    z = atan2(r21, r11)
    Problems arise when cos(y) is close to zero, because both of::
    z = atan2(cos(y)*sin(z), cos(y)*cos(z))
    x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    """
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = np.finfo(float).eps * 4.0  # _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = np.sqrt(r33 * r33 + r23 * r23)
    if seq == "zyx":
        if cy > cy_thresh:  # cos(y) not close to zero, standard form
            z = np.arctan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = np.arctan2(r13, cy)  # atan2(sin(y), cy)
            x = np.arctan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = np.arctan2(r21, r22)
            y = np.arctan2(r13, cy)  # atan2(sin(y), cy)
            x = 0.0
    elif seq == "xyz":
        if cy > cy_thresh:
            y = np.arctan2(-r31, cy)
            x = np.arctan2(r32, r33)
            z = np.arctan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi / 2
                x = np.arctan2(r12, r13)
            else:
                y = -np.pi / 2
    else:
        raise Exception("Sequence not recognized")
    return [z, y, x]


def euler_to_rotation(z=0, y=0, x=0, isRadian=True, seq="zyx"):
    """Return matrix for rotations around z, y and x axes
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    M : array shape (3,3)
         Rotation matrix giving same rotation as for given angles
    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True
    The output rotation matrix is equal to the composition of the
    individual rotations
    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True
    You can specify rotations by named arguments
    >>> np.all(M3 == euler2mat(x=xrot))
    True
    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.
    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)
    Rotations are counter-clockwise.
    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True
    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    """

    if seq != "xyz" and seq != "zyx":
        raise Exception("Sequence not recognized")

    if not isRadian:
        z = ((np.pi) / 180.0) * z
        y = ((np.pi) / 180.0) * y
        x = ((np.pi) / 180.0) * x
    if z < -np.pi:
        while z < -np.pi:
            z += 2 * np.pi
    if z > np.pi:
        while z > np.pi:
            z -= 2 * np.pi
    if y < -np.pi:
        while y < -np.pi:
            y += 2 * np.pi
    if y > np.pi:
        while y > np.pi:
            y -= 2 * np.pi
    if x < -np.pi:
        while x < -np.pi:
            x += 2 * np.pi
    if x > np.pi:
        while x > np.pi:
            x -= 2 * np.pi
    assert z >= (-np.pi) and z < np.pi, "Inappropriate z: %f" % z
    assert y >= (-np.pi) and y < np.pi, "Inappropriate y: %f" % y
    assert x >= (-np.pi) and x < np.pi, "Inappropriate x: %f" % x

    Ms = []

    if seq == "zyx":

        if z:
            cosz = np.cos(z)
            sinz = np.sin(z)
            Ms.append(np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]]))
        if y:
            cosy = np.cos(y)
            siny = np.sin(y)
            Ms.append(np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]]))
        if x:
            cosx = np.cos(x)
            sinx = np.sin(x)
            Ms.append(np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]]))
        if Ms:
            return functools.reduce(np.dot, Ms[::-1])
        return np.eye(3)

    elif seq == "xyz":

        if x:
            cosx = np.cos(x)
            sinx = np.sin(x)
            Ms.append(np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]]))
        if y:
            cosy = np.cos(y)
            siny = np.sin(y)
            Ms.append(np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]]))
        if z:
            cosz = np.cos(z)
            sinz = np.sin(z)
            Ms.append(np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]]))

        if Ms:
            return functools.reduce(np.dot, Ms[::-1])
        return np.eye(3)


def euler_to_rotation_torch(xyz, isRadian=True, seq="zyx"):
    """Return matrix for rotations around z, y and x axes
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    M : array shape (3,3)
         Rotation matrix giving same rotation as for given angles
    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True
    The output rotation matrix is equal to the composition of the
    individual rotations
    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True
    You can specify rotations by named arguments
    >>> np.all(M3 == euler2mat(x=xrot))
    True
    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.
    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)
    Rotations are counter-clockwise.
    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True
    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    """

    z, y, x = xyz
    if seq != "xyz" and seq != "zyx":
        raise Exception("Sequence not recognized")

    if not isRadian:
        z = ((torch.pi) / 180.0) * z
        y = ((torch.pi) / 180.0) * y
        x = ((torch.pi) / 180.0) * x
    if z < -torch.pi:
        while z < -torch.pi:
            z += 2 * torch.pi
    if z > torch.pi:
        while z > torch.pi:
            z -= 2 * torch.pi
    if y < -torch.pi:
        while y < -torch.pi:
            y += 2 * torch.pi
    if y > torch.pi:
        while y > torch.pi:
            y -= 2 * torch.pi
    if x < -torch.pi:
        while x < -torch.pi:
            x += 2 * torch.pi
    if x > torch.pi:
        while x > torch.pi:
            x -= 2 * torch.pi
    assert z >= (-torch.pi) and z < torch.pi, "Inappropriate z: %f" % z
    assert y >= (-torch.pi) and y < torch.pi, "Inappropriate y: %f" % y
    assert x >= (-torch.pi) and x < torch.pi, "Inappropriate x: %f" % x

    Ms = []

    if seq == "zyx":

        if z:
            cosz = torch.cos(z)
            sinz = torch.sin(z)
            Ms.append(
                torch.Tensor([[cosz, -sinz, 0.0], [sinz, cosz, 0.0], [0.0, 0.0, 1.0]])
            )
        if y:
            cosy = torch.cos(y)
            siny = torch.sin(y)
            Ms.append(torch.Tensor([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]]))
        if x:
            cosx = torch.cos(x)
            sinx = torch.sin(x)
            Ms.append(torch.Tensor([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]]))
        if Ms:
            return functools.reduce(torch.matmul, Ms[::-1])
        return torch.eye(3)

    elif seq == "xyz":

        if x:
            cosx = torch.cos(x)
            sinx = torch.sin(x)
            Ms.append(torch.Tensor([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]]))
        if y:
            cosy = torch.cos(y)
            siny = torch.sin(y)
            Ms.append(torch.Tensor([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]]))
        if z:
            cosz = torch.cos(z)
            sinz = torch.sin(z)
            Ms.append(torch.Tensor([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]]))

        if Ms:
            return functools.reduce(torch.dot, Ms[::-1])
        return torch.eye(3)


def matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        quaternion: [qw, qx, qy, qz] (scalar first)
    """
    # Use scipy's Rotation class
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    # Convert to [w, x, y, z]
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def quaternion_to_matrix(q):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: [qw, qx, qy, qz] (scalar first)
        
    Returns:
        R: 3x3 rotation matrix
    """
    # Convert from [w, x, y, z] to [x, y, z, w]
    quat = np.array([q[1], q[2], q[3], q[0]])
    rot = Rotation.from_quat(quat)
    return rot.as_matrix()


def pose_to_dual_quaternion(pose):
    """
    Convert KITTI pose (4x4 transformation matrix) to dual quaternion (7 elements).
    
    Args:
        pose: 4x4 transformation matrix
        
    Returns:
        dual_quat: 7-element array [qw, qx, qy, qz, tx, ty, tz]
    """
    # Extract rotation matrix and translation
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    # Convert rotation matrix to quaternion
    q = matrix_to_quaternion(R)  # [qw, qx, qy, qz]
    
    # For dual quaternion, we need to represent translation as a quaternion
    # The standard representation is: [qw, qx, qy, qz, tx, ty, tz]
    # where qw, qx, qy, qz are the rotation quaternion
    # and tx, ty, tz are the translation components
    
    # However, for a proper dual quaternion representation, we need to combine
    # rotation and translation into a dual quaternion. The common approach is:
    # dq = q + 0.5 * ε * t * q
    # where ε is the dual unit (ε² = 0)
    
    # For simplicity, we'll use a 7-element representation:
    # [qw, qx, qy, qz, tx, ty, tz]
    # where the first 4 elements are the rotation quaternion
    # and the last 3 elements are the translation
    
    dual_quat = np.concatenate([q, t])
    return dual_quat


def dual_quaternion_to_pose(dual_quat):
    """
    Convert dual quaternion (7 elements) to KITTI pose (4x4 transformation matrix).
    
    Args:
        dual_quat: 7-element array [qw, qx, qy, qz, tx, ty, tz]
        
    Returns:
        pose: 4x4 transformation matrix
    """
    # Extract rotation quaternion and translation
    q = dual_quat[:4]  # [qw, qx, qy, qz]
    t = dual_quat[4:]  # [tx, ty, tz]
    
    # Convert quaternion to rotation matrix
    R = quaternion_to_matrix(q)
    
    # Construct 4x4 transformation matrix
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    
    return pose


def quaternion_to_euler(q):
    """
    Convert quaternion to Euler angles (ZYX).
    
    Args:
        q: [qw, qx, qy, qz] (scalar first)
        
    Returns:
        angles: [z, y, x] in radians
    """
    # Convert from [w, x, y, z] to [x, y, z, w]
    quat = np.array([q[1], q[2], q[3], q[0]])
    rot = Rotation.from_quat(quat)
    # Get Euler angles in ZYX order
    euler = rot.as_euler('zyx', degrees=False)
    return euler  # [z, y, x]


def euler_to_quaternion(angles):
    """
    Convert Euler angles (ZYX) to quaternion.
    
    Args:
        angles: [z, y, x] in radians
        
    Returns:
        q: [qw, qx, qy, qz] (scalar first)
    """
    # Convert Euler angles to rotation matrix
    R = euler_to_rotation(angles[0], angles[1], angles[2], seq='zyx')
    # Convert rotation matrix to quaternion
    return matrix_to_quaternion(R)


def quaternion_loss(pred_quat, target_quat):
    """
    Compute quaternion loss between predicted and target quaternions.
    
    Args:
        pred_quat: predicted quaternion [qw, qx, qy, qz]
        target_quat: target quaternion [qw, qx, qy, qz]
        
    Returns:
        loss: quaternion loss
    """
    # Ensure unit quaternions
    pred_quat = pred_quat / torch.norm(pred_quat, dim=-1, keepdim=True)
    target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)
    
    # Compute dot product
    dot = torch.sum(pred_quat * target_quat, dim=-1)
    
    # Ensure dot product is in [-1, 1]
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Compute angular distance
    angle = 2 * torch.acos(torch.abs(dot))
    
    # Return squared angular distance
    return angle ** 2


def dual_quaternion_loss(pred_dq, target_dq):
    """
    Compute dual quaternion loss between predicted and target dual quaternions.
    
    Args:
        pred_dq: predicted dual quaternion [qw, qx, qy, qz, tx, ty, tz]
        target_dq: target dual quaternion [qw, qx, qy, qz, tx, ty, tz]
        
    Returns:
        loss: dual quaternion loss
    """
    # Extract rotation quaternions and translations
    pred_q = pred_dq[:, :4]
    target_q = target_dq[:, :4]
    pred_t = pred_dq[:, 4:]
    target_t = target_dq[:, 4:]
    
    # Compute quaternion loss for rotation
    rot_loss = quaternion_loss(pred_q, target_q)
    
    # Compute translation loss (MSE)
    trans_loss = torch.mean((pred_t - target_t) ** 2, dim=-1)
    
    # Combine losses
    total_loss = rot_loss + trans_loss
    
    return total_loss


def kitti_pose_to_dual_quaternion(pose):
    """
    Convert KITTI pose (12-element array) to dual quaternion (7 elements).
    
    Args:
        pose: 12-element array representing 3x4 transformation matrix
        
    Returns:
        dual_quat: 7-element array [qw, qx, qy, qz, tx, ty, tz]
    """
    # Reshape to 4x4 matrix
    pose_matrix = np.vstack([np.reshape(pose, (3, 4)), [[0.0, 0.0, 0.0, 1.0]]])
    return pose_to_dual_quaternion(pose_matrix)


def dual_quaternion_to_kitti_pose(dual_quat):
    """
    Convert dual quaternion (7 elements) to KITTI pose (12-element array).
    
    Args:
        dual_quat: 7-element array [qw, qx, qy, qz, tx, ty, tz]
        
    Returns:
        pose: 12-element array representing 3x4 transformation matrix
    """
    pose_matrix = dual_quaternion_to_pose(dual_quat)
    # Extract 3x4 transformation matrix
    return pose_matrix[:3, :4].flatten()
