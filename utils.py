import os
import numpy as np
from scipy.special import bernoulli
import pandas as pd
import subprocess

def dec2str(val):
    return str(val).replace('.', '_')


def add_noise(noise_stddev, input_folder, output_folder):
    # set random seed for reproducibility
    np.random.seed(7)

    # create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # process each CSV file in the input folder
    k = 0
    while True:
        # construct full file path
        filename = "Hokuyo_" + str(k) + ".csv"
        filepath = os.path.join(input_folder, filename)
        if not os.path.exists(filepath):
            break
        # read CSV data
        df = pd.read_csv(filepath)
        
        # Add Gaussian noise to x, y, z columns
        df['x'] += np.random.normal(0, noise_stddev, df.shape[0])
        df['y'] += np.random.normal(0, noise_stddev, df.shape[0])
        df['z'] += np.random.normal(0, noise_stddev, df.shape[0])
        
        # Write noisy data to the output folder
        df.to_csv(os.path.join(output_folder, filename), index=False)
        k += 1


class Param:
    # Monte-Carlo runs for computed pseudo ground-truth covariance
    
    n_mc = 30
    # dir_path = "/home/parallels/Desktop/idp/"
    dir_path = "/storage/user/qin/idp/"
    path_sequence_base = os.path.join(dir_path, 'data')
    path_pc = None
    
    # sensor noise, then init uncertainty
    # naming: 0.01 -> 0_01
    results_base = os.path.join(dir_path, "results")
    results_path = None
    results_pert = None
    
    # lpm_path = "/home/parallels/Desktop/idp/libpointmatcher/" # libpointmatcher path
    lpm_path = "/usr/stud/qin/Desktop/idp/libpointmatcher/" # libpointmatcher path
    config_yaml = os.path.join(lpm_path, 'martin', 'config', "base_config.yaml")
    
    cov_std_pos = 0.2/np.sqrt(3)  # standard deviation of T_odo, translation
    cov_std_rot = 10/(180*np.sqrt(3))*np.pi  # standard deviation of T_odo, rot
    xi = None
    
    # features, can alter
    sensor_noise = 0.0
    init_unc = 1.0
    
    @classmethod
    def update(cls):
        """
        Called when any feature changes to update associated variables.
        """
        # cls.results_path = os.path.join(cls.results_base, dec2str(0.0), dec2str(1.0))
        # cls.results_pert = os.path.join(cls.results_base, dec2str(cls.sensor_noise), dec2str(cls.init_unc))
        
        if cls.sensor_noise >= 0.0:
            cls.path_pc = os.path.join("local_frame", dec2str(cls.sensor_noise))
        else: # sensor_noise < 0
            raise Exception("Sensor noise should be nonnegative!")
        cls.xi = np.zeros((cls.n_mc, 6))
        np.random.seed(42)
        for i in range(cls.n_mc):
            cls.xi[i] = np.hstack((np.random.normal(0, cls.init_unc*cls.cov_std_rot, 3),
                                   np.random.normal(0, cls.init_unc*cls.cov_std_pos, 3)))



class SO3:
    @staticmethod
    def normalize(Rot):
        # The SVD is commonly written as a = U S V.H.
        # The v returned by this function is V.H and u = U.
        U, _, V = np.linalg.svd(Rot, full_matrices=False)

        S = np.eye(3)
        S[2, 2] = np.linalg.det(U) * np.linalg.det(V)
        return U.dot(S).dot(V)

    @staticmethod
    def skew(x):
        X = np.array([[0., -x[2], x[1]],
                   [x[2], 0., -x[0]],
                   [-x[1], x[0], 0.]])
        return X

    @staticmethod
    def to_rpy(Rot):
        """Convert a rotation matrix to RPY Euler angles :math:`(\\alpha, \\beta, \\gamma)`."""
        pitch = np.arctan2(-Rot[2, 0],
                           np.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))

        if np.isclose(pitch, np.pi / 2.):
            yaw = 0.
            roll = np.arctan2(Rot[0, 1], Rot[1, 1])
        elif np.isclose(pitch, -np.pi / 2.):
            yaw = 0.
            roll = -np.arctan2(Rot[0, 1], Rot[1, 1])
        else:
            sec_pitch = 1. / np.cos(pitch)
            yaw = np.arctan2(Rot[1, 0] * sec_pitch,
                             Rot[0, 0] * sec_pitch)
            roll = np.arctan2(Rot[2, 1] * sec_pitch,
                              Rot[2, 2] * sec_pitch)
        return roll, pitch, yaw

    @staticmethod
    def rot2rpy(Rot):
        if Rot.ndim == 3:
            ang = np.zeros((Rot.shape[0], 3))
            for i in range(Rot.shape[0]):
                ang[i] = SO3.to_rpy(Rot[i])
        else:
            ang = SO3.to_rpy(Rot)
        return ang

    @staticmethod
    def vee(Rot):
        phi = np.array([Rot[2, 1],
                Rot[0, 2],
                Rot[1, 0]])
        return phi

    @staticmethod
    def exp(phi):
        angle = np.linalg.norm(phi)

        # Near |phi|==0, use first order Taylor expansion
        if np.abs(angle) < 1e-8:
            skew_phi = np.array([[0, -phi[2], phi[1]],
                                 [phi[2], 0, -phi[0]],
                                 [-phi[1], phi[0], 0]])
            Rot = np.eye(3) + skew_phi
        else:
            axis = phi / angle
            skew_axis = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
            s = np.sin(angle)
            c = np.cos(angle)
            Rot = c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * skew_axis

        return Rot

    @staticmethod
    def inv_left_jacobian(phi):
        angle = np.linalg.norm(phi)

        # Near phi==0, use first order Taylor expansion
        if np.isclose(angle, 0.):
            return np.identity(3) - 0.5 * SO3.skew(phi)

        axis = phi / angle
        half_angle = 0.5 * angle
        cot_half_angle = 1. / np.tan(half_angle)

        return half_angle * cot_half_angle * np.identity(3) + \
            (1 - half_angle * cot_half_angle) * np.outer(axis, axis) - half_angle * SO3.skew(axis)

    @staticmethod
    def log(Rot):
        # The cosine of the rotation angle is related to the trace of C
        cos_angle = 0.5 * np.trace(Rot) - 0.5
        # Clip cos(angle) to its proper domain to avoid NaNs from rounding errors
        cos_angle = np.clip(cos_angle, -1., 1.)
        angle = np.arccos(cos_angle)

        # If angle is close to zero, use first-order Taylor expansion
        if np.isclose(angle, 0.):
            return SO3.vee(Rot - np.eye(3))

        # Otherwise take the matrix logarithm and return the rotation vector
        return SO3.vee((0.5 * angle / np.sin(angle)) * (Rot - Rot.T))

    @classmethod
    def rotx(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the x-axis.

        .. math::
            \\mathbf{C}_x(\\phi) =
            \\begin{bmatrix}
                1 & 0 & 0 \\\\
                0 & \\cos \\phi & -\\sin \\phi \\\\
                0 & \\sin \\phi & \\cos \\phi
            \\end{bmatrix}
        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return np.array([[1., 0., 0.],
                             [0., c, -s],
                             [0., s,  c]])

    @classmethod
    def roty(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the y-axis.

        .. math::
            \\mathbf{C}_y(\\phi) =
            \\begin{bmatrix}
                \\cos \\phi & 0 & \\sin \\phi \\\\
                0 & 1 & 0 \\\\
                \\sin \\phi & 0 & \\cos \\phi
            \\end{bmatrix}
        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return np.array([[c,  0., s],
                             [0., 1., 0.],
                             [-s, 0., c]])

    @classmethod
    def rotz(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the z-axis.

        .. math::
            \\mathbf{C}_z(\\phi) =
            \\begin{bmatrix}
                \\cos \\phi & -\\sin \\phi & 0 \\\\
                \\sin \\phi  & \\cos \\phi & 0 \\\\
                0 & 0 & 1
            \\end{bmatrix}
        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return np.array([[c, -s,  0.],
                             [s,  c,  0.],
                             [0., 0., 1.]])

    @classmethod
    def from_rpy(cls, roll, pitch, yaw):
        """Form a rotation matrix from RPY Euler angles :math:`(\\alpha, \\beta, \\gamma)`.

        .. math::
            \\mathbf{C} = \\mathbf{C}_z(\\gamma) \\mathbf{C}_y(\\beta) \\mathbf{C}_x(\\alpha)
        """
        return cls.rotz(yaw).dot(cls.roty(pitch).dot(cls.rotx(roll)))

    @staticmethod
    def from_quaternion(quat, ordering='wxyz'):
        """Form a rotation matrix from a unit length quaternion.
        Valid orderings are 'xyzw' and 'wxyz'.
        .. math::
            \\mathbf{C} =
            \\begin{bmatrix}
                1 - 2 (y^2 + z^2) & 2 (xy - wz) & 2 (wy + xz) \\\\
                2 (wz + xy) & 1 - 2 (x^2 + z^2) & 2 (yz - wx) \\\\
                2 (xz - wy) & 2 (wx + yz) & 1 - 2 (x^2 + y^2)
            \\end{bmatrix}
        """
        if not np.isclose(np.linalg.norm(quat), 1.):
            print(quat)
            raise ValueError("Quaternion must be unit length")

        if ordering == 'xyzw':
            qx, qy, qz, qw = quat
        elif ordering == 'wxyz':
            qw, qx, qy, qz = quat
        else:
            raise ValueError(
                "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

        # Form the matrix
        qw2 = qw * qw
        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz

        R00 = 1. - 2. * (qy2 + qz2)
        R01 = 2. * (qx * qy - qw * qz)
        R02 = 2. * (qw * qy + qx * qz)

        R10 = 2. * (qw * qz + qx * qy)
        R11 = 1. - 2. * (qx2 + qz2)
        R12 = 2. * (qy * qz - qw * qx)

        R20 = 2. * (qx * qz - qw * qy)
        R21 = 2. * (qw * qx + qy * qz)
        R22 = 1. - 2. * (qx2 + qy2)

        return np.array([[R00, R01, R02],
                             [R10, R11, R12],
                             [R20, R21, R22]])

    @staticmethod
    def to_quaternion(Rot, ordering='wxyz'):
        """Convert a rotation matrix to a unit length quaternion.
           Valid orderings are 'xyzw' and 'wxyz'.
        """
        qw = 0.5 * np.sqrt(1. + Rot[0, 0] + Rot[1, 1] + Rot[2, 2])

        if np.isclose(qw, 0.):
            if Rot[0, 0] > Rot[1, 1] and Rot[0, 0] > Rot[2, 2]:
                d = 2. * np.sqrt(1. + Rot[0, 0] - Rot[1, 1] - Rot[2, 2])
                qw = (Rot[2, 1] - Rot[1, 2]) / d
                qx = 0.25 * d
                qy = (Rot[1, 0] + Rot[0, 1]) / d
                qz = (Rot[0, 2] + Rot[2, 0]) / d
            elif Rot[1, 1] > Rot[2, 2]:
                d = 2. * np.sqrt(1. + Rot[1, 1] - Rot[0, 0] - Rot[2, 2])
                qw = (Rot[0, 2] - Rot[2, 0]) / d
                qx = (Rot[1, 0] + Rot[0, 1]) / d
                qy = 0.25 * d
                qz = (Rot[2, 1] + Rot[1, 2]) / d
            else:
                d = 2. * np.sqrt(1. + Rot[2, 2] - Rot[0, 0] - Rot[1, 1])
                qw = (Rot[1, 0] - Rot[0, 1]) / d
                qx = (Rot[0, 2] + Rot[2, 0]) / d
                qy = (Rot[2, 1] + Rot[1, 2]) / d
                qz = 0.25 * d
        else:
            d = 4. * qw
            qx = (Rot[2, 1] - Rot[1, 2]) / d
            qy = (Rot[0, 2] - Rot[2, 0]) / d
            qz = (Rot[1, 0] - Rot[0, 1]) / d

        # Check ordering last
        if ordering == 'xyzw':
            quat = np.array([qx, qy, qz, qw])
        elif ordering == 'wxyz':
            quat = np.array([qw, qx, qy, qz])
        else:
            raise ValueError(
                "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

        return quat

    @staticmethod
    def cot(x):
        return np.cos(x)/np.sin(x)

    @staticmethod
    def jacInv(phi, tolerance=1e-8):
        ph = np.linalg.norm(phi)
        if ph < tolerance:
            # If the angle is small, fall back on the series representation
            invJ = SO3.vec2jacInvSeries(phi)
        else:
            axis = phi/np.linalg.norm(phi)
            ph_2 = 0.5*ph

            invJ = ph_2 * SO3.cot(ph_2)* np.eye(3) + \
                   (1 - ph_2 * SO3.cot(ph_2))* np.outer(axis, axis) - ph_2 * SO3.skew(axis)
        return invJ

    @staticmethod
    def jacInvSeries(phi, N=5):
        invJ = np.eye(3)
        pxn = np.eye(3)
        px = SO3.skew(phi)
        for n in range(N):
            pxn = pxn.dot(px/(n+1))
            invJ = invJ + bernoulli(n+1)[-1] * pxn
        return invJ


class SE3:
    @staticmethod
    def new_cov(size):
        return np.zeros((size, 6, 6))

    @staticmethod
    def new(size):
        return np.repeat(np.eye(4)[np.newaxis, :, :], size, axis=0)

    @staticmethod
    def Ad(T):
        Ad = np.zeros((6, 6))
        R = T[:3, :3]
        Ad[:3, :3] = R
        Ad[3:6, 3:6] = R
        Ad[3:6, :3] = SO3.skew(T[:3, 3]).dot(R)
        return Ad

    @staticmethod
    def exp(xi):
        phi = xi[:3]
        angle = np.linalg.norm(phi)

        # Near |phi|==0, use first order Taylor expansion
        if np.abs(angle) < 1e-8:
            skew_phi = np.array([[0, -phi[2], phi[1]],
                                 [phi[2], 0, -phi[0]],
                                 [-phi[1], phi[0], 0]])
            J = np.eye(3) + 0.5 * skew_phi
            Rot = np.eye(3) + skew_phi
        else:
            axis = phi / angle
            skew_axis = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
            s = np.sin(angle)
            c = np.cos(angle)
            J = (s / angle) * np.eye(3) \
                + (1 - s / angle) * np.outer(axis, axis) + ((1 - c) / angle) * skew_axis
            Rot = c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * skew_axis

        T = np.eye(4)
        T[:3, :3] = Rot
        T[:3, 3] = J.dot(xi[3:])
        return T

    @staticmethod
    def inv(T):
        T_inv = np.eye(4)
        T_inv[:3, :3] = T[:3, :3].T
        T_inv[:3, 3] = -T_inv[:3, :3].dot(T[:3, 3])
        return T_inv

    @staticmethod
    def mul(T1, T2):
        T = np.eye(4)
        T[:3, :3] = T1[:3, :3].dot(T2[:3, :3])
        T[:3, 3] = T1[:3, 3] + T1[:3, :3].dot(T2[:3, 3])
        return T

    @staticmethod
    def normalize(T):
        # The SVD is commonly written as a = U S V.H.
        # The v returned by this function is V.H and u = U.
        U, _, V = np.linalg.svd(T[:3, :3], full_matrices=False)

        S = np.eye(3)
        S[2, 2] = np.linalg.det(U) * np.linalg.det(V)
        T[:3, :3] = U.dot(S).dot(V)
        return T

    @staticmethod
    def log(T):
        phi = SO3.log(T[:3, :3])
        rho = SO3.inv_left_jacobian(phi).dot(T[:3, 3])
        return np.hstack([phi, rho])

    @staticmethod
    def to_rpy(T):
        return SO3.to_rpy(T[:3, :3])

    @staticmethod
    def mal_dist_rot_trans(xi, mean, cov, T=np.eye(4)):

        D = np.eye(6)
        D[3:6, 0:3] = SO3.skew(T[:3, 3])
        cov = D.dot(cov).dot(D.T)

        mal_dist = np.zeros(2)
        xi_rot = xi[:3]
        mean_rot = mean[:3]
        cov_rot = cov[:3, :3]
        xi_trans = xi[3:]
        mean_trans = mean[3:]
        cov_trans = cov[3:, 3:]
        mal_dist[0] = SE3.mal_dist(xi_rot, mean_rot, cov_rot)
        mal_dist[1] = SE3.mal_dist(xi_trans, mean_trans, cov_trans)
        return mal_dist

    @staticmethod
    def mal_dist(xi, mean, cov):
        err = xi - mean
        return err.T.dot(np.linalg.inv(cov)).dot(err)


    @staticmethod
    def jacInv(xi, tolerance=1e-8):
        phi = xi[:3]

        ph = np.linalg.norm(phi)
        if ph < tolerance:
            # If the angle is small, fall back on the series representation
            invJ = SE3.jacInvSeries(xi)
        else:
            invJsmall = SO3.jacInv(phi)
            Q = SE3.Q(xi)
            invJ = np.zeros((6, 6))
            invJ[:3, :3] = invJsmall
            invJ[3:, 3:] = invJsmall
            invJ[3:6, :3] = -invJsmall.dot(Q).dot(invJsmall)
        return invJ

    @staticmethod
    def jacInvSeries(xi, N=10):
        invJ = np.eye(6)
        pxn = np.eye(6)
        px = SE3.curlyhat(xi)
        for n in range(N):
            pxn = pxn.dot(px)/(n+1)
            invJ = invJ + bernoulli(n+1)[-1] * pxn
        return invJ

    @staticmethod
    def Q(xi):
        phi = xi[:3]
        rho = xi[3:]
        ph = np.linalg.norm(phi)
        ph2 = ph*ph
        ph3 = ph2*ph
        ph4 = ph3*ph
        ph5 = ph4*ph

        cph = np.cos(ph)
        sph = np.sin(ph)

        rx = SO3.skew(rho)
        px = SO3.skew(phi)

        t1 = 0.5 * rx
        t2 = ((ph - sph)/ph3) * (px*rx + rx*px + px*rx*px)
        m3 = ( 1 - 0.5 * ph2 - cph ) / ph4
        t3 = -m3 * (px*px*rx + rx*px*px - 3*px*rx*px)
        m4 = 0.5 * ( m3 - 3*(ph - sph - ph3/6)/ph5 )
        t4 = -m4 * (px*rx*px*px + px*px*rx*px)

        Q = t1 + t2 + t3 + t4
        return Q

    @staticmethod
    def curlyhat(xi):
        phihat = SO3.skew(xi[:3])
        veccurlyhat = np.zeros((6, 6))
        veccurlyhat[:3, :3] = phihat
        veccurlyhat[3:, 3:] = phihat
        veccurlyhat[3:6, :3] = SO3.skew(xi[3:6])
        return veccurlyhat

    @staticmethod
    def toSE2(T):
        T_SE2 = np.eye(3)
        T_SE2[:2, :2] = T[:2, :2]
        T_SE2[:2, :2] = T[:2, 3]
        return T_SE2


def kl_div(cov1, cov2):
    a = np.trace(np.linalg.inv(cov2).dot(cov1))
    b = np.log(np.linalg.det(cov2)/np.linalg.det(cov1))
    return 1/2*(a+b-cov1.shape[0])


def rot_trans_kl_div(cov1, cov2):
    cov1_rot = cov1[:3, :3]
    cov1_t = cov1[3:, 3:]
    cov2_rot = cov2[:3, :3]
    cov2_t = cov2[3:, 3:]

    kl = np.zeros(2)
    kl[0] = kl_div(cov1_rot, cov2_rot)
    kl[1] = kl_div(cov1_t, cov2_t)
    return kl


def str_T(T):
    if T.ndim == 1:
        output = '[' + str(T[0])
        for i in range(1, T.shape[0]):
            output += ',' + str(T[i])
    elif T.ndim == 2:
        output = '[' + str(T[0, 0])
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                if j == 0 and i == 0:
                    continue
                output += ',' + str(T[i, j])
    else:
        print('error')
    output += ']'
    return output


def icp_without_cov(pc_ref, pc_in, T_init, pose_path):
    initTranslation = str_T(T_init[:3, 3])
    initRotation = str_T(T_init[:3, :3])

    command = "cd " + Param.lpm_path + "build/ \n" + " " + "examples/icp_without_cov"
    command += " " + "--config" + " " + Param.config_yaml
    command += " " + "--output" + " " + pose_path
    command += " " + "--initTranslation" + " " + initTranslation
    command += " " + "--initRotation" + " " + initRotation
    command += " " + pc_ref
    command += " " + pc_in
    subprocess.run(command, shell=True)
