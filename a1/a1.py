#! /usr/bin/env python3

import argparse
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


def MLKF_1dof(m1, l1, k1, f1):

    """Return mass, damping, stiffness & force matrices for 1DOF system"""

    M = np.array([[m1]])
    L = np.array([[l1]])
    K = np.array([[k1]])
    F = np.array([f1])

    return M, L, K, F


def MLKF_2dof(m1, l1, k1, f1, m2, l2, k2, f2):

    """Return mass, damping, stiffness & force matrices for 2DOF system"""

    M = np.array([[m1, 0], [0, m2]])
    L = np.array([[l1+l2, -l2], [-l2, l2]])
    K = np.array([[k1+k2, -k2], [-k2, k2]])
    F = np.array([f1, f2])

    return M, L, K, F

def MLKF_3_storey(m1, l1, k1, f1, m2, l2, k2, f2, m3, l3, k3, f3):
    """Return mass, damping, stiffness & force matrices for 3DOF system"""
    M = np.array([
        [m1, 0, 0],
        [0, m2, 0],
        [0, 0, m3]
    ])
    L = np.array([
        [l1+l2, -l2, 0],
        [-l2, l3+l2, -l3],
        [0, -l3, l3]
    ])
    K = np.array([
        [k1+k2, -k2, 0],
        [-k2, k3+k2, -k3],
        [0, -k3, k3]
    ])
    F = np.array([f1, f2, f3])

    return M, L, K, F

def MLKF_3_storey_1st_floor_damped(m1, l1, k1, f1, m2, l2, k2, f2, m3, l3, k3, f3, m4, l4, k4, f4):
    """Return mass, damping, stiffness & force matrices for 4DOF system.
    3-floor building with damper on first floor"""
    M = np.array([
        [m1, 0, 0, 0],
        [0, m2, 0, 0],
        [0, 0, m3, 0],
        [0, 0, 0, m4]
    ])
    L = np.array([
        [l1+l2+l4, -l2, 0, -l4],
        [-l2, l3+l2, -l3, 0],
        [0, -l3, l3, 0],
        [-l4, 0, 0, l4]
    ])
    K = np.array([
        [k1+k2+k4, -k2, 0, -k4],
        [-k2, k3+k2, -k3, 0],
        [0, -k3, k3, 0],
        [-k4, 0, 0, k4]
    ])
    F = np.array([f1, f2, f3, f4])

    return M, L, K, F

def MLKF_3_storey_2nd_floor_damped(m1, l1, k1, f1, m2, l2, k2, f2, m3, l3, k3, f3, m5, l5, k5, f5):
    """Return mass, damping, stiffness & force matrices for 4DOF system.
    3-floor building with damper on first floor"""
    M = np.array([
        [m1, 0, 0, 0],
        [0, m2, 0, 0],
        [0, 0, m3, 0],
        [0, 0, 0, m5]
    ])
    L = np.array([
        [l1+l2, -l2, 0, 0],
        [-l2, l3+l2+l5, -l3, -l5],
        [0, -l3, l3, 0],
        [0, -l5, 0, l5]
    ])
    K = np.array([
        [k1+k2, -k2, 0, 0],
        [-k2, k3+k2+k5, -k3, -k5],
        [0, -k3, k3, 0],
        [0, -k5, 0, k5]
    ])
    F = np.array([f1, f2, f3, f5])

    return M, L, K, F

def MLKF_3_storey_3rd_floor_damped(m1, l1, k1, f1, m2, l2, k2, f2, m3, l3, k3, f3, m6, l6, k6, f6):
    """Return mass, damping, stiffness & force matrices for 4DOF system.
    3-floor building with damper on first floor"""
    M = np.array([
        [m1, 0, 0, 0],
        [0, m2, 0, 0],
        [0, 0, m3, 0],
        [0, 0, 0, m6]
    ])
    L = np.array([
        [l1+l2, -l2, 0, 0],
        [-l2, l3+l2, -l3, 0],
        [0, -l3, l3+l6, -l6],
        [0, 0, -l6, l6]
    ])
    K = np.array([
        [k1+k2, -k2, 0, 0],
        [-k2, k3+k2, -k3, 0],
        [0, -k3, k3+k6, -k6],
        [0, 0, -k6, k6]
    ])
    F = np.array([f1, f2, f3, f6])

    return M, L, K, F

def MLKF_3_storey_all_floors_damped(m1, l1, k1, f1, m2, l2, k2, f2, m3, l3, k3, f3, m4, l4, k4, f4,
                                    m5, l5, k5, f5, m6, l6, k6, f6):
    """Return mass, damping, stiffness & force matrices for 4DOF system.
    3-floor building with damper on first floor"""
    M = np.array([
        [m1, 0, 0, 0, 0, 0],
        [0, m2, 0, 0, 0, 0],
        [0, 0, m3, 0, 0, 0],
        [0, 0, 0, m4, 0, 0],
        [0, 0, 0, 0, m5, 0],
        [0, 0, 0, 0, 0, m6]
    ])
    L = np.array([
        [l1+l2+l4, -l2, 0, -l4, 0, 0],
        [-l2, l3+l2+l5, -l3, 0, -l5, 0],
        [0, -l3, l3+l6, 0, 0, -l6],
        [-l4, 0, 0, l4, 0, 0],
        [0, -l5, 0, 0, l5, 0],
        [0, 0, -l6, 0, 0, l6]
    ])
    K = np.array([
        [k1+k2+k4, -k2, 0, -k4, 0, 0],
        [-k2, k3+k2+k5, -k3, 0, -k5, 0],
        [0, -k3, k3+k6, 0, 0, -k6],
        [-k4, 0, 0, k4, 0, 0],
        [0, -k5, 0, 0, k5, 0],
        [0, 0, -k6, 0, 0, k6]
    ])
    F = np.array([f1, f2, f3, f4, f5, f6])

    return M, L, K, F


def freq_response(w_list, M, L, K, F):

    """Return complex frequency response of system"""

    return np.array(
        [np.linalg.solve(-w*w * M + 1j * w * L + K, F) for w in w_list]
    )


def time_response(t_list, M, L, K, F):

    """Return time response of system"""

    mm = M.diagonal()

    def slope(t, y):
        xv = y.reshape((2, -1))
        a = (F - L@xv[1] - K@xv[0]) / mm
        s = np.concatenate((xv[1], a))
        return s

    solution = scipy.integrate.solve_ivp(
        fun=slope,
        t_span=(t_list[0], t_list[-1]),
        y0=np.zeros(len(mm) * 2),
        method='Radau',
        t_eval=t_list
    )

    return solution.y[0:len(mm), :].T


def last_nonzero(arr, axis, invalid_val=-1):

    """Return index of last non-zero element of an array"""

    mask = (arr != 0)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def plot(fig, hz, sec, M, L, K, F, show_phase=None):

    """Plot frequency and time domain responses"""

    # Generate response data

    f_response = freq_response(hz * 2*np.pi, M, L, K, F)
    f_amplitude = np.abs(f_response)
    t_response = time_response(sec, M, L, K, F)

    # Determine suitable legends

    f_legends = (
        'm{} peak {:.4g} metre at {:.4g} Hz'.format(
            i+1,
            f_amplitude[m][i],
            hz[m]
        )
        for i, m in enumerate(np.argmax(f_amplitude, axis=0))
    )

    equilib = np.abs(freq_response([0], M, L, K, F))[0]         # Zero Hz
    toobig = abs(100 * (t_response - equilib) / equilib) >= 2
    lastbig = last_nonzero(toobig, axis=0, invalid_val=len(sec)-1)

    t_legends = (
        'm{} settled to 2% beyond {:.4g} sec'.format(
            i+1,
            sec[lastbig[i]]
        )
        for i, _ in enumerate(t_response.T)
    )

    # Create plot

    fig.clear()

    if show_phase is not None:
        ax = [
            fig.add_subplot(3, 1, 1),
            fig.add_subplot(3, 1, 2),
            fig.add_subplot(3, 1, 3)
        ]
        ax[1].sharex(ax[0])
    else:
        ax = [
            fig.add_subplot(2, 1, 1),
            fig.add_subplot(2, 1, 2)
        ]

    ax[0].set_title('Amplitude of frequency domain response to sinusoidal force')
    ax[0].set_xlabel('Frequency/hertz')
    ax[0].set_ylabel('Amplitude/metre')
    ax[0].legend(ax[0].plot(hz, f_amplitude), f_legends)

    if show_phase is not None:
        p_legends = (f'm{i+1}' for i in range(f_response.shape[1]))

        f_phases = f_response
        if show_phase == 0:
            ax[1].set_title(f'Phase of frequency domain response to sinusoidal force')
        else:
            f_phases /= f_response[:, show_phase-1:show_phase]
            ax[1].set_title(f'Phase, relative to m{show_phase}, of frequency domain response to sinusoidal force')
        f_phases = np.degrees(np.angle(f_phases))

        ax[1].set_xlabel('Frequency/hertz')
        ax[1].set_ylabel('Phase/°')
        ax[1].legend(ax[1].plot(hz, f_phases), p_legends)

    ax[-1].set_title('Time domain response to step force')
    ax[-1].set_xlabel('Time/second')
    ax[-1].set_ylabel('Displacement/metre')
    ax[-1].legend(ax[-1].plot(sec, t_response), t_legends)

    fig.tight_layout()


def arg_parser():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='''
            For a system with one or two degrees of freedom, show the
            frequency domain response to an applied sinusoidal force,
            and the time domain response to an step force.
    ''')

    ap.add_argument('--m1', type=float, default=7.88, help='Mass 1')
    ap.add_argument('--l1', type=float, default=3.96, help='Damping 1')
    ap.add_argument('--k1', type=float, default=4200, help='Spring 1')
    ap.add_argument('--f1', type=float, default=0.25, help='Force 1')

    ap.add_argument('--m2', type=float, default=None, help='Mass 2')
    ap.add_argument('--l2', type=float, default=1, help='Damping 2')
    ap.add_argument('--k2', type=float, default=106.8, help='Spring 2')
    ap.add_argument('--f2', type=float, default=0, help='Force 2')

    ap.add_argument('--m3', type=float, default=None, help='Mass 3')
    ap.add_argument('--l3', type=float, default=1, help='Damping 3')
    ap.add_argument('--k3', type=float, default=106.8, help='Spring 3')
    ap.add_argument('--f3', type=float, default=0, help='Force 3')

    ap.add_argument('--m4', type=float, default=None, help='Mass 4')
    ap.add_argument('--l4', type=float, default=1, help='Damping 4')
    ap.add_argument('--k4', type=float, default=106.8, help='Spring 4')
    ap.add_argument('--f4', type=float, default=0, help='Force 4')

    ap.add_argument('--m5', type=float, default=None, help='Mass 5')
    ap.add_argument('--l5', type=float, default=1, help='Damping 5')
    ap.add_argument('--k5', type=float, default=106.8, help='Spring 5')
    ap.add_argument('--f5', type=float, default=0, help='Force 5')

    ap.add_argument('--m6', type=float, default=None, help='Mass 6')
    ap.add_argument('--l6', type=float, default=1, help='Damping 6')
    ap.add_argument('--k6', type=float, default=106.8, help='Spring 6')
    ap.add_argument('--f6', type=float, default=0, help='Force 6')

    ap.add_argument(
        '--hz', type=float, nargs=2, default=(0, 5),
        help='Frequency range'
    )
    ap.add_argument(
        '--sec', type=float, default=30,
        help='Time limit'
    )

    ap.add_argument(
        '--show-phase', type=int, nargs='?', const=0,
        help='''Show the frequency domain phase response(s).
        If this option is given without a value then phases are shown
        relative to the excitation.
        If a value is given then phases are shown relative to the
        phase of the mass with that number.
    ''')

    return ap


def main():

    """Main program"""

    # Read command line

    ap = arg_parser()
    args = ap.parse_args()

    # Generate matrices describing the system

    if args.m2 is None:
        M, L, K, F = MLKF_1dof(
            args.m1, args.l1, args.k1, args.f1
        )
    elif args.m3 is None:
        M, L, K, F = MLKF_2dof(
            args.m1, args.l1, args.k1, args.f1,
            args.m2, args.l2, args.k2, args.f2
        )
    elif args.m4 is None:
        M, L, K, F = MLKF_3_storey(
            args.m1, args.l1, args.k1, args.f1,
            args.m2, args.l2, args.k2, args.f2,
            args.m3, args.l3, args.k3, args.f3
        )
    elif args.m5 is None:
        M, L, K, F = MLKF_3_storey_1st_floor_damped(
            args.m1, args.l1, args.k1, args.f1,
            args.m2, args.l2, args.k2, args.f2,
            args.m3, args.l3, args.k3, args.f3,
            args.m4, args.l4, args.k4, args.f4
        )
    else:
        M, L, K, F = MLKF_3_storey_all_floors_damped(
            args.m1, args.l1, args.k1, args.f1,
            args.m2, args.l2, args.k2, args.f2,
            args.m3, args.l3, args.k3, args.f3,
            args.m4, args.l4, args.k4, args.f4,
            args.m5, args.l5, args.k5, args.f5,
            args.m6, args.l6, args.k6, args.f6
        )

    # Generate frequency and time arrays

    hz = np.linspace(args.hz[0], args.hz[1], 10001)
    sec = np.linspace(0, args.sec, 10001)

    # Plot results

    fig = plt.figure()
    plot(fig, hz, sec, M, L, K, F, args.show_phase)
    fig.canvas.mpl_connect('resize_event', lambda x: fig.tight_layout(pad=2.5))
    plt.show()


if __name__ == '__main__':
    main()
