"""Implementations of Lorenz 96 and Conway's
Game of Life on various meshes"""

import numpy as np
import scipy
import pprint as pp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def lorenz96(initial_state, nsteps):
    """
    Perform iterations of the Lorenz 96 update.

    Parameters
    ----------
    initial_state : array_like or list
        Initial state of lattice in an array of floats.
    nsteps : int
        Number of steps of Lorenz 96 to perform.

    Returns
    -------
    numpy.ndarray
         Final state of lattice in array of floats

    # >>> x = lorenz96([8.0, 8.0, 8.0], 1)
    # >>> print(x)
    array([8.0, 8.0, 8.0])
    """

    # write your code here to replace this return statement
    initial_state = np.array(initial_state, dtype=float)

    ncells = len(initial_state)  # the number of the cells
    res = np.copy(initial_state)  # inicialize the result array
    last_state = np.copy(initial_state)
    for i in range(nsteps):
        # compute the next state of all the cells

        for j in range(ncells):
            # if j == 0:
            #     res[j] = (1 / 101) * (100 * last_state[0] + (last_state[ncells - 2] - last_state[1]) *
            #                           last_state[ncells - 1] + 8)
            # elif j == 1:
            #     res[j] = (1 / 101) * (100 * last_state[1] + (last_state[ncells - 1] - last_state[2]) *
            #                           last_state[0] + 8)
            if j == ncells - 1:
                res[j] = (1 / 101) * (100 * last_state[j] + (last_state[j - 2] - last_state[0]) *
                                      last_state[j - 1] + 8)
            else:
                res[j] = (1 / 101) * (100 * last_state[j] + (last_state[j - 2] - last_state[j + 1]) *
                                      last_state[j - 1] + 8)
        last_state = res.copy()
    return res


# print(lorenz96([15, 16, 17], 10))
# x = lorenz96([15.0, 16.0, 17.0, 18.0, 5.0, 7.0], 50)
# print(x)


def life(initial_state, nsteps):
    """
    Perform iterations of Conway’s Game of Life.
    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial 2d state of grid in an array of booleans.
    nsteps : int
        Number of steps of Life to perform.
    Returns
    -------
    numpy.ndarray
         Final state of grid in array of booleans
    """

    # write your code here to replace return statement

    rows = len(initial_state) + 2
    cols = len(initial_state[0]) + 2
    pad_last = np.pad(initial_state, (1, 1), 'constant', constant_values=(0, 0))
    next_state = np.copy(pad_last)
    for i in range(nsteps):
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                nb_cnt = np.sum(pad_last[r - 1:r + 2, c - 1:c + 2]) - pad_last[r][c]
                if (nb_cnt == 3) | (pad_last[r][c] & (nb_cnt == 2)):
                    next_state[r][c] = 1
                else:
                    next_state[r][c] = 0
        pad_last = next_state.copy()
    return next_state[1:rows - 1, 1:cols - 1]
    # why????
    # return next_state[1:cols - 1][1:rows - 1]

#
# initial_state = [[True, False, False, False], [True, False, False, False], [True, False, False, False],
#                  [True, False, False, False]]
# pp.pprint(initial_state)
# print()
# nsteps = 1
# final_state = life(initial_state, nsteps)
# print("Generation" + str(nsteps) + ":")
# pp.pprint(final_state)


# initial_state = np.array(
#     [[False, False, False, False, False, False, False, False], [False, False, False, False, True, False, False, False],
#      [False, False, False, True, True, False, False, False], [False, False, False, False, False, False, False, False]])
# print(initial_state)
# print()
# nsteps = 1
# final_state = life(initial_state, nsteps)
# print("Generation" + str(nsteps) + ":")
# print(final_state)

# print(life(np.array([[True,False,False,False,True],[False,True,True,True,True],[False,True,False,False,False],[True,True,False,False,True]]),3))
def life_periodic(initial_state, nsteps):
    """
    Perform iterations of Conway's Game of Life on a doubly periodic mesh.

    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial 2d state of grid in an array of booleans.
    nsteps : int
        Number of steps of Life to perform.

    Returns
    -------

    numpy.ndarray
         Final state of grid in array of booleans
    """

    # write your code here to replace this return statement
    last_state = np.copy(initial_state)
    res_state = np.copy(initial_state)
    for i in range(nsteps):
        # compute the num of the alive cells from the 8 neighbours by rolling the matrix
        nb_cnt = sum(
            np.roll(np.roll(last_state, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0))
        # compute the next generation's state
        res_state = (nb_cnt == 3) | (last_state & (nb_cnt == 2))
        last_state = np.copy(res_state)
    return res_state


# initial_state = [[True, False, False, True], [True, False, False, True], [True, False, False, True],
#                  [True, False, False, True]]
# pp.pprint(initial_state)
# print()
# nsteps = 1
# final_state = life_periodic(initial_state, nsteps)
# print("Generation" + str(nsteps) + ":")
# pp.pprint(final_state)

# initial_state = np.array(
#     [[False, False, False, False, False, False, False, False], [False, False, False, False, True, False, False, False],
#      [False, False, False, True, True, False, False, False], [False, False, False, False, False, False, False, False]])
# print(initial_state)
# print()
# nsteps = 1
# final_state = life_periodic(initial_state, nsteps)
# print("Generation" + str(nsteps) + ":")
# print(final_state)

# print(life(np.array(
#     [[False, False, False, False, False], [False, True, True, True, False], [False, False, False, True, False],
#      [False, False, True, False, False], [False, False, False, False, False]]), 4))


def life2colour(initial_state, nsteps):
    """
    Perform iterations of Conway's Game of Life on a doubly periodic mesh.

    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial 2d state of grid in an array ints with value -1, 0, or 1.
        Values of -1 or 1 represent "on" cells of both colours. Zero
        values are "off".
    nsteps : int
        Number of steps of Life to perform.

    Returns
    -------

    numpy.ndarray
        Final state of grid in array of ints of value -1, 0, or 1.
    """

    # write your code here to replace this return statement
    tmp_state = np.copy(initial_state)
    pad_last = np.pad(tmp_state, (1, 1), 'constant', constant_values=(0, 0))
    rows = len(pad_last)
    cols = len(pad_last[0])
    next_state = np.copy(pad_last)
    for i in range(nsteps):
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                tmp = pad_last[r - 1:r + 2, c - 1:c + 2]
                nb_cnt = np.sum(abs(tmp)) - abs(pad_last[r][c])
                if pad_last[r][c] == 1:
                    positive_cnt = np.sum(tmp[tmp == 1]) - pad_last[r][c]
                    negtive_cnt = nb_cnt - positive_cnt
                else:
                    positive_cnt = np.sum(tmp[tmp == 1])
                    negtive_cnt = nb_cnt - positive_cnt
                if nb_cnt <= 1 or nb_cnt >= 4:
                    next_state[r][c] = 0
                elif nb_cnt == 2 and pad_last[r][c] == 0:
                    next_state[r][c] = 0
                elif nb_cnt == 3 and pad_last[r][c] == 0:
                    if positive_cnt >= negtive_cnt:
                        next_state[r][c] = 1
                    else:
                        next_state[r][c] = -1
                else:
                    next_state[r][c] = pad_last[r][c]
        pad_last = next_state.copy()
    return next_state[1:rows - 1, 1:cols - 1]


# a = np.ones([10, 10])
# a[0][1] = -1
# a[1][1] = -1
# a[0][0] = 0
# a[0][9] = 0
# a[9][0] = 0
# a[9][9] = 0
# # pp.pprint(a)
# pp.pprint(life2colour(a, 1))

print(life2colour(np.array([[1, 0, 1, 0, 0], [-1, 1, 0, -1, 1], [1, 0, 1, 0, -1], [1, 0, 0, -1, -1]]), 2))


# print(life2colour([[0, 0, 1, 1], [-1, 0, 0, 0], [1, -1, 0, 1], [0, 0, 0, 1]],2))
def lifepent(initial_state, nsteps):
    """
       Perform iterations of Conway's Game of Life on
       a pentagonal tessellation.

       Parameters
       ----------
       initial_state : array_like or list of lists
           Initial state of grid of pentagons.
       nsteps : int
           Number of steps of Life to perform.

       Returns
       -------

       numpy.ndarray
            Final state of tessellation.
       """
    rows = len(initial_state) + 2
    cols = len(initial_state[0]) + 2
    zero_state = initial_state[::-1]
    pad_last = np.pad(zero_state, (1, 1), 'constant', constant_values=(0, 0))
    next_state = np.copy(pad_last)
    for i in range(nsteps):
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                flag_r = r - 1
                flag_c = c - 1
                if flag_r % 2 == 1 and flag_c % 2 == 1:
                    nb_cnt = np.sum(pad_last[r - 1:r + 2, c - 1:c + 2]) - pad_last[r][c] - pad_last[r + 1][c - 1]
                elif flag_r % 2 == 1 and flag_c % 2 == 0:
                    nb_cnt = np.sum(pad_last[r - 1:r + 2, c - 1:c + 2]) - pad_last[r][c] - pad_last[r - 1][c - 1]
                elif flag_r % 2 == 0 and flag_c % 2 == 1:
                    nb_cnt = np.sum(pad_last[r - 1:r + 2, c - 1:c + 2]) - pad_last[r][c] - pad_last[r + 1][c + 1]
                else:
                    nb_cnt = np.sum(pad_last[r - 1:r + 2, c - 1:c + 2]) - pad_last[r][c] - pad_last[r - 1][c + 1]

                if nb_cnt == 2 and pad_last[r][c] == 1:
                    next_state[r][c] = 1
                elif nb_cnt == 3:
                    next_state[r][c] = 1
                elif nb_cnt == 4 and pad_last[r][c] == 0:
                    next_state[r][c] = 1
                elif nb_cnt == 6 and pad_last[r][c] == 0:
                    next_state[r][c] = 1
                else:
                    next_state[r][c] = 0
        pad_last = next_state.copy()
    res_state = next_state[::-1]
    return res_state[1:rows - 1, 1:cols - 1]


# initial_state = np.array(
#     [[True, True, False, False], [True, False, False, False], [True, True, False, False], [True, False, False, False]])
# print(lifepent(initial_state,2))
# print(lifepent(np.array([[True, True, False, False], [True, False, False, False], [True, False, False, False],
#                          [True, False, False, False]]), 3))
# print(lifepent(np.array(
#     [[True, True, False, False], [True, False, False, False], [True, True, False, False], [True, False, False, False]]),
#                5))


# pp.pprint(initial_state)
# pp.pprint(lifepent(initial_state, 1))

# Remaining routines are for plotting
print(lifepent(np.array(
    [[True, True, True, True], [True, False, False, False], [True, False, False, False], [True, False, False, False]]),
               4))


def plot_lorenz96(data, label=None):
    """
    Plot 1d array on a circle

    Parameters
    ----------
    data: arraylike
        values to be plotted
    label:
        optional label for legend.


    """

    offset = 8

    data = np.asarray(data)
    theta = 2 * np.pi * np.arange(len(data)) / len(data)

    vector = np.empty((len(data), 2))
    vector[:, 0] = (data + offset) * np.sin(theta)
    vector[:, 1] = (data + offset) * np.cos(theta)

    theta = np.linspace(0, 2 * np.pi)

    rings = np.arange(int(np.floor(min(data)) - 1),
                      int(np.ceil(max(data))) + 2)
    for ring in rings:
        plt.plot((ring + offset) * np.cos(theta),
                 (ring + offset) * np.sin(theta), 'k:')

    fig_ax = plt.gca()
    fig_ax.spines['left'].set_position(('data', 0.0))
    fig_ax.spines['bottom'].set_position(('data', 0.0))
    fig_ax.spines['right'].set_color('none')
    fig_ax.spines['top'].set_color('none')
    plt.xticks([])
    plt.yticks(rings + offset, rings)
    plt.fill(vector[:, 0], vector[:, 1],
             label=label, fill=False)
    plt.scatter(vector[:, 0], vector[:, 1], 20)


def plot_array(data, show_axis=False,
               cmap=plt.cm.get_cmap('seismic'), **kwargs):
    """Plot a 1D/2D array in an appropriate format.

    Mostly just a naive wrapper around pcolormesh.

    Parameters
    ----------

    data : array_like
        array to plot
    show_axis: bool, optional
        show axis numbers if true
    cmap : pyplot.colormap or str
        colormap

    Other Parameters
    ----------------

    **kwargs
        Additional arguments passed straight to pyplot.pcolormesh
    """
    plt.pcolormesh(1 * data[-1::-1, :], edgecolor='y',
                   vmin=-2, vmax=2, cmap=cmap, **kwargs)

    plt.axis('equal')
    if show_axis:
        plt.axis('on')
    else:
        plt.axis('off')


def plot_pent(x_0, y_0, theta_0, clr=0):
    """
    Plot a pentagram

    Parameters
    ----------
    x_0: float
        x coordinate of centre of the pentegram
    y_0: float
        y coordinate of centre of the pentegram
    theta_0: float
        angle of pentegram (in radians)
    """
    colours = ['w', 'r']
    s_1 = 1 / np.sqrt(3)
    s_2 = np.sqrt(1 / 2)

    theta = np.deg2rad(theta_0) + np.deg2rad([30, 90, 165, 240, 315, 30])
    r_pent = np.array([s_1, s_1, s_2, s_1, s_2, s_1])

    x_pent = x_0 + r_pent * np.sin(-theta)
    y_pent = y_0 + r_pent * np.cos(-theta)

    plt.fill(x_pent, y_pent, ec='k', fc=colours[clr])


def plot_pents(data):
    """
    Plot pentagrams in Cairo tesselation, coloured by value

    Parameters
    ----------
    data: arraylike
        integer array of values
    """
    plt.axis('off')
    plt.axis('equal')
    data = np.asarray(data).T
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            x_c = (row + 1) // 2 + (row // 2) * np.cos(np.pi / 6) - (col // 2) * np.sin(np.pi / 6)
            y_c = (col + 1) // 2 + (col // 2) * np.cos(np.pi / 6) + (row // 2) * np.sin(np.pi / 6)
            theta = (90 * (row % 2) * ((col + 1) % 2)
                     - 90 * (row % 2) * (col % 2) - 90 * (col % 2))
            clr = data[row, data.shape[1] - 1 - col]
            plot_pent(x_c, y_c, theta, clr=clr)
