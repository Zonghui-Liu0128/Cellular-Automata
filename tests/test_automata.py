"""Test automata module functions."""

import os

import numpy as np

import automata

BASE_PATH = os.path.dirname(__file__)


def test_lorenz96():
    """Test Lorenz 96 implementation"""
    initial64 = np.load(os.sep.join((BASE_PATH,
                                     'lorenz96_64_init.npy')))

    onestep64 = np.load(os.sep.join((BASE_PATH,
                                     'lorenz96_64_onestep.npy')))
    assert np.isclose(automata.lorenz96(initial64, 1), onestep64).all()

    thirtystep64 = np.load(os.sep.join((BASE_PATH,
                                        'lorenz96_64_thirtystep.npy')))
    assert np.isclose(automata.lorenz96(initial64, 30), thirtystep64).all()


def test_life():
    initial = np.asarray([
        [False, False, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, False, False]])

    assert ((automata.life(initial, 1) == np.asarray([
        [False, False, False],
        [False, False, False],
        [True, True, True],
        [False, False, False],
        [False, False, False], ])).all())


def test_life_periodic():
    initial = np.asarray([
        [False, False, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, False, False]])

    assert ((automata.life_periodic(initial, 2) ==
             np.asarray([[False, False, False],
                         [True, True, True],
                         [True, True, True],
                         [True, True, True],
                         [False, False, False]])).all())


def test_life2colour():
    initial = np.asarray([[1, 0, 1, 0, 0], [-1, 1, 0, -1, 1], [1, 0, 1, 0, -1], [1, 0, 0, -1, -1]])

    assert ((automata.life2colour(initial, 2) == np.asarray([[0, 1, 0, 1, 0], [-1, 0, 1, 0, 0], [1, 0, 1, 0, -1], [0, 1, 1, -1, 0]])).all())



def test_lifepent():
    initial = np.asarray([[True, True, False, False], [True, False, False, False], [True, True, False, False],
                          [True, False, False, False]])
    assert ((automata.lifepent(initial, 2) == np.asarray(
        [[True, True, False, False], [True, False, False, False], [False, True, False, False],
         [True, True, False, False]])).all())
