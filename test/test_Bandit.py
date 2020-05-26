import numpy as np
import numpy.testing as npt

import Bandit

def test_Bandit_smoke():
    #Smoke_test
    obt = Bandit.Bandit_object()

def test_Bandit_object_fizz():
    #test the fizz_function
    obj = Bandit.Bandit_object()
    output = obj.fizz()

    npt.assert_equal(output, "buzz")
