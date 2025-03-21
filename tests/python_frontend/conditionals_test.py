# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def simple_condition(i: dace.int32):
    return i % 2 == 0


@dace.program
def simple_condition2(fib: dace.int32, F: dace.int32, i: dace.int32, N: dace.int32):
    return fib < F and i < N


@dace.program
def simple_if(A: dace.int32[10]):
    for i in range(10):
        if i % 2 == 0:
            A[i] += 2 * i
        else:
            A[i] += 3 * i


def test_simple_if():
    A = np.random.randint(1, 10, size=(10, ), dtype=np.int32)
    ref = np.copy(A)
    for i in range(10):
        if i % 2 == 0:
            ref[i] += 2 * i
        else:
            ref[i] += 3 * i
    simple_if(A)
    assert (np.array_equal(A, ref))


@dace.program
def call_if(A: dace.int32[10]):
    for i in range(10):
        if simple_condition(i):
            A[i] += 2 * i
        else:
            A[i] += 3 * i


def test_call_if():
    A = np.random.randint(1, 10, size=(10, ), dtype=np.int32)
    ref = np.copy(A)
    for i in range(10):
        if i % 2 == 0:
            ref[i] += 2 * i
        else:
            ref[i] += 3 * i
    sdfg = call_if.to_sdfg()
    call_if(A)
    assert (np.array_equal(A, ref))


@dace.program
def call_if2(A: dace.int32[10]):
    A[0] = 0
    i = np.int32(1)
    fib = np.int32(1)
    while True:
        if simple_condition2(fib, 50, i, 10):
            A[i] = fib
            fib += A[i]
            i += 1
        else:
            break


def test_call_if2():
    A = np.random.randint(1, 10, size=(10, ), dtype=np.int32)
    ref = np.copy(A)
    ref[0] = 0
    i = 1
    fib = 1
    while fib < 50 and i < 10:
        ref[i] = fib
        fib += ref[i]
        i += 1
    call_if2(A)
    assert (np.array_equal(A, ref))


@dace.program
def simple_while(A: dace.int32[10]):
    i = 0
    while i < 10:
        A[i] += 2 * i
        i += 1


def test_simple_while():
    A = np.random.randint(1, 10, size=(10, ), dtype=np.int32)
    ref = np.copy(A)
    for i in range(10):
        ref[i] += 2 * i
    simple_while(A)
    assert (np.array_equal(A, ref))


@dace.program
def call_while(A: dace.int32[10]):
    A[0] = 0
    i = np.int32(1)
    fib = np.int32(1)
    while simple_condition2(fib, 50, i, 10):
        A[i] = fib
        fib += A[i]
        i += 1


def test_call_while():
    A = np.random.randint(1, 10, size=(10, ), dtype=np.int32)
    ref = np.copy(A)
    ref[0] = 0
    i = 1
    fib = 1
    while fib < 50 and i < 10:
        ref[i] = fib
        fib += ref[i]
        i += 1
    call_while(A)
    assert (np.array_equal(A, ref))


def test_if_return_both():

    @dace.program
    def if_return_both(i: dace.int64):
        if i < 5:
            return 0
        else:
            return 1
        return 2

    assert if_return_both(4)[0] == 0
    assert if_return_both(7)[0] == 1


def test_if_return_chain():

    @dace.program
    def if_return_chain(i: dace.int64):
        if i < 2:
            return 0
        if i < 4:
            return 1
        if i < 6:
            return 2
        if i < 8:
            return 3
        return 4

    assert if_return_chain(0)[0] == 0
    assert if_return_chain(2)[0] == 1
    assert if_return_chain(4)[0] == 2
    assert if_return_chain(7)[0] == 3
    assert if_return_chain(15)[0] == 4


def test_if_test_call():

    @dace.program
    def if_test_call(a, b):
        if bool(a):
            return a
        else:
            return b

    assert if_test_call(0, 2)[0] == if_test_call.f(0, 2)
    assert if_test_call(1, 2)[0] == if_test_call.f(1, 2)


if __name__ == "__main__":
    test_simple_if()
    test_call_if()
    test_call_if2()
    test_simple_while()
    test_call_while()
    test_if_return_both()
    test_if_return_chain()
    test_if_test_call()
