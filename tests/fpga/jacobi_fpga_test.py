# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.fpga_testing import xilinx_test, import_sample
from pathlib import Path
import pytest


# This kernel does not work with the Intel FPGA codegen, because it uses the
# constant systolic array index in the connector on the nested SDFG.
@pytest.mark.skip('Xilinx failure due to unresolved phi nodes, Intel FPGA failure due to systolic array index')
@xilinx_test(assert_ii_1=False)
def test_jacobi_fpga():
    jacobi = import_sample(Path("fpga") / "jacobi_fpga_systolic.py")
    return jacobi.run_jacobi(64, 512, 16, 4)


if __name__ == "__main__":
    test_jacobi_fpga(None)
