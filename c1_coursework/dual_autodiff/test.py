import math
from dual import Dual

def test_dual():
    # Initialize Dual numbers
    d1 = Dual(2, 3)
    d2 = Dual(4, 5)

    # Test addition
    d_add = d1 + d2
    assert d_add == Dual(6, 8), f"Addition failed: Expected Dual(6, 8), got {d_add}"

    # Test subtraction
    d_sub = d1 - d2
    assert d_sub == Dual(-2, -2), f"Subtraction failed: Expected Dual(-2, -2), got {d_sub}"

    # Test multiplication
    d_mul = d1 * d2
    assert d_mul == Dual(8, 22), f"Multiplication failed: Expected Dual(8, 22), got {d_mul}"

    # Test division
    d_div = d1 / d2
    expected_real = 0.5
    expected_dual = (3 * 4 - 2 * 5) / (4 ** 2)
    assert math.isclose(d_div.real, expected_real, rel_tol=1e-9), f"Division failed: Expected real={expected_real}, got {d_div.real}"
    assert math.isclose(d_div.dual, expected_dual, rel_tol=1e-9), f"Division failed: Expected dual={expected_dual}, got {d_div.dual}"

    # Test trigonometric functions
    d3 = Dual(math.pi / 4, 1)

    # Test sin
    d3_sin = d3.sin()
    expected_real = math.sin(math.pi / 4)
    expected_dual = math.cos(math.pi / 4) * 1
    assert math.isclose(d3_sin.real, expected_real, rel_tol=1e-9), f"Sin failed: Expected real={expected_real}, got {d3_sin.real}"
    assert math.isclose(d3_sin.dual, expected_dual, rel_tol=1e-9), f"Sin failed: Expected dual={expected_dual}, got {d3_sin.dual}"

    # Test cos
    d3_cos = d3.cos()
    expected_real = math.cos(math.pi / 4)
    expected_dual = -math.sin(math.pi / 4) * 1
    assert math.isclose(d3_cos.real, expected_real, rel_tol=1e-9), f"Cos failed: Expected real={expected_real}, got {d3_cos.real}"
    assert math.isclose(d3_cos.dual, expected_dual, rel_tol=1e-9), f"Cos failed: Expected dual={expected_dual}, got {d3_cos.dual}"

    # Test tan
    d3_tan = d3.tan()
    expected_real = math.tan(math.pi / 4)
    expected_dual = 1 / (math.cos(math.pi / 4) ** 2) * 1
    assert math.isclose(d3_tan.real, expected_real, rel_tol=1e-9), f"Tan failed: Expected real={expected_real}, got {d3_tan.real}"
    assert math.isclose(d3_tan.dual, expected_dual, rel_tol=1e-9), f"Tan failed: Expected dual={expected_dual}, got {d3_tan.dual}"

    # Test log
    d4 = Dual(2, 1)
    d4_log = d4.log()
    expected_real = math.log(2)
    expected_dual = 1 / 2
    assert math.isclose(d4_log.real, expected_real, rel_tol=1e-9), f"Log failed: Expected real={expected_real}, got {d4_log.real}"
    assert math.isclose(d4_log.dual, expected_dual, rel_tol=1e-9), f"Log failed: Expected dual={expected_dual}, got {d4_log.dual}"

    # Test exp
    d4_exp = d4.exp()
    expected_real = math.exp(2)
    expected_dual = math.exp(2) * 1
    assert math.isclose(d4_exp.real, expected_real, rel_tol=1e-9), f"Exp failed: Expected real={expected_real}, got {d4_exp.real}"
    assert math.isclose(d4_exp.dual, expected_dual, rel_tol=1e-9), f"Exp failed: Expected dual={expected_dual}, got {d4_exp.dual}"

    print("All tests passed successfully!")

# Run the test
test_dual()

