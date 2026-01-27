"""
P9 集成测试 - 部分测试（仅运行测试 2 和测试 5）

运行方式：python tests/test_integration_partial.py
"""

import sys
sys.path.insert(0, ".")

from tests.test_integration import (
    TestResult,
    test_parallel_speedup,
    test_algorithm_consistency,
)


def main():
    print("="*60)
    print("P9 集成测试 - 部分测试")
    print("="*60)
    print("运行: 测试 2 (并行加速比) + 测试 5 (算法一致性)")

    result = TestResult()

    # 测试 2: 并行加速比
    test_parallel_speedup(result, full_mode=False)

    # 测试 5: 算法一致性
    test_algorithm_consistency(result, full_mode=False)

    success = result.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
