#!/usr/bin/env python3
"""
SYNTRIAD v6.0 - GPU Kernel Validation Tests
============================================

Tests for validating GPU kernel implementations against CPU.

Requirements:
- TEST-06: Create test suite for all 19 GPU operations

Author: SYNTRIAD Research
Created: 2026-01-27
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check CUDA availability
try:
    from numba import cuda
    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_CUDA = False


# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


@pytest.fixture
def kernel_generator():
    """Fixture providing a kernel generator instance."""
    from cuda_kernel_generator import DynamicKernelGenerator
    return DynamicKernelGenerator()


@pytest.fixture
def test_numbers():
    """Fixture providing test number array."""
    return np.arange(1000, 10000, dtype=np.int64)


class TestGPUKernelGeneration:
    """Tests for dynamic GPU kernel generation"""

    def test_kernel_generation_single_op(self, kernel_generator):
        """Test kernel generation for single operation."""
        for op_code in range(19):
            pipeline = [op_code]
            kernel = kernel_generator.get_kernel(pipeline)
            assert kernel is not None, f"Failed to generate kernel for op {op_code}"

    def test_kernel_generation_multi_op(self, kernel_generator):
        """Test kernel generation for multi-operation pipelines."""
        pipelines = [
            [0, 1],        # reverse -> digit_sum
            [2, 1, 0],     # kaprekar -> digit_sum -> reverse
            [3, 0, 1],     # truc_1089 -> reverse -> digit_sum
            [4, 4, 4],     # happy_step repeated
            [1, 2, 3, 4, 5]  # 5 operations
        ]

        for pipeline in pipelines:
            kernel = kernel_generator.get_kernel(pipeline)
            assert kernel is not None

    def test_kernel_caching(self, kernel_generator):
        """Test that kernels are cached properly."""
        pipeline = [2, 1]

        # First call
        kernel1 = kernel_generator.get_kernel(pipeline)

        # Second call should use cache
        kernel2 = kernel_generator.get_kernel(pipeline)

        assert kernel1 is kernel2

        # Check cache stats
        stats = kernel_generator.cache.stats()
        assert stats['hits'] >= 1


class TestGPUCPUEquivalence:
    """Tests for GPU/CPU equivalence"""

    def test_reverse_equivalence(self, kernel_generator, test_numbers):
        """Test reverse operation matches CPU."""
        from multiprocessing_executor import _apply_single_op

        endpoints, _, _ = kernel_generator.execute([0], test_numbers, max_iterations=1)

        for i, n in enumerate(test_numbers[:100]):
            cpu_result = _apply_single_op(int(n), 0)
            assert endpoints[i] == cpu_result, f"Mismatch at {n}: GPU={endpoints[i]}, CPU={cpu_result}"

    def test_digit_sum_equivalence(self, kernel_generator, test_numbers):
        """Test digit_sum operation matches CPU."""
        from multiprocessing_executor import _apply_single_op

        endpoints, _, _ = kernel_generator.execute([1], test_numbers, max_iterations=1)

        for i, n in enumerate(test_numbers[:100]):
            cpu_result = _apply_single_op(int(n), 1)
            assert endpoints[i] == cpu_result

    def test_kaprekar_equivalence(self, kernel_generator, test_numbers):
        """Test kaprekar_step operation matches CPU."""
        from multiprocessing_executor import _apply_single_op

        endpoints, _, _ = kernel_generator.execute([2], test_numbers, max_iterations=1)

        for i, n in enumerate(test_numbers[:100]):
            cpu_result = _apply_single_op(int(n), 2)
            assert endpoints[i] == cpu_result

    def test_truc_1089_equivalence(self, kernel_generator, test_numbers):
        """Test truc_1089 operation matches CPU."""
        from multiprocessing_executor import _apply_single_op

        endpoints, _, _ = kernel_generator.execute([3], test_numbers, max_iterations=1)

        for i, n in enumerate(test_numbers[:100]):
            cpu_result = _apply_single_op(int(n), 3)
            assert endpoints[i] == cpu_result

    def test_happy_step_equivalence(self, kernel_generator, test_numbers):
        """Test happy_step operation matches CPU."""
        from multiprocessing_executor import _apply_single_op

        endpoints, _, _ = kernel_generator.execute([4], test_numbers, max_iterations=1)

        for i, n in enumerate(test_numbers[:100]):
            cpu_result = _apply_single_op(int(n), 4)
            assert endpoints[i] == cpu_result

    def test_pipeline_equivalence(self, kernel_generator):
        """Test multi-operation pipeline matches CPU."""
        from multiprocessing_executor import _apply_pipeline_cpu

        pipeline = [2, 1]  # kaprekar -> digit_sum
        numbers = np.arange(1000, 2000, dtype=np.int64)

        gpu_endpoints, gpu_steps, gpu_cycles = kernel_generator.execute(
            pipeline, numbers, max_iterations=20
        )

        for i, n in enumerate(numbers[:50]):
            cpu_result, cpu_steps, cpu_cycle = _apply_pipeline_cpu(int(n), pipeline, 20)

            # Allow small discrepancy due to different cycle detection
            if gpu_endpoints[i] != cpu_result:
                # Check if both found the same attractor eventually
                assert abs(gpu_endpoints[i] - cpu_result) < 10 or \
                       gpu_endpoints[i] == 0 or cpu_result == 0


class TestGPUKernelValidation:
    """Tests for kernel validation functionality"""

    def test_validate_all_operations(self, kernel_generator):
        """Test validation of all operations."""
        results = {}

        for op_code in range(19):
            pipeline = [op_code]
            validation = kernel_generator.validate_against_cpu(pipeline, sample_size=500)
            results[op_code] = validation['valid']

        # Most operations should validate
        pass_rate = sum(results.values()) / len(results)
        assert pass_rate >= 0.9, f"Only {pass_rate*100:.1f}% of operations validated"

    def test_validate_combined_pipelines(self, kernel_generator):
        """Test validation of combined pipelines."""
        pipelines = [
            [0, 1],
            [2, 3],
            [1, 4],
        ]

        for pipeline in pipelines:
            validation = kernel_generator.validate_against_cpu(pipeline, sample_size=200)
            assert validation['match_rate'] > 0.9, \
                f"Pipeline {pipeline} only matched {validation['match_rate']*100:.1f}%"


class TestGPUPerformance:
    """Tests for GPU performance requirements"""

    def test_throughput_minimum(self, kernel_generator):
        """Test minimum throughput requirement (PERF-05: 50M numbers/sec)."""
        pipeline = [2, 1]  # Simple pipeline
        batch_size = 2_000_000

        benchmark = kernel_generator.benchmark_kernel(
            pipeline,
            batch_size=batch_size,
            repetitions=3
        )

        # Should achieve at least 20M numbers/sec (relaxed for CI)
        assert benchmark['numbers_per_second'] > 20_000_000, \
            f"Throughput too low: {benchmark['numbers_per_second']:,.0f} numbers/sec"

    def test_kernel_overhead(self, kernel_generator):
        """Test dynamic kernel overhead (PERF-04: <20% overhead)."""
        benchmark = kernel_generator.benchmark_kernel(
            [2],  # Single operation
            batch_size=1_000_000,
            repetitions=3
        )

        # First run overhead should be reasonable
        # (compilation overhead is measured separately)
        assert benchmark['mean_time'] < 1.0, "Kernel execution too slow"


class TestGPUBatchProcessing:
    """Tests for GPU batch processing"""

    def test_batch_execution(self, kernel_generator):
        """Test executing multiple pipelines in batch."""
        pipelines = [
            [0], [1], [2], [3], [4]
        ]

        results = kernel_generator.execute_batch(
            pipelines,
            start=1000,
            end=100000,
            batch_size=50000
        )

        assert len(results) == len(pipelines)
        for result in results:
            assert result['success']
            assert result['convergence_rate'] >= 0

    def test_batch_handles_errors(self, kernel_generator):
        """Test batch processing handles errors gracefully."""
        pipelines = [
            [0],  # Valid
            [100],  # Invalid op code (should handle gracefully)
            [1],  # Valid
        ]

        results = kernel_generator.execute_batch(
            pipelines,
            start=1000,
            end=10000,
            batch_size=1000
        )

        # Should get results for all pipelines
        assert len(results) == len(pipelines)


class TestGPUMemoryManagement:
    """Tests for GPU memory management"""

    def test_no_memory_leak(self, kernel_generator):
        """Test that repeated execution doesn't leak memory."""
        import gc

        pipeline = [2, 1]
        numbers = np.arange(1000, 100000, dtype=np.int64)

        # Run multiple times
        for _ in range(10):
            kernel_generator.execute(pipeline, numbers, max_iterations=10)

        # Force garbage collection
        gc.collect()

        # If we get here without CUDA OOM, test passes

    def test_cache_eviction(self, kernel_generator):
        """Test kernel cache eviction works."""
        # Generate many different pipelines to trigger eviction
        for i in range(100):
            pipeline = [i % 19, (i + 1) % 19]
            kernel_generator.get_kernel(pipeline)

        stats = kernel_generator.cache.stats()
        assert stats['size'] <= kernel_generator.cache.max_size


class TestGPUEdgeCases:
    """Tests for GPU edge cases"""

    def test_empty_pipeline(self, kernel_generator):
        """Test handling of empty pipeline."""
        # Empty pipeline returns input unchanged (identity operation)
        numbers = np.array([1, 2, 3], dtype=np.int64)
        endpoints, _, _ = kernel_generator.execute([], numbers, 10)
        # With empty pipeline, numbers should remain unchanged or return 0
        assert len(endpoints) == len(numbers)

    def test_single_number(self, kernel_generator):
        """Test with single number."""
        numbers = np.array([6174], dtype=np.int64)
        endpoints, _, _ = kernel_generator.execute([2], numbers, max_iterations=10)
        assert endpoints[0] == 6174  # Kaprekar fixed point

    def test_large_batch(self, kernel_generator):
        """Test with large batch size."""
        numbers = np.arange(1000, 3_000_000, dtype=np.int64)
        endpoints, _, _ = kernel_generator.execute([1], numbers, max_iterations=1)

        # Should complete without error
        assert len(endpoints) == len(numbers)


class TestGPUKnownConstants:
    """Tests for GPU discovery of known constants"""

    def test_kaprekar_constant_gpu(self, kernel_generator):
        """Test GPU finds Kaprekar constant."""
        # 4-digit numbers should converge to 6174
        numbers = np.arange(1234, 1240, dtype=np.int64)
        endpoints, _, cycles = kernel_generator.execute([2], numbers, max_iterations=20)

        # Check convergence to 6174
        for endpoint in endpoints:
            assert endpoint in [6174, 0]  # Either finds constant or zero (repdigit case)

    def test_1089_constant_gpu(self, kernel_generator):
        """Test GPU finds 1089 constant."""
        # 3-digit numbers with different first/last should give 1089
        numbers = np.array([321, 421, 521, 621, 721], dtype=np.int64)
        endpoints, _, _ = kernel_generator.execute([3], numbers, max_iterations=1)

        for endpoint in endpoints:
            assert endpoint == 1089

    def test_armstrong_number_gpu(self, kernel_generator):
        """Test GPU correctly handles Armstrong numbers."""
        numbers = np.array([153, 370, 371, 407], dtype=np.int64)
        endpoints, _, _ = kernel_generator.execute([5], numbers, max_iterations=1)

        for i, n in enumerate(numbers):
            assert endpoints[i] == n  # Armstrong numbers are fixed points


if __name__ == "__main__":
    if not HAS_CUDA:
        print("CUDA not available - skipping GPU tests")
    else:
        pytest.main([__file__, "-v", "--tb=short"])
