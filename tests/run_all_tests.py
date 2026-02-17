"""
Run all verification tests for the Selector Complexity Framework.

Each test is a self-contained proof that outputs PASS/FAIL.

Author: Carmen Esteban
Date: February 2026
"""

import subprocess
import sys
import os
import time


def run_script(path, name):
    """Run a Python script and capture output."""
    print("\n" + "=" * 60)
    print("RUNNING: {}".format(name))
    print("=" * 60)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True, timeout=600
    )
    elapsed = time.time() - t0

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])

    passed = result.returncode == 0
    return passed, elapsed


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    theory_dir = os.path.join(base, "theory")

    tests = [
        ("01_definitions.py", "Formal Definitions"),
        ("02_php_level0.py", "PHP is Level 0"),
        ("03_phpe_level1.py", "PHP-E is Level 1"),
        ("04_phpc_level2_conjecture.py", "PHP-C Level 2+ Evidence"),
    ]

    results = []
    for filename, name in tests:
        path = os.path.join(theory_dir, filename)
        if os.path.exists(path):
            try:
                passed, elapsed = run_script(path, name)
                results.append((name, passed, elapsed))
            except subprocess.TimeoutExpired:
                print("  TIMEOUT after 600s")
                results.append((name, False, 600))
        else:
            print("  FILE NOT FOUND: {}".format(path))
            results.append((name, False, 0))

    # Summary
    print("\n\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print()
    all_pass = True
    for name, passed, elapsed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print("  [{}] {} ({:.1f}s)".format(status, name, elapsed))

    print()
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
