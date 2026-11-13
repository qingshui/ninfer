"""Configure-stage checks for the B200 (sm_100a) platform target and presets.

Host-only. The platform layer is exercised through the real
cmake/NinferPlatform.cmake inside a minimal wrapper project, so the checks do
not depend on per-op ``${NINFER_TARGET}/sources.cmake`` files; the full
``cmake --preset b200-release`` configure and build is covered by the feature
acceptance once every op registers its b200 sources.
"""

from __future__ import annotations

import json
import platform as host_platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORM_CMAKE = ROOT / "cmake" / "NinferPlatform.cmake"
PLATFORM_H_IN = ROOT / "src" / "core" / "platform.h.in"
PRESETS = ROOT / "CMakePresets.json"

WRAPPER_CMAKELISTS = """\
cmake_minimum_required(VERSION 3.28)
include("{platform_cmake}")
project(ninfer_platform_probe LANGUAGES NONE)
ninfer_validate_platform()
"""

AARCH64_TOOLCHAIN = """\
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
"""

CAPABILITY_PROBE = """\
#include "core/platform.h"
static_assert(NINFER_TARGET_B200 == 1, "b200 target flag");
static_assert(NINFER_SHARED_PHYSICAL_MEMORY == 0, "discrete GPU memory");
static_assert(NINFER_EXPERT_OFFLOAD == 0, "weights fit in HBM");
int main() { return 0; }
"""

# Pre-existing rtx5090/dgx-spark preset definitions; b200 must not alter them.
UNTOUCHED_CONFIGURE_PRESETS = {
    "rtx5090-release": {
        "name": "rtx5090-release",
        "displayName": "RTX 5090 product binaries",
        "inherits": "product",
        "cacheVariables": {"NINFER_TARGET": "rtx5090"},
    },
    "rtx5090-dev": {
        "name": "rtx5090-dev",
        "displayName": "RTX 5090 products, tests and benchmarks",
        "inherits": "rtx5090-release",
        "cacheVariables": {"BUILD_TESTING": "ON", "NINFER_BUILD_BENCHMARKS": "ON"},
    },
    "rtx5090-ops": {
        "name": "rtx5090-ops",
        "displayName": "RTX 5090 Core, Artifact and Ops development",
        "inherits": "rtx5090-dev",
        "cacheVariables": {"NINFER_BUILD_ENGINE": "OFF", "NINFER_BUILD_APPS": "OFF"},
    },
    "dgx-spark-release": {
        "name": "dgx-spark-release",
        "displayName": "DGX Spark product binaries",
        "inherits": "product",
        "cacheVariables": {"NINFER_TARGET": "dgx_spark"},
    },
    "dgx-spark-dev": {
        "name": "dgx-spark-dev",
        "displayName": "DGX Spark products, tests and benchmarks",
        "inherits": "dgx-spark-release",
        "cacheVariables": {"BUILD_TESTING": "ON", "NINFER_BUILD_BENCHMARKS": "ON"},
    },
    "dgx-spark-ops": {
        "name": "dgx-spark-ops",
        "displayName": "DGX Spark Core, Artifact and Ops development",
        "inherits": "dgx-spark-dev",
        "cacheVariables": {"NINFER_BUILD_ENGINE": "OFF", "NINFER_BUILD_APPS": "OFF"},
    },
}

B200_CONFIGURE_PRESETS = {
    "b200-release": {
        "name": "b200-release",
        "displayName": "B200 product binaries",
        "inherits": "product",
        "cacheVariables": {"NINFER_TARGET": "b200"},
    },
    "b200-dev": {
        "name": "b200-dev",
        "displayName": "B200 products, tests and benchmarks",
        "inherits": "b200-release",
        "cacheVariables": {"BUILD_TESTING": "ON", "NINFER_BUILD_BENCHMARKS": "ON"},
    },
    "b200-ops": {
        "name": "b200-ops",
        "displayName": "B200 Core, Artifact and Ops development",
        "inherits": "b200-dev",
        "cacheVariables": {"NINFER_BUILD_ENGINE": "OFF", "NINFER_BUILD_APPS": "OFF"},
    },
}

UNTOUCHED_BUILD_PRESETS = {
    name: {"name": name, "configurePreset": name}
    for name in (
        "rtx5090-release",
        "rtx5090-dev",
        "rtx5090-ops",
        "dgx-spark-release",
        "dgx-spark-dev",
        "dgx-spark-ops",
    )
}

B200_BUILD_PRESETS = {
    name: {"name": name, "configurePreset": name}
    for name in ("b200-release", "b200-dev", "b200-ops")
}

UNTOUCHED_TEST_PRESETS = {
    name: {
        "name": name,
        "configurePreset": name,
        "output": {"outputOnFailure": True},
    }
    for name in ("rtx5090-dev", "rtx5090-ops", "dgx-spark-dev", "dgx-spark-ops")
}

B200_TEST_PRESETS = {
    name: {
        "name": name,
        "configurePreset": name,
        "output": {"outputOnFailure": True},
    }
    for name in ("b200-dev", "b200-ops")
}


def _run(cmd, cwd=None):
    return subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


class B200PlatformTests(unittest.TestCase):
    """b200 branch of cmake/NinferPlatform.cmake via a wrapper project."""

    def setUp(self):
        if shutil.which("cmake") is None:
            self.skipTest("cmake not available")
        machine = host_platform.machine()
        if machine not in ("x86_64", "amd64", "AMD64"):
            self.skipTest(f"b200 main path expects an x86_64 host, got {machine}")

    def _wrapper(self, root: Path) -> Path:
        (root / "src" / "core").mkdir(parents=True)
        shutil.copyfile(PLATFORM_H_IN, root / "src" / "core" / "platform.h.in")
        (root / "CMakeLists.txt").write_text(
            WRAPPER_CMAKELISTS.format(platform_cmake=PLATFORM_CMAKE)
        )
        return root

    def _configure(self, root: Path, *args):
        return _run(
            ["cmake", "-S", str(root), "-B", str(root / "build"), "-G", "Ninja", *args]
        )

    def test_configure_reports_b200_target_and_capabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._wrapper(Path(tmp))
            result = self._configure(root, "-DNINFER_TARGET=b200")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn(
                "NInfer target: b200 (x86_64, sm_100a)", result.stdout, result.stdout
            )
            header = root / "build" / "generated" / "core" / "platform.h"
            self.assertTrue(header.is_file())
            self.assertIn("#define NINFER_TARGET_B200 1", header.read_text())
            compiler = next(
                (
                    found
                    for found in ("c++", "g++", "clang++")
                    if shutil.which(found)
                ),
                None,
            )
            if compiler is None:
                self.skipTest("no C++ compiler for the capability probe")
            compiler = shutil.which(compiler)
            probe = root / "probe.cpp"
            probe.write_text(CAPABILITY_PROBE)
            compiled = _run(
                [compiler, "-std=c++20", "-I", str(root / "build" / "generated"),
                 "-fsyntax-only", str(probe)]
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout)

    def test_rejects_mismatched_cuda_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._wrapper(Path(tmp))
            result = self._configure(
                root, "-DNINFER_TARGET=b200", "-DCMAKE_CUDA_ARCHITECTURES=120a"
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(
                "b200 requires CMAKE_CUDA_ARCHITECTURES=100a", result.stdout
            )

    def test_rejects_non_x86_64_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._wrapper(Path(tmp))
            toolchain = root / "aarch64.toolchain.cmake"
            toolchain.write_text(AARCH64_TOOLCHAIN)
            result = self._configure(
                root,
                "-DNINFER_TARGET=b200",
                f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("b200 does not support Linux/aarch64", result.stdout)


class B200PresetTests(unittest.TestCase):
    """b200 presets exist, mirror rtx5090, and leave other targets untouched."""

    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(PRESETS.read_text())
        cls.configure = {p["name"]: p for p in cls.doc["configurePresets"]}
        cls.build = {p["name"]: p for p in cls.doc["buildPresets"]}
        cls.test = {p["name"]: p for p in cls.doc["testPresets"]}

    def test_list_presets_offer_b200_release(self):
        if shutil.which("cmake") is None:
            self.skipTest("cmake not available")
        for flag in ("--list-presets", "--build --list-presets"):
            result = _run(["cmake", *flag.split()], cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("b200-release", result.stdout)

    def test_b200_configure_presets_inherit_product(self):
        product = self.configure["product"]
        self.assertEqual(product["binaryDir"], "${sourceDir}/build/${presetName}")
        for name, expected in B200_CONFIGURE_PRESETS.items():
            self.assertIn(name, self.configure)
            self.assertEqual(self.configure[name], expected)
            self.assertNotIn("binaryDir", self.configure[name])

    def test_b200_build_and_test_presets_mirror_rtx5090(self):
        for name, expected in B200_BUILD_PRESETS.items():
            self.assertEqual(self.build.get(name), expected)
        for name, expected in B200_TEST_PRESETS.items():
            self.assertEqual(self.test.get(name), expected)

    def test_rtx5090_and_dgx_spark_presets_unchanged(self):
        for name, expected in UNTOUCHED_CONFIGURE_PRESETS.items():
            self.assertEqual(self.configure.get(name), expected)
        for name, expected in UNTOUCHED_BUILD_PRESETS.items():
            self.assertEqual(self.build.get(name), expected)
        for name, expected in UNTOUCHED_TEST_PRESETS.items():
            self.assertEqual(self.test.get(name), expected)


if __name__ == "__main__":
    unittest.main()
