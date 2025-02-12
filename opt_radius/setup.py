from setuptools import find_packages, setup, Extension
from torch.utils import cpp_extension

setup(
    name="mlcg_opt_radius",
    packages=find_packages(),
    ext_modules=[
        cpp_extension.CUDAExtension(
            "mlcg_opt_radius.radius_cu",
            ["mlcg_opt_radius/radius.cu"],
            py_limited_api=True,
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={
        "build_ext": cpp_extension.BuildExtension.with_options(
            no_python_abi_suffix=True, use_ninja=False
        )
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
