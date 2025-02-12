from . import radius

try:
    from . import radius_cu
except ImportError:
    print(
        "Package `mlcg_opt_radius` was not installed. Running with JIT compilation."
    )
