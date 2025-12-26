from .perf_profiler import PerformanceProfiler
def get_profile(quick=True):
    return PerformanceProfiler(quick=quick).profile()
