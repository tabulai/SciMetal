"""ctypes binding to the native core (libmetalsci) and array hand-off.

Inputs go in as C-contiguous NumPy memory; results come back as NumPy arrays
that view the native buffer directly (no copy) and return it to the core's
buffer cache when the last view is dropped.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import functools
import math
import os
import pathlib
import threading

import numpy as np

from . import _config
from ._dispatch import NotAccelerated, Route, record, warn_precision_once

_CPU_COUNT = os.cpu_count() or 1


@functools.cache
def performance_cores() -> int:
    """Physical performance cores: the worker count handed to SciPy for batches."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        libc.sysctlbyname.argtypes = [
            ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        libc.sysctlbyname.restype = ctypes.c_int
        value, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        if libc.sysctlbyname(b"hw.perflevel0.physicalcpu", ctypes.byref(value), ctypes.byref(size), None, 0) == 0 \
                and value.value > 0:
            return value.value
    except (OSError, AttributeError):
        pass
    return _CPU_COUNT

MS_OK, MS_EINVAL, MS_ENOTSUP, MS_EOVERFLOW, MS_EOOM, MS_ETIMEOUT, MS_EBACKEND, MS_ESTATE = range(8)
MS_BUFFER_CPU, MS_BUFFER_METAL = 1, 2
MS_EXECUTION_CPU, MS_EXECUTION_GPU, MS_EXECUTION_AUTO = 1, 2, 3
MS_RUNTIME_ABI_VERSION = 1
MS_ARRAY_ABI_VERSION = 3
MS_MAX_NDIM = 8

_DTYPES = {np.dtype(np.float32): 1, np.dtype(np.float64): 2,
           np.dtype(np.complex64): 3, np.dtype(np.complex128): 4,
           np.dtype(np.bool_): 5,
           np.dtype(np.int8): 6, np.dtype(np.int16): 7, np.dtype(np.int32): 8, np.dtype(np.int64): 9,
           np.dtype(np.uint8): 10, np.dtype(np.uint16): 11, np.dtype(np.uint32): 12, np.dtype(np.uint64): 13}
_NUMPY_DTYPES = {code: dtype for dtype, code in _DTYPES.items()}
_PRECISIONS = {"fast": 1, "refined": 2, "strict": 3}
_EXECUTIONS = {"cpu": MS_EXECUTION_CPU, "gpu": MS_EXECUTION_GPU, "auto": MS_EXECUTION_AUTO}
_ENGINES = {MS_EXECUTION_CPU: "cpu", MS_EXECUTION_GPU: "gpu"}


class NativeError(RuntimeError):
    """The native core reported an error."""


class ContextOptions(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                ("storage_kind", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("memory_limit_bytes", ctypes.c_uint64), ("external_reservation_bytes", ctypes.c_uint64),
                ("reserved", ctypes.c_uint64 * 2)]


class Array(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                ("data", ctypes.c_void_p), ("dtype", ctypes.c_uint32), ("ndim", ctypes.c_uint32),
                ("shape", ctypes.c_uint64 * MS_MAX_NDIM)]


class RouteOptions(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                ("execution", ctypes.c_uint32), ("precision", ctypes.c_uint32),
                ("reserved", ctypes.c_uint64 * 2)]


class RouteInfo(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                ("engine", ctypes.c_uint32), ("compute_dtype", ctypes.c_uint32),
                ("reserved", ctypes.c_uint64 * 2),
                ("reason", ctypes.c_char * 256)]


class ArrayResult(ctypes.Structure):
    _fields_ = [("buffer", ctypes.c_void_p), ("dtype", ctypes.c_uint32), ("ndim", ctypes.c_uint32),
                ("shape", ctypes.c_uint64 * MS_MAX_NDIM)]


def _library_path() -> pathlib.Path:
    override = os.environ.get("METAL_SCI_LIBRARY")
    here = pathlib.Path(__file__).resolve().parent
    candidates = [pathlib.Path(override)] if override else [
        here / "libmetalsci.dylib",                                        # installed wheel
        here.parents[1] / ".build" / "metal" / "libmetalsci.dylib",        # source tree
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ImportError("libmetalsci.dylib not found; build it (cmake --preset metal && cmake --build "
                      "--preset metal) or set METAL_SCI_LIBRARY")


_lock = threading.Lock()
_lib = None
_context_state: tuple[int, int] | None = None


def library() -> ctypes.CDLL:
    """The loaded native library, with prototypes declared."""
    global _lib
    if _lib is None:
        with _lock:
            if _lib is None:
                lib = ctypes.CDLL(str(_library_path()))
                c_void_pp = ctypes.POINTER(ctypes.c_void_p)
                lib.ms_context_create.argtypes = [ctypes.POINTER(ContextOptions), c_void_pp]
                lib.ms_context_release.argtypes = [ctypes.c_void_p]
                lib.ms_context_release.restype = None
                lib.ms_context_error_copy.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64]
                lib.ms_context_trim.argtypes = [ctypes.c_void_p]
                lib.ms_status_message.argtypes = [ctypes.c_int]
                lib.ms_status_message.restype = ctypes.c_char_p
                lib.ms_buffer_release.argtypes = [ctypes.c_void_p]
                lib.ms_buffer_release.restype = None
                lib.ms_buffer_data.argtypes = [ctypes.c_void_p, c_void_pp, ctypes.POINTER(ctypes.c_uint64)]
                lib.ms_array_astype.argtypes = [ctypes.c_void_p, ctypes.POINTER(Array), ctypes.c_uint32,
                                                ctypes.POINTER(ArrayResult)]
                lib.ms_array_capabilities.argtypes = []
                lib.ms_array_capabilities.restype = ctypes.c_uint32
                _lib = lib
    return _lib


MS_CAPABILITY_METAL = 1


def has_metal() -> bool:
    """Whether the native library was built with Metal storage and the GPU engines."""
    return bool(library().ms_array_capabilities() & MS_CAPABILITY_METAL)


def decline_small(elements: int, break_even: int) -> None:
    """Hand a call below its break-even size to SciPy, unless small inputs are kept native."""
    if elements < break_even and _config.small_to_scipy():
        raise NotAccelerated(f"{elements} elements: below the {break_even} where the native route wins")


def _memory_limit() -> int:
    override = os.environ.get("METAL_SCI_MEMORY_LIMIT_GB")
    if override:
        gb = float(override)
        if not math.isfinite(gb) or gb <= 0:
            raise ValueError(f"METAL_SCI_MEMORY_LIMIT_GB must be a positive finite number, got {override!r}")
        limit_bytes = int(gb * (1 << 30))
        if limit_bytes <= 0 or limit_bytes >= (1 << 64):
            raise ValueError(f"METAL_SCI_MEMORY_LIMIT_GB out of range: {override!r}")
        return limit_bytes
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return physical // 2


def context() -> int:
    """The process-wide native context (Metal storage), created on first use.

    A forked child cannot use the parent's Metal state, so there the call is
    handed back to SciPy.
    """
    global _context_state
    state = _context_state
    if state is not None:
        handle, pid = state
        if pid != os.getpid():
            raise NotAccelerated("the native context is unavailable after fork")
        return handle
    lib = library()
    with _lock:
        state = _context_state
        if state is not None:
            handle, pid = state
            if pid != os.getpid():
                raise NotAccelerated("the native context is unavailable after fork")
            return handle
        # A build without Metal has CPU storage and the CPU engines only.
        storage = MS_BUFFER_METAL if has_metal() else MS_BUFFER_CPU
        options = ContextOptions(struct_size=ctypes.sizeof(ContextOptions),
                                 abi_version=MS_RUNTIME_ABI_VERSION, storage_kind=storage,
                                 memory_limit_bytes=_memory_limit())
        handle = ctypes.c_void_p()
        status = lib.ms_context_create(ctypes.byref(options), ctypes.byref(handle))
        if status != MS_OK:
            raise NativeError(lib.ms_status_message(status).decode())
        _context_state = (handle.value, os.getpid())
        return handle.value


def check(status: int) -> None:
    """Raise for a failed native call; unsupported requests fall through to SciPy."""
    if status == MS_OK:
        return
    lib = library()
    message = ctypes.create_string_buffer(512)
    state = _context_state
    ctx = state[0] if state is not None and state[1] == os.getpid() else None
    if ctx is None or lib.ms_context_error_copy(ctx, message, len(message)) != MS_OK:
        text = lib.ms_status_message(status).decode()
    else:
        text = message.value.decode(errors="replace")
    if status in (MS_ENOTSUP, MS_ESTATE):
        raise NotAccelerated(text)
    if status == MS_EOOM:
        raise MemoryError(text)
    if status == MS_EINVAL:
        raise ValueError(text)
    raise NativeError(text)


def native_dtype(dtype) -> int | None:
    """The native dtype code for a NumPy dtype, or None when there is no native path."""
    dt = np.dtype(dtype)
    if dt.char in ("g", "G") or not dt.isnative:
        return None
    return _DTYPES.get(dt)


def as_native(array: np.ndarray) -> tuple[Array, np.ndarray]:
    """Describe a NumPy array to the native core.

    Returns the descriptor and the (possibly copied) C-contiguous, aligned array
    that must stay alive for the duration of the call.
    """
    if not isinstance(array, np.ndarray):
        raise NotAccelerated(f"expected a NumPy ndarray, got {type(array).__name__}")
    code = native_dtype(array.dtype)
    if code is None:
        raise NotAccelerated(f"no native path for dtype {array.dtype}")
    if not 1 <= array.ndim <= MS_MAX_NDIM:
        raise NotAccelerated(f"no native path for {array.ndim}-dimensional arrays")
    held = np.require(array, requirements=("C", "A"))
    if held.dtype != array.dtype or not 1 <= held.ndim <= MS_MAX_NDIM:
        raise NotAccelerated("array could not be normalized to native C-contiguous aligned storage")
    descriptor = Array(struct_size=ctypes.sizeof(Array), abi_version=MS_ARRAY_ABI_VERSION,
                       data=held.ctypes.data, dtype=code, ndim=held.ndim)
    for axis, extent in enumerate(held.shape):
        descriptor.shape[axis] = extent
    return descriptor, held


class _OwnedBuffer:
    """Keeps a native buffer alive for the NumPy arrays that view it."""

    __slots__ = ("_handle", "_pid", "_getpid", "_release", "__array_interface__")

    def __init__(self, handle, pointer, shape, dtype, release):
        self._handle = handle
        self._getpid = os.getpid
        self._pid = self._getpid()
        self._release = release
        self.__array_interface__ = {"version": 3, "data": (pointer, False), "shape": shape,
                                    "typestr": dtype.str}

    def __del__(self):
        # Inherited handles must not be touched in a forked child.
        handle = self._handle
        if handle and self._pid == self._getpid():
            self._handle = None
            self._release(handle)


def to_numpy(result: ArrayResult) -> np.ndarray:
    """A NumPy array viewing a native result without copying it."""
    lib = library()
    release = lib.ms_buffer_release
    handle = result.buffer
    try:
        dtype = _NUMPY_DTYPES[result.dtype]
        shape = tuple(int(result.shape[axis]) for axis in range(result.ndim))
        pointer, size = ctypes.c_void_p(), ctypes.c_uint64()
        check(lib.ms_buffer_data(handle, ctypes.byref(pointer), ctypes.byref(size)))
        if size.value == 0:
            release(handle)
            handle = None
            return np.empty(shape, dtype)
        owner = _OwnedBuffer(handle, pointer.value, shape, dtype, release)
        handle = None
        return np.asarray(owner)
    except BaseException:
        if handle:
            release(handle)
        raise


def route_options() -> RouteOptions:
    """Routing for the current precision and device policy."""
    return RouteOptions(struct_size=ctypes.sizeof(RouteOptions), abi_version=MS_ARRAY_ABI_VERSION,
                        execution=_EXECUTIONS[_config.get_device()],
                        precision=_PRECISIONS[_config.get_precision()])


def new_route_info() -> RouteInfo:
    return RouteInfo(struct_size=ctypes.sizeof(RouteInfo), abi_version=MS_ARRAY_ABI_VERSION)


def record_route(function: str, info: RouteInfo, reason: str = "", input_dtype=None) -> None:
    c_reason = info.reason.decode("utf-8", errors="replace") if info.reason else ""
    final_reason = c_reason or reason
    compute_dtype = _NUMPY_DTYPES[info.compute_dtype].name
    record(Route(function, _ENGINES.get(info.engine, "?"), compute_dtype,
                 _config.get_precision(), final_reason))
    warn_precision_once(input_dtype, compute_dtype)


def astype(array: np.ndarray, dtype) -> np.ndarray:
    """Threaded precision conversion in the native core (float32 <-> float64, complex64 <-> complex128)."""
    code = native_dtype(dtype)
    if code is None:
        raise NotAccelerated(f"no native path for dtype {np.dtype(dtype)}")
    descriptor, held = as_native(np.asarray(array))
    result = ArrayResult()
    check(library().ms_array_astype(context(), ctypes.byref(descriptor), code, ctypes.byref(result)))
    del held
    return to_numpy(result)
