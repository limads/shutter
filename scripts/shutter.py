from ctypes import CDLL, c_char_p, c_void_p, c_ulong, c_byte, pointer, POINTER, c_int
import slices

_SHUTTER = CDLL("/home/diego/Software/eyelab/target/debug/deps/libshutter.so")
_SHUTTER.image_open.argtypes = [c_char_p]
_SHUTTER.image_open.restype = c_void_p

class Image(c_void_p):
    pass

_SHUTTER.image_new.argtypes = [c_ulong, c_ulong, c_byte]
_SHUTTER.image_new.restype = c_void_p

_SHUTTER.image_show.argtypes = [c_void_p]
_SHUTTER.image_show.restype = None

class ByteImage:

    def __init__(self, ptr : c_void_p):

        if type(ptr) != c_void_p:
            raise Exception("Type must be c_void_p")

        self._ptr = ptr

    def new(rows, cols, value):
        ptr = c_void_p(_SHUTTER.image_new(c_ulong(10),c_ulong(10),c_byte(255)))
        return ByteImage(ptr)

    def open(path : str):
        ptr = c_void_p(_SHUTTER.image_open(c_char_p(path.encode('utf-8'))))
        if not ptr:
            raise Exception("Invalid image path")
        else:
            return ByteImage(ptr)

    def show(self):
        _SHUTTER.image_show(self._ptr)

    def __del__(self):
        _SHUTTER.image_free(self._ptr)

