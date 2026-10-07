"""Reading the metadata of a GGUF model file, without a dependency.

GGUF (llama.cpp's format, versions 2 and 3): the magic `GGUF`, a version, the tensor count and
the key-value count, then the key-values (typed), then the tensor descriptions, then the tensor
data at an aligned offset. Only the header is read: the key-values, the number of tensors and
where their data starts (the weights are the rest of the file). Arrays of more than
`MAX_ARRAY` items (the vocabulary) are skipped, not kept.
"""

import struct
from pathlib import Path

MAGIC = b"GGUF"
MAX_ARRAY = 4096
MAX_STRING = 1 << 20

_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
    12: "<d",
}  # fmt: skip
STRING, ARRAY = 8, 9


class GgufError(Exception):
    """Not a GGUF file, or one this reader cannot read; the message is safe to show."""


class _Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise GgufError("the file ends inside its header")
        return struct.unpack(fmt, data)[0]

    def string(self) -> str:
        length = self.unpack("<Q")
        if length > MAX_STRING:
            raise GgufError("a string of the header is too long")
        return self.f.read(length).decode("utf-8", errors="replace")

    def value(self, kind: int):
        if kind in _SCALARS:
            return self.unpack(_SCALARS[kind])
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            item_kind, count = self.unpack("<I"), self.unpack("<Q")
            if count > MAX_ARRAY:
                self.skip(item_kind, count)
                return None
            return [self.value(item_kind) for _ in range(count)]
        raise GgufError(f"unknown value type {kind}")

    def skip(self, kind: int, count: int) -> None:
        if kind in _SCALARS:
            self.f.seek(struct.calcsize(_SCALARS[kind]) * count, 1)
        elif kind == STRING:
            for _ in range(count):
                self.f.seek(self.unpack("<Q"), 1)
        else:
            raise GgufError("nested arrays are not read")


def read_header(path: Path) -> dict:
    """{"metadata": {key: value}, "tensors": n, "data_offset": bytes, "file_size": bytes}."""
    with Path(path).open("rb") as f:
        if f.read(4) != MAGIC:
            raise GgufError("not a GGUF file")
        r = _Reader(f)
        version = r.unpack("<I")
        if version not in (2, 3):
            raise GgufError(f"GGUF version {version} is not read")
        tensors, count = r.unpack("<Q"), r.unpack("<Q")
        metadata = {}
        for _ in range(count):
            key = r.string()
            metadata[key] = r.value(r.unpack("<I"))
        for _ in range(tensors):
            r.string()
            dims = r.unpack("<I")
            f.seek(8 * dims + 4 + 8, 1)  # dimensions, type, offset
        alignment = int(metadata.get("general.alignment", 32))
        offset = f.tell()
        offset += (-offset) % alignment
        size = f.seek(0, 2)
    return {"metadata": metadata, "tensors": tensors, "data_offset": offset, "file_size": size}
