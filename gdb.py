"""
GDB pretty-printers for the sigo runtime.

Usage:
    (gdb) source /path/to/sigo_pretty_printers.py

Or add to ~/.gdbinit:
    source /path/to/sigo_pretty_printers.py
"""

import gdb
import gdb.printing


class StringPrinter:
    """Pretty-printer for runtime._string."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        length = int(self.val['len'])

        if length == 0:
            return '""'

        if length < 0:
            return '<invalid string: negative length {}>'.format(length)

        # Guard against absurd lengths from uninitialized memory
        if length > (1 << 24):
            return '<suspect string: length {}>'.format(length)

        array = self.val['array']
        if int(array) == 0:
            return '<nil string with len {}>'.format(length)

        try:
            # Read `length` raw bytes from the string's backing array.
            # Strings in sigo are not NUL-terminated, so we must use the
            # explicit length rather than relying on GDB's string reader.
            inferior = gdb.selected_inferior()
            raw = inferior.read_memory(int(array), length).tobytes()
        except (gdb.MemoryError, gdb.error) as e:
            return '<unreadable string at {}: {}>'.format(array, e)

        try:
            # Strings in Go (and sigo) are conventionally UTF-8.
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            # Fall back to a best-effort representation with escapes
            # for invalid byte sequences.
            text = raw.decode('utf-8', errors='backslashreplace')

        # Escape embedded quotes and control characters for readability.
        return '"{}"'.format(
            text.replace('\\', '\\\\')
            .replace('"', '\\"')
            .replace('\n', '\\n')
            .replace('\r', '\\r')
            .replace('\t', '\\t')
        )

    def display_hint(self):
        return 'string'


def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter('sigo_runtime')
    # Match both the qualified and unqualified forms that GDB may present,
    # depending on how the compiler emits the DWARF type name.
    pp.add_printer('_string', r'^runtime__string$', StringPrinter)
    return pp


gdb.printing.register_pretty_printer(
    gdb.current_objfile(),
    build_pretty_printer(),
    replace=True,
)