"""Minimal stub of the deprecated aifc module for Python >=3.13.
This satisfies imports performed by third-party libraries (e.g. SpeechRecognition)
that expect the standard library aifc to exist. The application only uses WAV
files, so AIFF parsing is intentionally not implemented.
If you attempt to open an AIFF file, an informative error is raised.
"""
__all__ = ["Error", "open"]

class Error(Exception):
    pass

def open(file, mode='r'):
    raise Error("AIFF files are not supported in this deployment environment.")
