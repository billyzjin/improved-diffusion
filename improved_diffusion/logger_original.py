"""
Logger copied from OpenAI baselines to avoid extra RL-based dependencies:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""
import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter(object):
    def write(self, s):
        print(s)

    def writekvs(self, kvs):
        for k, v in kvs.items():
            print("{}: {}".format(k, v))

    def close(self):
        pass


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def write(self, s):
        self.file.write(s)
        self.file.flush()

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = "%-8.3g" % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = valstr

        # Find max widths
        if key2str:
            keys = sorted(key2str.keys())
            keywidth = max(map(len, keys))
            valwidth = max(map(len, key2str.values()))
        else:
            keywidth, valwidth = 0, 0

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for key in keys:
            lines.append(
                "| %s%s | %s%s |"
                % (
                    key,
                    " " * (keywidth - len(key)),
                    key2str[key],
                    " " * (valwidth - len(key2str[key])),
                )
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")
        self.file.flush()

    def _truncate(self, s):
        return s[:20] + "..." if len(s) > 23 else s

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        for k, v in kvs.items():
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = kvs.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
            self.file.flush()
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()


class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        if self.output_formats:
            for fmt in self.output_formats:
                fmt.writekvs(self.name2val)
        self.name2val.clear()
        self.name2cnt.clear()

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.write(*args)


def get_current():
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT


def _configure_default_logger():
    configure()
    Logger.CURRENT = Logger.DEFAULT


def configure(dir=None, format_strs=None, comm=None):
    """
    If comm is provided, average all numerical stats across that comm
    """
    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR")
    if dir is None:
        dir = osp.join(tempfile.gettempdir(),
                       datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)

    rank = 0
    if comm is not None:
        rank = comm.Get_rank()
    else:
        rank = 0

    if rank > 0:
        format_strs = None
    else:
        if format_strs is None:
            format_strs = ["stdout", "log", "csv"]

    output_formats = [make_output_format(f, dir) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    log("Logging to %s" % dir)


def make_output_format(format, dir):
    os.makedirs(dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(dir, "log.txt"))
    elif format == "json":
        return JSONOutputFormat(osp.join(dir, "progress.json"))
    elif format == "csv":
        return CSVOutputFormat(osp.join(dir, "progress.csv"))
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


# Backwards compatibility
def log(*args):
    get_current().log(*args)


def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)


def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)


def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)


def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    """
    get_current().dumpkvs()


def getkvs():
    return get_current().name2val.copy()


def log(*args, level=INFO):
    get_current().log(*args, level=level)


def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    get_current().set_level(level)


def get_dir():
    """
    Get directory that log files are being written to.
    """
    return get_current().get_dir()


def close():
    """
    Close the current logger.
    """
    if Logger.CURRENT is not None:
        Logger.CURRENT.close()
        Logger.CURRENT = None


# Configure the default logger
configure()
Logger.DEFAULT = Logger.CURRENT
