"""
A CLI which can be used to copy, move or delete files
and folders. Folder operations copy the structure first creating new folders,
and then process the files. This module will not allow you to delete folders,
empty or otherwise, as a precaution.
"""
import sys
import os
import argparse
import hashlib
import time
import functools
import shutil
import logging
import traceback
import datetime as dt
from pathlib import Path

def error_handler(func):
    """
    Wrapper function to log any errors
    """
    @functools.wraps(func)
    def inner(*args, **kwargs):
        retval = None
        try:
            retval = func(*args, **kwargs)
        except Exception as error:
            error = str(error)
            start = kwargs["start"]

            logging.critical(traceback.format_exc())
            print(error)

            end = time.perf_counter()
            mins, secs = divmod(end-start, 60)
            eof = ("{} Operation Failed. It may be partially complete."
                   " Run time: {:02d}:{:02d}").format(kwargs["mode"], int(mins), int(secs))
            logging.error(eof)

            sys.exit() # quit on failure

        return retval
    return inner


def path_walk(path):
    """
    Recursively iterate through a directory yielding a generator of directories and files.
    Returns the most nested folder first (bottom-up)
    """
    names = list(path.iterdir())

    dirs = (node for node in names if node.is_dir() is True)
    nondirs = (node for node in names if node.is_dir() is False)

    for folder in dirs:
        if folder.is_symlink() is False:
            for ret_val in path_walk(folder):
                yield ret_val

    yield path, dirs, nondirs


def generate_checksum(file_path):
    """
    Generate SHA256 Checksum of a file
      - I tested this function vs a 65596 block size, and blake2b or md5 hashing.
      - This is the fastest combination.
    """
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def bytes2human(_bytes):
    """
    >>> bytes2human(10000)
    '9K'
    >>> bytes2human(100001221)
    '95M'
    """
    symbols = ("B", "KB", "MB", "GB", "TB")
    prefix = {}
    _format = "{value:1.1f}{symbol}"
    for i, sym in enumerate(symbols[1:]):
        prefix[sym] = 1 << (i+1) * 10 # For each symbol, do a bitwise left shift multiples of 10
    for symbol in reversed(symbols[1:]):
        # Starting at the largest unit, if this is larger than that unit divide it into that unit
        if _bytes >= prefix[symbol]:
            value = float(_bytes) / prefix[symbol]
            return _format.format(value=value, symbol=symbol)
    return _format.format(value=_bytes, symbol=symbols[0])


def create_destination_folder(output_path, is_file=False):
    """
    Iterate over a paths parents and create them if it doesn't exist
    """
    for dst_path in reversed(output_path.parents):
        if not dst_path.exists():
            logging.debug("Folder %s could not be found and was created.", dst_path)
            os.mkdir(dst_path)

    if not output_path.exists() and not is_file:
        os.mkdir(output_path)


@error_handler
def copy_file(**kwargs):
    """
    Copy a file from source to destination by writing a new file using
    generated buffers from the original. This is much faster than shutil.copy()
    """

    src = kwargs["input"]
    dst = kwargs["output"]

    # If a filename hasn't been provided, use the original file name
    if src.is_file() and (dst.name != src.name):
        dst = dst.joinpath(src.name)

    # Be sure that a folder exists to copy the file to.
    try:
        create_destination_folder(dst, is_file=True)
    except:
        raise OSError("Could not create output folder\n" + traceback.format_exc())

    if dst.exists() and kwargs["overwrite"] == 1:
        logging.info("Output file %s already exists, the overwrite option was not selected so"
                     " the original file has been kept.", dst)
        return True

    # Setup file copy os args
    try:
        o_binary = os.O_BINARY
    except:
        o_binary = 0
    read_flags = os.O_RDONLY | o_binary
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | o_binary
    buffer_size = 128 * 1024

    # Generate a checksum
    checksum = generate_checksum(src)
    logging.debug("Checksum for %s is %s", src, checksum)

    try:
        source_file = os.open(src, read_flags)
        stat = os.fstat(source_file)
        destination_file = os.open(dst, write_flags, stat.st_mode)

        for chunk in iter(lambda: os.read(source_file, buffer_size), b""):
            os.write(destination_file, chunk)

        # Copy over metadata
        shutil.copystat(src, dst)

        # Validate checksum
        new_checksum = generate_checksum(dst)
        logging.debug("Checksum for %s is %s", dst, new_checksum)

        if new_checksum != checksum:
            raise Exception(f"Checksum for src could not be validated")

        return True

    except Exception as error:
        raise Exception from error
    finally:
        try:
            os.close(source_file)
        except:
            pass
        try:
            os.close(destination_file)
        except:
            pass
        logging.info("Checksum for %s validated", src)
        logging.info("Transferred: %s - size: %s", src.name, bytes2human(stat.st_size))


@error_handler
def move_file(**kwargs):
    """
    Move a target folder to a destination. This is effectively a copy and delete.
    """
    copy_file(**kwargs)
    delete_file(**kwargs)

    return True


@error_handler
def delete_file(**kwargs):
    """
    Delete a target file
    """
    try:
        kwargs["input"].unlink()
        logging.debug("%s was deleted without error.", kwargs["input"])
    except OSError as error:
        if error.errno == 13:
            raise OSError("Attempted to delete a folder in file mode.")

    return True


@error_handler
def copy_folder(**kwargs):
    """
    Copy the input folder to the output location
    """

    input_folder = kwargs["input"]
    output_folder = kwargs["output"]

    # Create destination folder
    try:
        create_destination_folder(output_folder)
    except:
        raise OSError("Could not create output folder\n" + traceback.format_exc())

    # Copy input folder directory structure and then copy files
    for path, dirs, nondirs in path_walk(input_folder):
        for _dir in dirs:
            rel_path = _dir.relative_to(input_folder)
            dst_path = output_folder.joinpath(rel_path)
            if not dst_path.exists():
                os.mkdir(dst_path)
                logging.debug("Folder %s could not be found and was created.", dst_path)

        for file in nondirs:
            rel_path = file.relative_to(input_folder)
            dst_path = output_folder.joinpath(rel_path)

            if not file.is_file() or dst_path.exists():
                logging.debug("Non directory object %s could not be identified "
                              "as a file and was not copied.", file)
                continue

            kwargs["output"] = dst_path
            kwargs["input"] = file
            copy_file(**kwargs)

    logging.debug("Folder %s copied successfully", input_folder)

    return True


@error_handler
def move_folder(**kwargs):
    """
    Move a target folder to a destination. This is effectively a copy and delete.
    """
    copy_folder(**kwargs)
    shutil.rmtree(kwargs["input"])

    return True


def main():
    """
    Parse CLI arguments and call relevant functions
    """
    parser = argparse.ArgumentParser(
        description=("A CLI Program which will move, copy or delete a specified file."
                     " This script has the same permissions as the user calling it.")
        )

    parser.add_argument(
        "-m", "--mode",
        required=True,
        type=str,
        choices=["MOVE", "COPY", "DELETE"],
        help="The mode of file manipulation."
        )

    parser.add_argument(
        "-t", "--type",
        required=True,
        type=str,
        choices=["file", "folder"],
        help=("Determine if the operation is recursive over a folder, or a single file."
              " You cannot specify DELETE mode on a folder.")
        )

    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="An input file path"
        )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help=("A path to copy/move files to, if the path does not exist it will be created"
              " if possible. If a folder is provided when --type is file, the input filename"
              " will be used. This argument is ignored in delete mode.")
        )

    parser.add_argument(
        "-w", "--overwrite",
        type=int,
        default=1,
        choices=[1, 2],
        help="Set the behaviour of name conflicts: 1 -keep original, 2 -overwrite"
        )

    parser.add_argument(
        "-v", "--verbosity",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="Verbosity of logging: 0 -critical, 1- error, 2 -warning, 3 -info, 4 -debug"
        )

    parser.add_argument(
        "-l", "--logfolder",
        type=str,
        default="./log",
        help=("A folder path to write any log files to. If the folder doesn't exist"
              " it will be created if possible. Defaults to a local log folder. "
              "Failing that the log will be in the same folder as the script.")
        )

    parser.add_argument(
        "-f", "--flagfile",
        action="store_true",
        help=("Setting this flag will make the script output a .done file at the end of processing."
              " The .done will contain the date and time the process finished.")
        )

    args = parser.parse_args()
    kwargs = vars(args)

    # Process arguments
    kwargs["input"] = Path(kwargs["input"])
    kwargs["output"] = Path(kwargs["output"]) if kwargs["output"] is not None else None

    failed_log_setup = False
    script_name = Path(sys.argv[0])

    if kwargs["logfolder"] == "./log":
        log_folder = script_name.parent.joinpath("log")
    else:
        log_folder = Path(kwargs["logfolder"])

    if not log_folder.exists() and not log_folder.is_file():
        try:
            os.mkdir(log_folder)
        except OSError:
            failed_log_setup = True
            log_folder = Path("")

    start = time.perf_counter()
    kwargs["start"] = start

    # Setup logging
    date_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    verbosity = {
        0: logging.CRITICAL,
        1: logging.ERROR,
        2: logging.WARNING,
        3: logging.INFO,
        4: logging.DEBUG
        }

    logging.basicConfig(
        filename=log_folder.joinpath(f"{script_name.stem}_{kwargs['mode']}_{date_tag}.log"),
        level=verbosity[kwargs["verbosity"]],
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
        )

    logging.debug(str(kwargs))

    if failed_log_setup:
        logging.warning("Could not create requested log folder: %s", kwargs["logfolder"])

    # Validate arguments
    if kwargs["mode"] == "DELETE" and kwargs["type"] == "folder":
        logging.error("When mode is set to DELETE, only a --type of file can be used.")
        parser.error("When mode is set to DELETE, only a --type of file can be used.")

    if kwargs["mode"] != "DELETE" and (kwargs["output"] == "" or kwargs["output"] is None):
        logging.error("An --output argument must be supplied.")
        parser.error("An --output argument must be supplied.")

    if kwargs["type"] == "folder" and kwargs["input"].is_file():
        logging.error("An --input of type 'file' cannot be supplied with a --type of 'folder'.")
        parser.error("An --input of type 'file' cannot be supplied with a --type of 'folder'.")

    if kwargs["type"] == "file" and kwargs["input"].is_dir():
        logging.error("An --input of type 'folder' cannot be supplied with a --type of 'file'.")
        parser.error("An --input of type 'folder' cannot be supplied with a --type of 'file'.")

    # Select function to run
    mode_selector = {
        "file": {
            "MOVE": move_file,
            "COPY": copy_file,
            "DELETE": delete_file
            },
        "folder": {
            "MOVE": move_folder,
            "COPY": copy_folder
            }
        }

    success = mode_selector[kwargs["type"]][kwargs["mode"]](**kwargs)

    if success:
        end = time.perf_counter()
        mins, secs = divmod(end-start, 60)
        eof = "File Transfer Success. Run time: {:02d}:{:02d}".format(int(mins), int(secs))
        logging.info(eof)

        if kwargs["flagfile"]:
            with open(kwargs["output"].joinpath(".done"), "w") as done_file:
                done_file.write(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
