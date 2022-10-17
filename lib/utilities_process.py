import os
from subprocess import Popen
import tifffile
import PIL
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS=100000000000

def workernoshell(cmd):
    """
    Set up an shell command. That is what the shell true is for.
    Args:
        cmd:  a command line program with arguments in a list
    Returns: nothing
    """
    stderr_template = os.path.join(os.getcwd(), "workernoshell.err.log")
    stdout_template = os.path.join(os.getcwd(), "workernoshell.log")
    stdout_f = open(stdout_template, "w")
    stderr_f = open(stderr_template, "w")
    my_env = os.environ.copy()
    my_env["PATH"] = "/usr/sbin:/sbin:" + my_env["PATH"]
    proc = Popen(cmd, shell=False, stderr=stderr_f, stdout=stdout_f, env=my_env)
    proc.wait()

def get_image_dimension(path):
    im = Image.open(path)
    width, height = im.size
    return width, height