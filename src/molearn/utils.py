import os, sys
import numpy as np
import torch
import random

def random_string(length=32):
    '''
    generate a random string of arbitrary characters. Useful to generate temporary file names.

    :param length: length of random string
    '''
    return ''.join([random.choice(string.ascii_letters)
                    for n in range(length)])


def as_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.data.cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        return np.array(tensor)
    
    
def cpu_count():
    """ detect the number of available CPU """
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in list(os.sysconf_names):
            # Linux & Unix
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:
            # OSX
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())

    # Windows
    if "NUMBER_OF_PROCESSORS" in list(os.environ):
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"]);
        if ncpus > 0:
            return ncpus
        
    return 1
    

class ShutUp(object):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout =  self._stdout