# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import sys, os

from datetime import datetime

from functools import partial

def createDirIfNeeded(dirname):
    try:
        os.mkdir(dirname);
    except:
        pass


class cMemoize:
    def __init__(self, f):
        self.mFunction = f
        self.mCache = {}
    def __call__(self, *args):
        # print("MEMOIZING" , self.mFunction , args)
        if args not in self.mCache:
            self.mCache[args] = self.mFunction(*args)
        return self.mCache[args]
    
    def __get__(self, obj, objtype):
        # Support instance methods.
        return partial(self.__call__, obj)
  
class PyAF_Error(Exception):
    """Exception raised for errors in the forecasting process.

    Attributes:
        mReason : explanation of the error
    """

    def __init__(self, reason):
        self.mReason = reason

class Internal_PyAF_Error(PyAF_Error):
    """Exception raised for errors in the forecasting process.

    Attributes:
        mReason : explanation of the error
    """

    def __init__(self, reason):
        self.mReason = reason



def get_pyaf_logger():
    import logging;
    logger = logging.getLogger('pyaf.std');
    if(logger.handlers == []):
        import logging.config
        logging.basicConfig(level=logging.INFO)        
    return logger;

def get_pyaf_timing_logger():
    import logging;
    logger = logging.getLogger('pyaf.timing');
    if(logger.handlers == []):
        import logging.config
        logging.basicConfig(level=logging.INFO)        
    return logger;

def get_pyaf_hierarchical_logger():
    import logging;
    return logging.getLogger('pyaf.hierarchical')



class cTimer:
    def __init__(self, iMess = "PYAF_UNKNOWN_OP", iDebug = False):
        self.mMessage = iMess
        self.mStart = datetime.now();
        logger = get_pyaf_timing_logger();
        logger.info(("OPERATION_START", self.mMessage))

    def __del__(self):
        self.mEnd = datetime.now();
        lDelta = self.mEnd - self.mStart
        logger = get_pyaf_timing_logger();
        logger.info(("OPERATION_END_ELAPSED" , round(lDelta.total_seconds(), 3), self.mMessage))

