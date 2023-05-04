# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import Utils as tsutil

class cCuttingInfo:
    def __init__(self):
        pass


    def estimate(self):
        self.defineCuttingParameters();

    def set_default_split(self):
        self.mTrainSize = self.mSignalFrame.shape[0];
        assert(self.mTrainSize > 0);
        lEstEnd = int((self.mTrainSize - self.mHorizon) * self.mOptions.mEstimRatio);
        lValSize = self.mTrainSize - self.mHorizon - lEstEnd;
        lTooSmall = self.mTrainSize < 30 or lValSize < self.mHorizon
        if lTooSmall:
            self.mEstimEnd = self.mTrainSize;
            self.mValidStart = 0;
            self.mValidEnd = self.mTrainSize;
            self.mTestStart = 0;
        else:
            self.mEstimEnd = lEstEnd;
            self.mValidStart = self.mEstimEnd;
            self.mValidEnd = self.mTrainSize - self.mHorizon;
            self.mTestStart = self.mValidEnd;

        self.mTestEnd = self.mTrainSize;
        self.mEstimStart = 0;

    def check_split(self, iSplit):
        if (len(iSplit) != 3):
            raise tsutil.PyAF_Error(f'Invalid Split {str(iSplit)}');
        if (iSplit[0] < 0.0 or iSplit[0] > 1.0):
            raise tsutil.PyAF_Error(f'Invalid Estimation Ratio {str(iSplit[0])}');
        if (iSplit[1] < 0.0 or iSplit[1] > 1.0):
            raise tsutil.PyAF_Error(f'Invalid Validation Ratio {str(iSplit[1])}');
        if (iSplit[2] < 0.0 or iSplit[2] > 1.0):
            raise tsutil.PyAF_Error(f'Invalid Test Ratio {str(iSplit[2])}');
        lTotal =  iSplit[0] + iSplit[1] + iSplit[2]
        if (lTotal < 0 or lTotal > 1):
            raise tsutil.PyAF_Error(f'Invalid Split Ratio Sum{str(iSplit)}');

            
    def set_split(self, iSplit):
        self.mTrainSize = self.mSignalFrame.shape[0];
        assert(self.mTrainSize > 0);
        self.check_split(iSplit)
        lEstEnd = int(self.mTrainSize * iSplit[0]);
        lValSize = int(self.mTrainSize * iSplit[1]);
        lTestSize = int(self.mTrainSize * iSplit[2]);
        
        self.mEstimStart = 0;
        self.mEstimEnd = lEstEnd;
        self.mValidStart = self.mEstimEnd;
        self.mValidEnd = self.mValidStart + lValSize;
        self.mTestStart = self.mValidEnd;
        self.mTestEnd = self.mTestStart + lTestSize;
        
    def defineCuttingParameters(self):
        lStr = "CUTTING_START SignalVariable='" + self.mSignal +"'";
        # print(lStr);
        #print(self.mSignalFrame.head())
        if(self.mOptions.mCustomSplit is not None):
            self.set_split(self.mOptions.mCustomSplit)
        else:
            self.set_default_split()

        lStr = f"CUTTING_PARAMETERS {str(self.mTrainSize)} Estimation = ({str(self.mEstimStart)} , {str(self.mEstimEnd)})";
        lStr += f" Validation = ({str(self.mValidStart)} , {str(self.mValidEnd)})";
        lStr += f" Test = ({str(self.mTestStart)} , {str(self.mTestEnd)})";

    def cutFrame(self, df):
        lFrameFit = df[self.mEstimStart : self.mEstimEnd];
        lFrameForecast = df[self.mValidStart : self.mValidEnd];
        lFrameTest = df[self.mTestStart : self.mTestEnd];
        return (lFrameFit, lFrameForecast, lFrameTest)

    def getEstimPart(self, df):
        return df[self.mEstimStart : self.mEstimEnd]

    def getValidPart(self, df):
        return df[self.mValidStart : self.mValidEnd]


    def info(self):
        lStr2 += f" Estimation = ({str(self.mEstimStart)} , {str(self.mEstimEnd)})";
        lStr2 += f" Validation = ({str(self.mValidStart)} , {str(self.mValidEnd)})";
        lStr2 += f" Test = ({str(self.mTestStart)} , {str(self.mTestEnd)})";
        lStr2 += f" Horizon={str(self.mHorizon)}";
        return lStr2;

