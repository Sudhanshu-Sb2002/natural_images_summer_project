import math
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.fft as fourier
import pyspike as spk
import scipy.io as sio
import pandas
import matlab.engine
import cv2 as cv
import math
from numba import njit, prange
import scipy.signal
import nitime.algorithms as tsa
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import antropy as ent
import sklearn.metrics as metrics
import random
import scipy.stats as stats
import multitaper
# %%
matlab_needed = False
if matlab_needed:
    sessions = matlab.engine.find_matlab()
    engine = []
    if len(sessions) == 0:
        engine = matlab.engine.start_matlab()
    else:
        engine = matlab.engine.connect_matlab(sessions[-1])
    engine.addpath(
        'D:\\OneDrive - Indian Institute of Science\\5th Sem\\Summer project\\natural_images_code\\NaturalImagesGammaProject\\programs',
        nargout=0)


# %% md
# Loader functions
# %%
def loadlfpInfo(folderLFP):
    x = sio.loadmat(os.path.join(folderLFP, 'lfpInfo.mat'))
    analogChannelsStored = np.squeeze(x["analogChannelsStored"]) - 1
    goodStimPos = np.squeeze(x["goodStimPos"]) - 1
    timeVals = np.squeeze(x["timeVals"])
    analogInputNums = []
    if 'analogInputNums' in x:
        analogInputNums = np.squeeze(x["analogInputNums"])
    return [analogChannelsStored, timeVals, goodStimPos, analogInputNums]


def loadspikeInfo(folderSpikes):
    fileName = os.path.join(folderSpikes, 'spikeInfo.mat')
    try:
        x = sio.loadmat(fileName)
        return [np.squeeze(x["neuralChannelsStored"] - 1), np.squeeze(x["SourceUnitID"])]
    except FileNotFoundError:
        print('No spikeInfo.mat found in', folderSpikes)
        return [[], []]


def loadParameterCombinations(folderExtract=None):
    p = sio.loadmat(os.path.join(folderExtract, 'parameterCombinations.mat'))
    # subtract one from indices bcos matlab->python
    p["parameterCombinations"] = p["parameterCombinations"] - 1
    p["fValsUnique"] = p["fValsUnique"] - 1
    if 'sValsUnique' not in p:
        p['sValsUnique'] = p["rValsUnique"] / 3
    if 'cValsUnique' not in p:
        p['cValsUnique'] = 100
    if 'tValsUnique' not in p:
        p['tValsUnique'] = 0
    return p


def change_reference(analogData, folderLFP, referenceChannelString=None):
    if referenceChannelString is not None:
        if referenceChannelString == 'AvgRef':
            print('Changing to average reference')
            return analogData - np.squeeze(sio.loadmat(os.path.join(folderLFP, 'AvgRef.mat'))["analogData"])
        else:
            print('Changing to bipolar reference')
            return analogData - np.squeeze(sio.loadmat(os.path.join(folderLFP, referenceChannelString))["analogData"])
    return analogData


def get_bad_trials(badTrialFileName=None, useCommonBadTrialsFlag=True, channelString=None):
    badTrials = []
    allBadTrials = []

    if os.path.isfile(badTrialFileName):
        x = sio.loadmat(badTrialFileName)
        badTrials = np.squeeze(x["badTrials"]) - 1
        if "allBadTrials" in x:
            allBadTrials = np.squeeze(x["allBadTrials"]) - 1
        else:
            allBadTrials = badTrials

    if not useCommonBadTrialsFlag:
        badTrials = allBadTrials[channelString[5:]] - 1
    print(str(len(badTrials)), ' bad trials')

    return badTrials, allBadTrials


# %%
subjectName = "alpaH"
expDate = "240817"
protocolName = "GRF_002"
imageFolderName = 'ImagesTL'
folderSourceString = 'D:\\OneDrive - Indian Institute of Science\\5th Sem\\Summer project\\natural_images_code\\NaturalImagesGammaProject'
gridType = "Microelectrode"
gridLayout = 2,
badTrialNameStr = ''

# get the folder names
folderName = os.path.join(folderSourceString, 'data', subjectName, gridType, expDate, protocolName)
folderExtract = os.path.join(folderName, 'extractedData')
folderSegment = os.path.join(folderName, 'segmentedData')
folderLFP = os.path.join(folderSegment, 'LFP')
folderSpikes = os.path.join(folderSegment, 'Spikes')
rawImageFolder = os.path.join(folderSourceString, 'data', 'images', imageFolderName)
# standardised parameters
blRange = np.array([-0.25, 0])
stRange = np.array([0.25, 0.5])

# load    Spike and LFP    Information
[analogChannelsStored, timeVals, goodStimPos, analogInputNums] = loadlfpInfo(folderLFP)
[neuralChannelsStored, sourceUnitIDs] = loadspikeInfo(folderSpikes)
timeVals_blSt = timeVals[np.logical_and(timeVals >= blRange[0], timeVals <= stRange[-1])]
# Get RF details
rfData = sio.loadmat(
    os.path.join(folderSourceString, 'data', 'rfData', subjectName, subjectName + gridType + 'RFData.mat'))
# work with only high RMSE electrodes
highRMSElectrodes = np.squeeze(rfData['highRMSElectrodes']) - 1

# work only with sorted units
sortedPos = np.argwhere(sourceUnitIDs == 1)
sourceUnitIDs = sourceUnitIDs[sortedPos]
spikeChannels = np.squeeze(neuralChannelsStored[sortedPos])

referenceChannelString = None
plotColor = 'k'
analogChannels = np.setdiff1d(analogChannelsStored, highRMSElectrodes)
# define a commonArgs dictionary for all the plots

kwargs = {"folderName": folderName, "timeVals": timeVals, "plotColor": plotColor,
          "blRange": blRange, "stRange": stRange, "referenceChannelString": referenceChannelString,
          "badTrialNameStr": badTrialNameStr}
# %%
rfStats = np.squeeze(sio.loadmat(
    os.path.join(folderSourceString, 'data', 'rfData', subjectName, subjectName + 'Microelectrode' + 'RFData.mat'))[
                         'rfStats'])


# %% md
# LFP and Spike stuff
# %%


# %%
def custom_concat(array1):
    size = int(0)
    for i in array1:
        size += i.size
    result = np.zeros(size, dtype=np.float64)
    index = int(0)
    for i in range(len(array1)):
        for j in range(len(array1[i])):
            result[index] = array1[i][j]
            index += 1
    return result


def get_spike_data(folderSpikes, channelNumber, unitID):
    return np.squeeze(
        sio.loadmat(os.path.join(folderSpikes, 'elec' + str(channelNumber + 1) + '_SID' + str(unitID) + '.mat'))[
            "spikeData"])


def getPSTH_forp(X, tRange, d=0.001, Ntrials=1, smoothSigmaMs=None):
    if d >= 1:
        d = d / 1000

    spk = np.sort(X)

    # select only those values which lie between tRange[0] and tRange[-1]

    ind1 = np.argwhere(spk >= tRange[0])[0][0]
    ind2 = -1
    try:
        ind2 = np.argwhere(spk >= tRange[-1])[0][0]
    except:
        pass
    spk = spk[ind1:ind2]
    N = int((tRange[-1] - tRange[0]) / d)
    H = np.histogram(spk, bins=N, range=tRange)[0]
    timeVals = np.linspace(tRange[0] + d / 2, tRange[-1] - d / 2, num=N)
    # compute the PSTH
    psth = H / d / Ntrials
    # smooth the PSTH if requested
    return psth, timeVals


def plotSpikeData1Channel(plotHandles=None, channelNumber=None, stimulus_list=None, folderName=None, analysisType=None,
                          timeVals=None, plotColor='g', unitID=None, badTrialNameStr="", plot=False, bin_time=50):
    # plots the data for a single channel
    if unitID is None:
        unitID = np.ones(len(channelNumber), dtype=int)
    # get folder names
    folderExtract = os.path.join(folderName, 'extractedData')
    folderSegment = os.path.join(folderName, 'segmentedData')
    folderSpikes = os.path.join(folderSegment, 'Spikes')
    numPlots = len(stimulus_list)
    return_value = np.zeros((numPlots, len(channelNumber), 2, int(((timeVals[-1] - timeVals[0]) * 1000) / bin_time)))
    # get the stimuli
    parameterCombinations = loadParameterCombinations(folderExtract)['parameterCombinations']
    badTrialFileName = os.path.join(folderSegment
                                    , 'badTrials' + badTrialNameStr + '.mat')
    badTrials, allBadTrials = [np.squeeze(i) for i in
                               get_bad_trials(badTrialFileName)]
    # get the spike data
    spikeData = {i: get_spike_data(folderSpikes, channelNumber[i], unitID[i]) for i in range(len(channelNumber))}

    for i in range(numPlots):
        goodPos = np.squeeze(np.setdiff1d(parameterCombinations[0, 0, 0, stimulus_list[i]], badTrials))
        Ntrials = len(goodPos)
        for j, electrode in enumerate(channelNumber):
            # get the good trials for each stimulus by removing the bad trials
            data = custom_concat(spikeData[j][goodPos])
            if goodPos is None:
                print('No entries for this combination..')
                continue
            print('image', str(stimulus_list[i]), ', electrode=', str(electrode))

            if analysisType == 2:
                return_value[i][j] = getPSTH_forp(data, timeVals, d=bin_time, Ntrials=Ntrials)
                if plot:
                    pass
            elif analysisType == 1:
                # Raster Plot
                X = spikeData[goodPos]
                if plot:
                    plotHandles[i].eventplot(X, colors='k')
                return X
    return np.array(return_value)


best_electrodes = np.squeeze(np.intersect1d(highRMSElectrodes[:-3], spikeChannels))
best_electrode_names = ['elec' + str(i + 1) for i in best_electrodes]
channelPos_names = [str(i) + ", SID 1" for i in best_electrodes]
channelNumber = best_electrodes
stimValsToUse = np.array([i for i in range(32)])


# %%
def plotImageData(hImagesPlot=None, hImagePatches=None, rawImageFolder="", fValsToUse=None, channelNumber=[1],
                  colorName='g', plottingDetails=None, rfStats=None, RFtype=None):
    plot = hImagesPlot is not None or hImagePatches is not None
    patchSizeDeg = 2
    data = np.zeros(len(fValsToUse), dtype='object')

    for i in range(len(fValsToUse)):
        imageFileName = os.path.join(rawImageFolder, 'Image' + str(fValsToUse[i] + 1) + '.png')
        if plot:
            plottingDetails["hImagePlot"] = hImagesPlot[i]
            plottingDetails["hImagePatches"] = hImagePatches[i]
            plottingDetails["colorNames"] = colorName
            # data.append(getImagePatches(imageFileName,channelNumber,subjectName,'',patchSizeDeg,plottingDetails,nargout=1))
        else:
            data[i] = getImagePatches_forpython(rfStats, imageFileName, channelNumber, patchSizeDeg, RFtype=RFtype)
        if plot:
            if i > 1:
                hImagesPlot[i].set_xticklabels([])
                hImagesPlot[i].set_yticklabels([])
            else:
                hImagesPlot[i].set_xlabel('Degrees')
                hImagesPlot[i].set_ylabel('Degrees')
                hImagePatches[i].set_xlabel('Degrees')
                hImagePatches[i].set_ylabel('Degrees')
    return np.array(data)


@njit(parallel=True)
def twoDguassian(grid, x_0, y_0, sigma_x, sigma_y, A=1):
    for y_pix in prange(grid.shape[0]):
        for x_pix in prange(grid.shape[1]):
            x = x_pix - x_0
            y = y_pix - y_0
            grid[y_pix, x_pix] = np.exp(-((x ** 2) / (2 * sigma_x ** 2) + (y ** 2) / (2 * sigma_y ** 2)))
            if grid[y_pix, x_pix] < 0.05:
                grid[y_pix, x_pix] = 0


@njit(parallel=True)
def twoDellipse(grid, x_0, y_0, sigma_x, sigma_y, A=1):
    for y_pix in prange(grid.shape[0]):
        for x_pix in prange(grid.shape[1]):
            x = x_pix - x_0
            y = y_pix - y_0
            if (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2) <= 1:
                grid[y_pix, x_pix] = A


def getImagePatches_forpython(rfStats, imageFileName, electrodeList, patchSizeDeg=2, viewingDistanceCM=50,
                              monitorSpecifications=None, RFtype=None):
    numElectrodes = len(electrodeList)
    if monitorSpecifications is None:
        monitorSpecifications = {"height": 11.8, "width": 20.9, "xRes": 1280, "yRes": 720}
    # min_x and min_y max_x max_y
    inputImage = plt.imread(imageFileName)[:, :, :3]
    [xAxisDeg, yAxisDeg] = getImageInDegrees(inputImage, monitorSpecifications, viewingDistanceCM)
    xResDeg = xAxisDeg[1] - xAxisDeg[0]
    yResDeg = yAxisDeg[1] - yAxisDeg[0]
    xPosToTake = int(patchSizeDeg // xResDeg)
    yPosToTake = int(patchSizeDeg // yResDeg)
    patchData = np.zeros(numElectrodes, dtype='object')
    for i in range(numElectrodes):
        rfTMP = rfStats[electrodeList[i]]
        mAzi = np.squeeze(rfTMP["meanAzi"])
        mEle = np.squeeze(rfTMP["meanEle"])
        xCenterPos = int(np.argwhere(xAxisDeg >= mAzi)[0][0])
        yCenterPos = int(np.argwhere(yAxisDeg >= -mEle)[0][0])

        if RFtype == "ellipse":
            xPosToTake1 = np.squeeze(rfTMP["rfSizeAzi"] / xResDeg)
            yPosToTake1 = np.squeeze(rfTMP["rfSizeEle"] / yResDeg)
            mask = np.zeros(shape=inputImage.shape[:-1])
            twoDellipse(mask, xCenterPos, yCenterPos, xPosToTake1, yPosToTake1)
            mask = np.stack([mask] * inputImage.shape[-1], axis=2)
            patchData[i] = np.multiply(inputImage, mask)[yCenterPos - yPosToTake:yCenterPos + yPosToTake,
                           xCenterPos - xPosToTake:xCenterPos + xPosToTake, :]
        elif RFtype == "gaussian":
            xPosToTake1 = np.squeeze(rfTMP["stdAzi"] / xResDeg)
            yPosToTake1 = np.squeeze(rfTMP["stdEle"] / yResDeg)
            mask = np.zeros(shape=inputImage.shape[:-1])
            twoDguassian(mask, xCenterPos, yCenterPos, xPosToTake1, yPosToTake1)
            mask = np.stack([mask] * inputImage.shape[-1], axis=2)
            patchData[i] = np.multiply(inputImage, mask)[yCenterPos - yPosToTake:yCenterPos + yPosToTake,
                           xCenterPos - xPosToTake:xCenterPos + xPosToTake, :]

        else:
            patchData[i] = inputImage[yCenterPos - yPosToTake:yCenterPos + yPosToTake,
                           xCenterPos - xPosToTake:xCenterPos + xPosToTake, :]
    return patchData


def getImageInDegrees(inputImage, monitorSpecifications, viewingDistanceCM):
    viewingDistance = viewingDistanceCM / 2.54
    yDeg = (math.atan((monitorSpecifications["height"] / 2) / viewingDistance)) * 180 / math.pi
    xDeg = (math.atan((monitorSpecifications["width"] / 2) / viewingDistance)) * 180 / math.pi
    imageXRes = inputImage.shape[1]
    imageYRes = inputImage.shape[0]
    if monitorSpecifications["xRes"] != imageXRes or monitorSpecifications["yRes"] != imageYRes:
        raise Exception("Image resolution does not match monitor resolution")
    xAxisDeg = np.arange(-xDeg, xDeg, (2 * xDeg / imageXRes))[:imageXRes]
    yAxisDeg = np.arange(-yDeg, yDeg, (2 * yDeg / imageYRes))[:imageYRes]
    return [xAxisDeg, yAxisDeg]


# %%
best_electrodes = np.squeeze(np.intersect1d(highRMSElectrodes[:-3], spikeChannels))
best_electrode_names = ['elec' + str(i + 1) for i in best_electrodes]
channelPos_names = [str(i) + ", SID 1" for i in best_electrodes]
channelNumber = best_electrodes
stimValsToUse = np.array([i for i in range(32)])
numStimulus = stimValsToUse.shape[0]
numElectrodes = channelNumber.shape[0]
# %%
'''
# electrodes and spikes
analogChannelString = 'elec11'
channelPos=1
channelNumber = spikeChannels[channelPos]
stimValsToUse = np.array([i for i in range(16)])
analysisType = 6
fig, ax = plt.subplots(4,4)

result=plotLFPData1Channel(plotHandles=ax.flatten(), channelString=analogChannelString, stimulus_list=stimValsToUse,analysisType=analysisType,**kwargs)

plt.show()

gammas=[]
for trial in result:
    deltaF=trial[1]/trial[0]
    fVals=trial[2]
    # select gamma band
    gammaBand=np.logical_and(fVals>=30, fVals<=80)
    deltaF_gamma=deltaF[gammaBand]
    gammas.append(np.average(deltaF_gamma))
plt.plot(gammas)
plt.show()
'''

# %%
savepath = folderSourceString + "\\python_data"
reload = False
resave = True
[imagePatches, imagePatches_e, imagePatches_g, imageFreqs, imageSpikes] = None, None, None, None, None



imageFreqs = plotLFPData1Channel(plotHandles=None, channelString=best_electrode_names, stimulus_list=stimValsToUse,
                                 analysisType=6, **kwargs)

np.savez_compressed(savepath + "\\freq_and_spikes.npz", imageFreqs=imageFreqs, imageSpikes=imageSpikes)