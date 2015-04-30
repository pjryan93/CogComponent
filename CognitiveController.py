from scipy.fftpack import fft, rfft
import sqlite3
from librosa.beat import *
from numpy import *
import wave as wv
from sda import SdA
import theano
import time
import theano.tensor as T
import sys
import numpy.fft as ft
import os
from random import randint
import cPickle
from preProc import *
from sda import *

def train(datasets):
    datasets = asarray(datasets)
    train = datasets[0]
    train_set_x = train[0]
    numpy_rng = random.RandomState(89677)
    sda = SdA(
            numpy_rng=numpy_rng,
            n_ins=3000,
            #hidden_layers_sizes=[1300, 1200, 1000,900,500,500,500,500,500,400],
            #hidden_layers_sizes=[2300, 2300, 2300,2300,2300],
            #hidden_layers_sizes=[2600, 2500, 2200,2200,2200],
            #hidden_layers_sizes=[2700,2700,2700,2700],
            hidden_layers_sizes=[1600,1600,1600],
            n_outs=14
    )
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    print n_train_batches
    batch_size = 1
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,batch_size=batch_size)
    print '... pre-training the model'
    start_time = time.clock()
    corruption_levels = [.1,.2,.3]
    pretraining_epochs = 10
    pretrain_lr = 0.1
    for i in xrange(sda.n_layers):
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                             corruption=corruption_levels[i],
                             lr=pretrain_lr))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print mean(c)

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

    print '... getting the finetuning functions'
    train_fn, validate_model, test_model  = sda.build_finetune_functions(
            datasets=datasets,
            batch_size=batch_size,
            learning_rate=0.2
    )

    patience = 20 * n_train_batches  # look as this many examples regardless
    patience_increase = 1.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    training_epochs=20,
    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and ite           best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    return sda
def getFFT(audio,startIndex,endIndex):
        sample = audio[startIndex:endIndex]
        dataToReturn = ft.fft(sample)
        return absolute(dataToReturn[1:3001])/3000

def getIndexesOfRaw(self,songIn):
        inds = zeros(len(songIn.segments)-1)
        for i in range(0,len(inds)-1):
            inds[i] = int(songIn.segments[i].startIndex)
        print inds
        return inds

def getMaxIndex(chordsIn):
    print 'update'
    print chordsIn
    maxIn = chordsIn[0]
    index = 0
    for i in range(0,len(chordsIn)-1):
        if chordsIn[i] > maxIn:
           # print 'chords'
            #print chordsIn[i]
            #print 'index'
            index = i
            #print index
            maxIn = chordsIn[i]
    return (index,maxIn)
def getAllChordIndexes(chordList):
    chordsToReturn = list()
    print chordsList
    for i in range(0,len(chordList)-1):
        index , conf = getMaxIndex(chordList)
        chordsToReturn.append(index,conf)
    return chordsToReturn
def prac():
    #aud = Song("/Users/patrickryan/cdev/proj/mirtoolkit/wgi/myproject/userInterface/cognitive/data/whilemyguitar.wav")
    #segments = aud.segments
   # test = getSongsSet(90)
    #four = getSongsSet(93)
    #five = getSongsSet(94)s
    """
    f = file('abbeyRoadFFT2.dat','wb')
    x = list()
    for i in range(557,566):
        x.append(i)
    six = getSongsSets(x)

    pickleDict = dict()
    pickleDict['x'] = six[0].eval()
    pickleDict['y'] = six[1]

    cPickle.dump(pd, file('validationSet.dat','wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    #seven = getSongsSet(96)
    #eight = getSongsSet(97)
    # x_Data_set.extend(five[0])
    print 'length'
    print shape(six)
    print len(six)
    return pickleDict
    """
    h = file('train.dat','rb')
    f = file('biggestDataSet.dat','rb')
    g = file('testingSet.dat','rb')

    bigData = cPickle.load(h)
    print bigData
    x0 = bigData['x']/10
    print len(x0)
    y0 = bigData['y']
    d0 = shared_dataset2((x0,y0))
    h.close()

    sixDict= cPickle.load(f)
    f.close()
    x_data = sixDict['x']/10
    y_data = sixDict['y']


    y_data = asarray(y_data)
    six = (x_data,y_data)
    six = shared_dataset2(six)
    sets =list()
    sets.append(d0)    #two = getSongsSet(91)
    #three = getSongsSet(92)
    pDict= cPickle.load(g)
    x1 = pDict['x']/10
    y1 = pDict['y'].eval()
    dset = shared_dataset2((x1,y1))
    sets.append(six)
    sets.append(dset)
    sets = asarray(sets)
    userSda = train(sets)
    return userSda
    

def makePredictions(aud,userSda):
    segments = aud.segments
    chords = list()
    for i in range(0,len(segments)-1):
        inData = getFFT(aud.audio,segments[i].startIndex,segments[i+1].startIndex)
        pred = getPrediction(inData)
        chords.append(pred)
    return chords
def getScore(chords):
    score = list()
    for i in range(0,len(chords)):
        #print chords
        tempBuff = asarray(chords[i][0])
        #print 'tempBuff'
        #print tempBuff
        #maxes, confs = getMaxIndex(tempBuff)
        noteVal = mapKeysToNotes(tempBuff)
        score.append(noteVal)
    return score
def getResult(songPath):
    aud = Song(songPath)
    myCog = prac()
    chords = makePredictions(aud,myCog)
    score = getScore(chords)
    return score
def testing():
    aud = Song("/Users/patrickryan/cdev/proj/mirtoolkit/wsgi/myproject/userInterface/cognitive/data/whilemyguitar.wav")
    myCog = prac()
    chords = makePredictions(aud,myCog)
    score = getScore(chords)
    return chords, myCog, score
def getRandomResults():
        notes = asarray(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B","B#","N"])
        spacer = randint(2,9)
        retValue  = " "
        for i in range(0,40 + spacer):
            index = randint(1,12) 
            retValue = retValue + " " + notes[index]
        print retValue
        return retValue
def getHiddenParams(paramsDict):
    hiddenLayerW = dict()
    hiddenLayerb = dict()
    logLayerW = list()
    logLayerb = list()
    print type(paramsDict)
    for key in paramsDict.keys():
        if "Whidden" in key:
            hiddenLayerW[key[-1:]] = paramsDict[key]
        elif "bhidden" in key:
            hiddenLayerb[key[-1:]] = paramsDict[key]
        elif "Wlog" in key:
            logLayerW.append(paramsDict[key])
        elif "blog" in key:
            logLayerb.append(paramsDict[key])
    orderedHidden = list()
    logLayer = list()
    for i in range(0,len(hiddenLayerW)):
        tempLayer = simpleHiddenLayer(hiddenLayerW[str(i)],hiddenLayerb[str(i)],i)
        orderedHidden.append(tempLayer)
    if len(logLayerW) > 1:
        print "opps"
    logLayer.append(simpleLogLayer(logLayerW[0],logLayerb[0],0))
    return orderedHidden , logLayer

class simpleHiddenLayer(object):
    def __init__(self,weightIn,biasIn,pos):
        self.W = weightIn
        self.b = biasIn
        self.pos = pos
    def getOutput(self,dataIn):
        return T.nnet.sigmoid(add(dot(dataIn,self.W), self.b)).eval()
class simpleLogLayer(object):
    def __init__(self,weightIn,biasIn,pos):
        self.W = weightIn
        self.b = biasIn
        self.pos = pos
    def getOutput(self,inData):
        #print "W"
        #print self.W
        #add(dot(inData,self.W), self.b)
        ret = T.nnet.softmax(add(dot(inData,self.W), self.b)).eval()
        #T.argmax(
        #,axis=1).eval()
        print ret.shape
        print ret
        return ret
def getPrediction(inp):
    f = file('sdaParams10.dat','rb')
    params = cPickle.load(f)
    print 'output'
    pDict = dict()
    for i in params:
       pDict[i.name] = i.eval()
    layers = getHiddenParams(pDict)
    hidden = layers[0]
    print hidden
    log = layers[1]
    output = inp
    for i in hidden:
        #print 'output'
        #print output
        output = i.getOutput(output)
    print 'out'
    print shape(log)
    output = log[0].getOutput(output)
    return output
def getInde(data):
    for i in range(0,len(data)):
        if data[i] != 1:
            print i
import numpy
def smooth(x,window_len=11,window='hanning'):
        if x.ndim != 1:
                raise ValueError, "smooth only accepts 1 dimension arrays."
        if x.size < window_len:
                raise ValueError, "Input vector needs to be bigger than window size."
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        s=numpy.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=numpy.ones(window_len,'d')
        else:  
                w=eval('numpy.'+window+'(window_len)')
        y=numpy.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]
def setUpSda():
    #addSongs()
    myCog = prac()
    params = get_params(myCog.params)
    save_params(params,'sdaParams.dat')
def loadDict(path):
    f = file(path,'rb')
    ret = cPickle.load(f)
    f.close()
    return ret

if __name__ == '__main__':
    t = prac()
