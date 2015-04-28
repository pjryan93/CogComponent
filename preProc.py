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

class PreProcHelper(object):
    def __init__(self):
        self.datab = dataBase()
    def getLabeldDataFromSong(self,songId):
        songPath = self.datab.getSong(songId)
        dataSet = (self.getFFTforSong(songId,songPath[0][1]),self.mapNotesToKeys(songId))
        return shared_dataset2(dataSet)
    def getLabeldDataFromSongs(self,songId):
        tempX = list()
        tempY = list()
        for i in range(0,len(songId)):
            print i 
            songPath = self.datab.getSong(songId[i])
            print 'path shape'
            print shape(songPath)
            tempX.extend(self.getFFTforSong(songId[i],songPath[0][1]))
            tempY.extend(self.mapNotesToKeys(songId[i]))
        dataSet = (tempX,tempY)
        #for i in range(0,len(dataSet))
        return shared_dataset2(dataSet)
    def getFFTforSong(self,songId,songPath):
        aud = librosa.load(songPath,sr=44100.0)[0]
        indexes = self.getIndexes(songId)
        fftData = list()
        for i in range(0,len(indexes)-1):
            s1 = indexes[i]
            s2 = indexes[i+1]
            data = self.getFFT(aud,s1,s2)
            print 'song has ' + str(len(indexes)-1 - i) + " left to perform the FFT on"
            fftData.append(data)
        return asarray(fftData)
    def getFFTforSongRaw(self,songIn):
        aud = librosa.load(songIn,sr=44100.0)[0]
        indexes = self.getIndexes(songId)
        print indexes
        fftData = list()
        for i in range(0,len(indexes)-1):
            s1 = indexes[i]
            s2 = indexes[i+1]
            data = self.getFFT(aud,s1,s2)
            print 'song has ' + str(len(indexes)-1 - i) + "left to perform the FFT on"
            fftData.append(data)
        return asarray(fftData)
    def getIndexes(self,songId):
        sng = self.datab.getSong(songId)
        segs = self.datab.getSegments(songId)
        inds = zeros(len(segs)-1)
        for i in range(0,len(segs)-1):
            inds[i] = int(segs[i][2])
        print inds
        return inds
    def getFFT(self,audio,startIndex,endIndex):
        sample = audio[startIndex:endIndex]
        dataToReturn = ft.fft(sample)
        return absolute(dataToReturn[1:3001])/3000
    def mapNotesToKeys(self,songId):
        correctData = loadCorrect(self.datab.getSong(songId)[0][2])
        notes = { "C":1.0,
                  "C/2":1.0,
                  "C/3":1.0,
                  "C/5":1.0,
                  "C/6":1.0,
                  "C:7":1.0,
                  "C#":2.0,
                  "Db":2.0,
                  "D/b7":2.0,
                  "Db/5":2.0,
                  "D:7":3.0,
                  "D/6" :3.0,
                  "D":3.0,
                  "D/3":3.0,
                  "D#":4.0,
                  "Eb":4.0,
                  "Eb:7":4.0,
                  "E/b7":5.0,
                  "E/b6":5.0,
                  "Eb/6":5.0,
                  "E":5.0,
                  "E:5":5.0,
                  "E:7":5.0,
                  "E:9":5.0,
                  "Fb":5.0,
                  "F#:dim/b3":5.0,
                  "F":6.0,
                  "F/3":6.0,
                  "F:7":6.0,
                  "F/6":6.0,
                  "F/b7":6.0,
                  "F#":7.0,
                  "F#/5":7.0,
                  "F/5":6.0,
                  "Gb":7.0,
                  "G":8.0,
                  "G/2":8.0,
                  "G/3":8.0,
                  "G/5":8.0,
                  "G/6":8.0,
                  "G:7":8.0,
                  "G:9":8.0,
                  "G#":9.0,
                  "G/#4":9.0,
                  "Ab":9.0,
                  "A":10.0,
                  "A/2":10.0,
                  "A/3":10.0,
                  "A/5":10.0,
                  "A/6":10.0,
                  "A/b3":10.0,
                  "A/b7":10.0,
                  "A/b6":10.0,
                  "A:6":10.0,
                  "A:7":10.0,
                  "A:9":10.0,
                  "A#":11.0,
                  "Bb":11.0,
                  "B":12.0,
                  "B/5":12.0,
                  "B/7":12.0,
                  "B/4":12.0,
                  "Bb/5":11.0,
                  "Bb/7":11.0,
                  "Bb/9":11.0,
                  "B:7":12.0,
                  "B:9":12.0,
                  "Bb:sus4(9)":12.0,
                  "B#":12.0,
                  "Cb":12.0,
                  "N":13.0
        }
        chordsNumbs = list()
        for i in range(0,len(correctData)-1):
            chordsNumbs.append(notes[correctData[i]])
        return chordsNumbs
def mapKeysToNotes(indexIn):
    notes = asarray(["X","C","C#","D","D#","E","F","F#","G","G#","A","A#","B","B#","N"])
    return notes[indexIn]
def shared_dataset2(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
class dataBase(object):
    def __init__(self):
        self.conn = sqlite3.connect('/Users/patrickryan/cdev/proj/mirtoolkit/wsgi/myproject/db.sqlite3')
        self.c = self.conn.cursor()
        self.setup()
    def setup(self):
        self.c.execute('''CREATE TABLE if not exists songs
              (id INTEGER PRIMARY KEY,
              name TEXT,
              correctPath TEXT,
              bpm INTEGER)''')
        self.c.execute('''CREATE TABLE if not exists segments
             (position INTEGER  ,
              startTime TEXT,
              startIndex INTEGER,
              songID INTEGER,
              FOREIGN KEY(songID) REFERENCES song(id))''')
    def addSong(self,song,correctPath,id):
        if correctPath is not None:
            self.c.execute("INSERT INTO songs VALUES("+str(id)+",'"+song.path+"','"+correctPath+"',"+'"nothing"'+")")
        else:
            self.c.execute("INSERT INTO songs VALUES("+str(id)+",'"+song.path+"','"+'nothing'+"',"+'nothing'+")")
        for i in song.segments:
            self.addSegment(i,id)
        self.conn.commit()
    def addSegment(self,segment,id):
        self.c.execute("INSERT INTO segments VALUES("+str(segment.position)+","+str(segment.startTime)+","+str(segment.startIndex)+","+str(id)+")")
    def getSegments(self,songId):
        self.c.execute("SELECT * FROM segments where songID=:songId",{"songId": songId})
        return self.c.fetchall()
    def getSong(self,songId):
        self.c.execute("SELECT * FROM songs where id=:songId",{"songId": songId})
        return self.c.fetchall()
class Segment(object):
    def __init__(self,startT,startI,pos,data):
        self.startTime = startT
        self.startIndex = double(startI)
        self.position = pos
        self.data = asarray(data)
class Song(object):
    def __init__(self,audioPath,times = None):
        self.wave_file = wv.open(audioPath, 'r')
        self.numberOfChannels = self.wave_file.getnchannels()
        self.fs = 44100.0
        self.path = audioPath
        self.audio = librosa.load(audioPath,sr=self.fs)[0]
        print self.audio
        print shape(self.audio)
        print self.audio[0]
        self.beat_start = zeros(len(self.audio)/self.fs)

        print len(self.beat_start)
        if times is not None:
            self.segments = self.getSegmentsFromData(times)
        else:
            self.segments = self.getSegmentsDividedIntoSeconds()
        print len(self.segments)
        print self.segments[2].startIndex
        #self.segments = self.setSegments()
    def getSegmentsFromData(self,times):
        self.beat_start = asarray(times)
        return asarray(self.getSegmentsFromOnsets())
    def getSegmentsDividedIntoSeconds(self):
        self.beat_start = zeros(len(self.audio)/self.fs)
        self.beat_start[0] = 0
        self.beat_start = zeros(len(self.audio)/self.fs)
        self.beat_start[0] = 0
        self.segments = list()
        for i in range(1,len(self.beat_start-3)):
            print i * self.fs
            currentSegment = Segment(i,(self.fs*i),i,self.audio[i*(self.fs-1):i*(self.fs)])
            self.segments.append(currentSegment)
        self.segments = asarray(self.segments)
        return self.segments
    def getSegmentsFromOnsets(self):
        timeIndex = 0;
        segments = list()
        seg = list()
        startTime = 0.0
        increment = 1
        endTime = self.beat_start[increment]
        startIndex = 0
        for frameIndex in range(0,len(self.audio)-1):
            time = float(frameIndex)/float(self.fs)
            if  startTime < time and time < endTime:
                seg.append(self.audio[frameIndex])
            elif time > endTime:
                currentSegment = Segment(startTime,frameIndex,len(segments),(seg))
                segments.append(currentSegment)
                seg = list()
                timeIndex= timeIndex + increment
                startTime = self.beat_start[timeIndex]
                if timeIndex < len(self.beat_start)-increment:
                    endTime = self.beat_start[timeIndex+increment]
                else:
                    break
            else:
                continue
        currentSegment = Segment(startTime,startIndex,len(segments),(seg))
        segments.append(currentSegment)
        return asarray(segments)
    def setSegments(self):
        segments = self.getSegmentsFromOnsets()
        print type(asarray(segments))
        return asarray(segments)
def getPaths():
    """
    path1 = "/Users/patrickryan/cdev/cognitive/data/Blackbird.wav"
    path2 = "/Users/patrickryan/cdev/cognitive/data/backintheussr.wav"
    path3 = "/Users/patrickryan/cdev/cognitive/data/dearprudence.wav"
    path4 = "/Users/patrickryan/cdev/cognitive/data/whilemyguitar.wav"
    path5 = "/Users/patrickryan/cdev/cognitive/data/GlassOnion.wav"
    path6 = "/Users/patrickryan/cdev/cognitive/data/Ob-La-DiOb-La-Da.wav"
    path7 = "/Users/patrickryan/cdev/cognitive/data/WildHoneyPie.wav"
    path8 = "/Users/patrickryan/cdev/cognitive/data/TheContinuingStoryOfBungalowBill.wav"

    


    tPath1 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_11_-_Black_Bird.lab"
    tPath2 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_01_-_Back_in_the_USSR.lab"
    tPath3 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_02_-_Dear_Prudence.lab"
    tPath4 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_07_-_While_My_Guitar_Gently_Weeps.lab"
    tPath5 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_03_-_Glass_Onion.lab"
    tPath6 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_04_-_Ob-La-Di,_Ob-La-Da.lab"
    tPath7 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_05_-_Wild_Honey_Pie.lab"
    tPath8 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_06_-_The_Continuing_Story_of_Bungalow_Bill.lab"
    
    path1 = "/Users/patrickryan/cdev/cognitive/data/01Birthday.wav"
    path2 = "/Users/patrickryan/cdev/cognitive/data/02YerBlues.wav"
    path3 = "/Users/patrickryan/cdev/cognitive/data/03MotherNaturesSon.wav"
    path4 = "/Users/patrickryan/cdev/cognitive/data/04EverybodysGotSomethingtoHideExceptMeandMyMonkey.wav"
    path5 = "/Users/patrickryan/cdev/cognitive/data/05SexySadie.wav"
    path6 = "/Users/patrickryan/cdev/cognitive/data/06HelterSkelter.wav"
    path7 = "/Users/patrickryan/cdev/cognitive/data/07Long,Long,Long.wav"
    path8 = "/Users/patrickryan/cdev/cognitive/data/08HappinessIsAWarmGun.wav"
    path9 = "/Users/patrickryan/cdev/cognitive/data/09MarthaMyDear.wav"
    path10 = "/Users/patrickryan/cdev/cognitive/data/10ImSoTired.wav"
    path12 = "/Users/patrickryan/cdev/cognitive/data/12Piggies.wav"
    path13 = "/Users/patrickryan/cdev/cognitive/data/13RockyRaccoon.wav"
    path14 = "/Users/patrickryan/cdev/cognitive/data/14DontPassMeBy.wav"
    path15 = "/Users/patrickryan/cdev/cognitive/data/15WhyDontWeDoItIntheRoad_.wav"
    path16 = "/Users/patrickryan/cdev/cognitive/data/16IWill.wav"
    path17 = "/Users/patrickryan/cdev/cognitive/data/17Julia.wav"


    tPath1 = "/Users/patrickryan/cdev/cognitive/data/10CD2_-_The_Beatles/CD2_-_01_-_Birthday.lab"
    tPath2 = "/Users/patrickryan/cdev/cognitive/data/10CD2_-_The_Beatles/CD2_-_02_-_Yer_Blues.lab"
    tPath3 = "/Users/patrickryan/cdev/cognitive/data/10CD2_-_The_Beatles/CD2_-_03_-_Mother_Natures_Son.lab"
    tPath4 = "/Users/patrickryan/cdev/cognitive/data/10CD2_-_The_Beatles/CD2_-_04_-_Everybodys_Got_Something_To_Hide_Except_Me_and_My_Monkey.lab"
    tPath5 = "/Users/patrickryan/cdev/cognitive/data/10CD2_-_The_Beatles/CD2_-_05_-_Sexy_Sadie.lab"
    tPath6 = "/Users/patrickryan/cdev/cognitive/data/10CD2_-_The_Beatles/CD2_-_06_-_Helter_Skelter.lab"
    tPath7 = "/Users/patrickryan/cdev/cognitive/data/10CD2_-_The_Beatles/CD2_-_07_-_Long_Long_Long.lab"
    tPath8 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_08_-_Happiness_is_a_Warm_Gun.lab"
    tPath9 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_09_-_Martha_My_Dear.lab"
    tPath10 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_10_-_Im_So_Tired.lab"
    tPath11 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_12_-_Piggies.lab"
    tPath12 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_13_-_Rocky_Raccoon.lab"
    tPath13 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_14_-_Dont_Pass_Me_By.lab"
    tPath14 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_15_-_Why_Dont_We_Do_It_In_The_Road.lab"
    tPath15 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_16_-_I_Will.lab"
    tPath16 = "/Users/patrickryan/cdev/cognitive/data/10CD1_-_The_Beatles/CD1_-_17_-_Julia.lab"
    """

    path1 = "/Users/patrickryan/cdev/cognitive/data/help/01Help!.wav"
    path2 = "/Users/patrickryan/cdev/cognitive/data/help/02TheNightBefore.wav"
    path3 = "/Users/patrickryan/cdev/cognitive/data/help/03YouveGottoHideYourLoveAway.wav"
    path4 = "/Users/patrickryan/cdev/cognitive/data/help/04INeedYou.wav"
    path5 = "/Users/patrickryan/cdev/cognitive/data/help/05AnotherGirl.wav"
    path6 = "/Users/patrickryan/cdev/cognitive/data/help/06YoureGoingtoLoseThatGirl.wav"
    path7 = "/Users/patrickryan/cdev/cognitive/data/help/07TickettoRide.wav"
    path8 = "/Users/patrickryan/cdev/cognitive/data/help/08ActNaturally.wav"
    path9 = "/Users/patrickryan/cdev/cognitive/data/help/09ItsOnlyLove.wav"
    path10 = "/Users/patrickryan/cdev/cognitive/data/help/10YouLikeMeTooMuch.wav"
    path11 = "/Users/patrickryan/cdev/cognitive/data/help/11TellMeWhatYouSee.wav"
    path12 = "/Users/patrickryan/cdev/cognitive/data/help/12IveJustSeenaFace.wav"
    path13 = "/Users/patrickryan/cdev/cognitive/data/help/13Yesterday.wav"
    path14 = "/Users/patrickryan/cdev/cognitive/data/help/14DizzyMissLizzy.wav"



    tPath1 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/01_-_Help!.lab"
    tPath2 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/02_-_The_Night_Before.lab"
    tPath3 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/03_-_Youve_Got_To_Hide_Your_Love_Away.lab"
    tPath4 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/04_-_I_Need_you.lab"
    tPath5 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/05_-_Another_Girl.lab"
    tPath6 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/06_-_Youre_Going_To_Lose_That_Girl.lab"
    tPath7 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/07_-_Ticket_to_Ride.lab"
    tPath8 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/08_-_Act_Naturally.lab"
    tPath9 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/09_-_Its_Only_Love.lab"
    tPath10 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/10_-_You_Like_Me_Too_Much.lab"
    tPath11 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/11_-_Tell_Me_What_You_See.lab"
    tPath12 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/12_-_Ive_Just_Seen_a_Face.lab"
    tPath13 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/13_-_Yesterday.lab"
    tPath14 = "/Users/patrickryan/cdev/cognitive/data/05_-_Help!/14_-_Dizzy_Miss_LIzzy.lab"





    pathsX = list()
    pathsX.append(path1)
    pathsX.append(path2)
    pathsX.append(path3)
    pathsX.append(path4)
    pathsX.append(path5)
    pathsX.append(path6)
    pathsX.append(path7)
    pathsX.append(path8)
    pathsX.append(path9)
    pathsX.append(path10)
    pathsX.append(path11)
    pathsX.append(path12)
    pathsX.append(path13)
    pathsX.append(path14)

    pathsY = list()
    pathsY.append(tPath1)
    pathsY.append(tPath2)
    pathsY.append(tPath3)
    pathsY.append(tPath4)
    pathsY.append(tPath5)
    pathsY.append(tPath6)
    pathsY.append(tPath7)
    pathsY.append(tPath8)
    pathsY.append(tPath9)
    pathsY.append(tPath10)
    pathsY.append(tPath11)
    pathsY.append(tPath12)
    pathsY.append(tPath13)
    pathsY.append(tPath14)


    songsList = list()
    #songsList.append(100)
    songsList.append(101)
    songsList.append(102)
    songsList.append(103)
    songsList.append(104)
    songsList.append(105)
    songsList.append(106)
    songsList.append(107)
    songsList.append(108)
    songsList.append(109)
    songsList.append(110)
    songsList.append(111)
    songsList.append(112)
    songsList.append(113)
    songsList.append(114)
    songsList.append(115)
    songsList.append(116)
    return pathsX, pathsY
def getMoreData():
    """
    path1 = "/Users/patrickryan/cdev/cognitive/pepper/01SgtPeppersLonelyHeartsClubBand.wav"
    path2 = "/Users/patrickryan/cdev/cognitive/pepper/02WithaLittleHelpFromMyFriends.wav"
    path3 = "/Users/patrickryan/cdev/cognitive/pepper/03LucyIntheSkyWithDiamonds.wav"
    path4 = "/Users/patrickryan/cdev/cognitive/pepper/04GettingBetter.wav"
    path5 = "/Users/patrickryan/cdev/cognitive/pepper/05FixingaHole.wav"
    path6 = "/Users/patrickryan/cdev/cognitive/pepper/06ShesLeavingHome.wav"
    path7 = "/Users/patrickryan/cdev/cognitive/pepper/07BeingFortheBenefitofMrKite!.wav"
    path8 = "/Users/patrickryan/cdev/cognitive/pepper/08WithinYouWithoutYou.wav"
    path9 = "/Users/patrickryan/cdev/cognitive/pepper/09WhenImSixtyFour.wav"
    path10 = "/Users/patrickryan/cdev/cognitive/pepper/10LovelyRita.wav"
    path11 = "/Users/patrickryan/cdev/cognitive/pepper/11GoodMorningGoodMorning.wav"
    path12 = "/Users/patrickryan/cdev/cognitive/pepper/13ADayIntheLife.wav"

    tPath1 = "/Users/patrickryan/cdev/cognitive/data/08_-_Sgt._Peppers_Lonely_Hearts_Club_Band/01_-_Sgt._Peppers_Lonely_Hearts_Club_Band.lab"
    tPath2 = "/Users/patrickryan/cdev/cognitive/data/08_-_Sgt._Peppers_Lonely_Hearts_Club_Band/02_-_With_A_Little_Help_From_My_Friends.lab"
    tPath3 = "/Users/patrickryan/cdev/cognitive/data/08_-_Sgt._Peppers_Lonely_Hearts_Club_Band/03_-_Lucy_In_The_Sky_With_Diamonds.lab"
    tPath4 = "/Users/patrickryan/cdev/cognitive/data/08_-_Sgt._Peppers_Lonely_Hearts_Club_Band/04_-_Getting_Better.lab"
    tPath5 = "/Users/patrickryan/cdev/cognitive/data/08_-_Sgt._Peppers_Lonely_Hearts_Club_Band/05_-_Fixing_A_Hole.lab"
    tPath6 = "/Users/patrickryan/cdev/cognitive/data/08_-_Sgt._Peppers_Lonely_Hearts_Club_Band/06_-_Shes_Leaving_Home.lab"
    tPath7 = "/Users/patrickryan/cdev/cognitive/data/08_-_Sgt._Peppers_Lonely_Hearts_Club_Band/07_-_Being_For_The_Benefit_Of_Mr._Kite!.lab"
    tPath8 = "/Users/patrickryan/cdev/cognitive/data/08_-_Sgt._Peppers_Lonely_Hearts_Club_Band/08_-_Within_You_Without_You.lab"
    tPath9 = "/Users/patrickryan/cdev/cognitive/data/08_-_Sgt._Peppers_Lonely_Hearts_Club_Band/09_-_When_Im_Sixty-Four.lab"
    tPath10 = "/Users/patrickryan/cdev/cognitive/data/08_-_Sgt._Peppers_Lonely_Hearts_Club_Band/10_-_Lovely_Rita.lab"
    tPath11 = "/Users/patrickryan/cdev/cognitive/data/08_-_Sgt._Peppers_Lonely_Hearts_Club_Band/11_-_Good_Morning_Good_Morning.lab"
    tPath12 = "/Users/patrickryan/cdev/cognitive/data/08_-_Sgt._Peppers_Lonely_Hearts_Club_Band/13_-_A_Day_In_The_Life.lab"
    """
    path1 = "/Users/patrickryan/cdev/cognitive/abbeyroad/01ComeTogether.wav"
    path2 = "/Users/patrickryan/cdev/cognitive/abbeyroad/02Something.wav"
    path3 = "/Users/patrickryan/cdev/cognitive/abbeyroad/03MaxwellsSilverHammer.wav"
    path4 = "/Users/patrickryan/cdev/cognitive/abbeyroad/04Oh!Darling.wav"
    path5 = "/Users/patrickryan/cdev/cognitive/abbeyroad/05OctopussGarden.wav"
    path6 = "/Users/patrickryan/cdev/cognitive/abbeyroad/06IWantYou(ShesSoHeavy).wav"
    path7 = "/Users/patrickryan/cdev/cognitive/abbeyroad/07HereComestheSun.wav"
    path8 = "/Users/patrickryan/cdev/cognitive/abbeyroad/08Because.wav"
    path9 = "/Users/patrickryan/cdev/cognitive/abbeyroad/09YouNeverGiveMeYourMoney.wav"
    path10 = "/Users/patrickryan/cdev/cognitive/abbeyroad/10SunKing.wav"
    path11 = "/Users/patrickryan/cdev/cognitive/abbeyroad/11MeanMr.Mustard.wav"
    path12 = "/Users/patrickryan/cdev/cognitive/abbeyroad/12PolythenePam.wav"
    path13 = "/Users/patrickryan/cdev/cognitive/abbeyroad/13SheCameInThroughTheBathroomWindow.wav"
    path14 = "/Users/patrickryan/cdev/cognitive/abbeyroad/14GoldenSlumbers.wav"
    path15 = "/Users/patrickryan/cdev/cognitive/abbeyroad/15CarryThatWeight.wav"
    path16 = "/Users/patrickryan/cdev/cognitive/abbeyroad/16TheEnd.wav"
    path17 = "/Users/patrickryan/cdev/cognitive/abbeyroad/17HerMajesty.wav"

    tPath1 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/01_-_Come_Together.lab"
    tPath2 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/02_-_Something.lab"
    tPath3 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/03_-_Maxwells_Silver_Hammer.lab"
    tPath4 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/04_-_Oh!_Darling.lab"
    tPath5 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/05_-_Octopuss_Garden.lab"
    tPath6 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/06_-_I_Want_You.lab"
    tPath7 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/07_-_Here_Comes_The_Sun.lab"
    tPath8 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/08_-_Because.lab"
    tPath9 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/09_-_You_Never_Give_Me_Your_Money.lab"
    tPath10 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/10_-_Sun_King.lab"
    tPath11 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/11_-_Mean_Mr_Mustard.lab"
    tPath12 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/12_-_Polythene_Pam.lab"
    tPath13 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/13_-_She_Came_In_Through_The_Bathroom_Window.lab"
    tPath14 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/14_-_Golden_Slumbers.lab"
    tPath15 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/15_-_Carry_That_Weight.lab"
    tPath16 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/16_-_The_End.lab"
    tPath17 = "/Users/patrickryan/cdev/cognitive/11_-_Abbey_Road/17_-_Her_Majesty.lab"

    pathsX = list()
    pathsX.append(path1)
    pathsX.append(path2)
    pathsX.append(path3)
    pathsX.append(path4)
    pathsX.append(path5)
    pathsX.append(path6)
    pathsX.append(path7)
    pathsX.append(path8)
    pathsX.append(path9)
    pathsX.append(path10)
    pathsX.append(path11)
    pathsX.append(path12)
    pathsX.append(path13)
    pathsX.append(path14)
    pathsX.append(path15)
    pathsX.append(path16)
    pathsX.append(path17)

    pathsY = list()
    pathsY.append(tPath1)
    pathsY.append(tPath2)
    pathsY.append(tPath3)
    pathsY.append(tPath4)
    pathsY.append(tPath5)
    pathsY.append(tPath6)
    pathsY.append(tPath7)
    pathsY.append(tPath8)
    pathsY.append(tPath9)
    pathsY.append(tPath10)
    pathsY.append(tPath11)
    pathsY.append(tPath12)
    pathsY.append(tPath13)
    pathsY.append(tPath14)
    pathsY.append(tPath15)
    pathsY.append(tPath16)
    pathsY.append(tPath17)
    for i in pathsX:
        f = file(i,'rb')
        f.close()
    for i in pathsY:
        f = file(i,'rb')
        f.close()
    return pathsX, pathsY

def getTimes(filePath):
    crs = open(filePath, "r")
    data = list()
    for columns in ( raw.strip().split() for raw in crs ):  
        data.append(float(columns[0]))
    return data
def loadCorrect(filePath):
    crs = open(filePath, "r")
    data = list()
    counter = 0
    for columns in ( raw.strip().split() for raw in crs ):
        if counter == 0:
            counter= counter + 1
            continue
        elif len(columns[2]) == 3:
            note = columns[2].split('/',1)[0]
            data.append(note)
        else:
            note = columns[2].split(':',1)[0]
            data.append(note)
    return data 
def addSongs():
    paths, correctPaths = getMoreData()
    helper = PreProcHelper()
    for i in range(0,len(paths)):
       songBuffer = Song(paths[i],getTimes(correctPaths[i]))
       helper.datab.addSong(songBuffer,correctPaths[i],i+550)
def getSongsSet(songId):
    helper = PreProcHelper()
    return helper.getLabeldDataFromSong(songId)
def getSongsSets(songIds):
    helper = PreProcHelper()
    return helper.getLabeldDataFromSongs(songIds)

def startProcess():
    paths, correctPaths = getPaths()
    helper = PreProcHelper()
    for i in range(0,len(paths)):
       songBuffer = Song(paths[i],getTimes(correctPaths[i]))
       helper.datab.addSong(songBuffer,correctPaths[i],i+90)

#addSongs()
"""
data = list()
for i in range(30,33):
    data.append(getSongsSet(i))
"""

