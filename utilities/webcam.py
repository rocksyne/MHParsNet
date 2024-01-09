"""
Credit: https://github.com/PyImageSearch/imutils/blob/master/imutils/video/webcamvideostream.py

Author: Rockson Ayeman (rockson.agyeman@aau.at, rocksyne@gmail.com)
        Bernhard Rinner (bernhard.rinner@aau.at)

For:    Pervasive Computing Group (https://nes.aau.at/?page_id=6065)
        Institute of Networked and Embedded Systems (NES)
        University of Klagenfurt, 9020 Klagenfurt, Austria.

Date:   Thursday 3rd Aug. 2023 (First authored date)

Documentation:
--------------------------------------------------------------------------------
Increasing webcam FPS with Python and OpenCV. 
See https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

TODO:   [x] Do proper documentation
        [x] Search where code base was gotten from and credit appropriately
"""

# import the necessary packages
from threading import Thread
import cv2

class WebcamVideoStream:
    def __init__(self, src=0, name="WebcamVideoStream", resolution="480p"):

        # see https://typito.com/blog/video-resolutions/ for resolutions
        video_resolutions = {'360p':[640,360],
                            '480p':[640,480], # {'res_key':[width, height]}
                            '720p':[1280,720],
                            '1080p':[1920,1080]}
        
        width, height = video_resolutions[resolution]
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True