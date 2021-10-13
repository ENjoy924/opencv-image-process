import datetime
import time
from threading import Thread
import cv2
from queue import Queue


class FPS():
    def __init__(self):
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._numFrames += 1

    def elapsed(self):
        return (self._end - self._start).total_seconds()

    def fps(self):
        return self._numFrames / self.elapsed()


class WebCamStream():
    def __init__(self, path, maxSize=256):
        self.stream = cv2.VideoCapture(path)
        self.q = Queue(maxsize=maxSize)
        self.s = Queue(maxsize=maxSize)
        self.stoped = True

    def start(self):
        thread = Thread(target=self.update, args={})
        thread.daemon = True
        sThread = Thread(target=self.imshow, args={})
        sThread.daemon = True
        self.stoped = False
        thread.start()
        sThread.start()

    def update(self):
        while True:
            if self.stoped:
                break
            graded, frame = self.stream.read()
            if not graded:
                self.stop()
            self.q.put(frame)

    def read(self):
        if self.more():
            return self.q.get()

    def stop(self):
        self.stoped = True

    def more(self):
        return self.q.qsize() > 0

    def show(self, frame):
        if not self.stoped:
            self.s.put(frame)

    def imshow(self):
        while True:
            if self.stoped:
                break
            frame = self.s.get()
            cv2.imshow('', frame)
            cv2.waitKey(0.01)


webcam = WebCamStream(path='video/1.mp4')
webcam.start()
time.sleep(0.1)
ff = FPS()
ff.start()
while True:
    frame = webcam.read()
    if frame is not None:
        # imshow也是一个耗时操作
        webcam.show(frame)
        ff.update()
    else:
        break
ff.stop()
print('spend time:{}'.format(ff.elapsed()))
print('fps:{}'.format(ff.fps()))
