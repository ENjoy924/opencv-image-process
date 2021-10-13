import time

import cv2
import imutils
import numpy as np
from imutils.video import FPS
from threading import Thread
import sys
from queue import Queue


# stream = cv2.VideoCapture('video/1.mp4')
# ff = FPS().start()
# while True:
#     grab, frame = stream.read()
#     if not grab:
#         break
#     frame = imutils.resize(frame, width=450)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = np.dstack([frame, frame, frame])
#     cv2.putText(frame, 'slow method', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv2.imshow('', frame)
#     cv2.waitKey(1)
#     ff.update()
# ff.stop()
# print('info elapsed time :{:.2f}'.format(ff.elapsed()))
# print('FPS:{:.2f}'.format(ff.fps()))
# stream.release()


class FileVideoStream():
    def __init__(self, path, queueSize=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.q = Queue(maxsize=queueSize)

    def start(self):
        t = Thread(target=self.update, args={})
        # True：当主线程结束时不会等待子线程结束
        # False:主线程结束会等待子线程完毕在结束
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            if not self.q.full():
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    self.stop()
                    return
                self.q.put(frame)

    def read(self):
        return self.q.get()

    def more(self):
        return self.q.qsize() > 0

    def stop(self):
        self.stopped = True


fvs = FileVideoStream('video/1.mp4',queueSize=1024)
fvs.start()
time.sleep(0.1)
ff = FPS().start()

while fvs.more():
    frame = fvs.read()
    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])
    cv2.putText(frame, 'Queue size:{}'.format(fvs.q.qsize()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    ff.update()

ff.stop()
print('elasped time:{:.2f}'.format(ff.elapsed()))
print('approx, FPS:{:.2f}'.format(ff.fps()))
