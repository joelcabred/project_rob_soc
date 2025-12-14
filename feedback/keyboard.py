import cv2
from .base import FeedbackBackend

class KeyboardFeedback(FeedbackBackend):
    def poll(self):
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('+'), ord('=')):
            return +1
        if key in (ord('-'), ord('_')):
            return -1
        if key == ord('0'):
            return 0

        return None

    def reset(self):
        pass
