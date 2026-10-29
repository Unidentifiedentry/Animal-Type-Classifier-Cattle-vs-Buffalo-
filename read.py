import cv2 as cv
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
capture = cv.VideoCapture(r'D:/misc/Downloads/Vids/Sabrina Carpenter.mp4')
while True:
    isTrue, frame = capture.read()
    frame = rescaleFrame(frame)
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
    audio = capture.get(cv.CAP_PROP_POS_MSEC) + 1000
    capture.set(cv.CAP_PROP_POS_MSEC, audio)
capture.release()
cv.destroyAllWindows()