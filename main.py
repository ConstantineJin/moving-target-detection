import numpy as np
import cv2 as cv

# 读入视频
video = cv.VideoCapture("./test/test.avi")
frame_width = int(video.get(3))  # 宽度
frame_height = int(video.get(4))  # 高度
frame_rate = int(video.get(5))  # 帧数
frame_interval = 1000 / frame_rate  # 帧间隔

# 创建输出
out = cv.VideoWriter('./result/result.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, (frame_width, frame_height))
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
# 混合高斯模型
mog = cv.createBackgroundSubtractorMOG2()

while video.isOpened():
    ret, frame = video.read()
    mog_mask = mog.apply(frame)
    # 先腐蚀再膨胀去噪点
    mog_mask = cv.morphologyEx(mog_mask, cv.MORPH_OPEN, kernel)
    cv.imshow('mog_mask', mog_mask)
    contours, hierarchy = cv.findContours(mog_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        perimeter = cv.arcLength(c, True)
        # 超参数perimeter影响TPR等
        if perimeter > 100:
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if ret:
        out.write(frame)
        cv.imshow('frame', frame)
    if cv.waitKey(100) & 0xFF == ord('q'):
        break
video.release()
out.release()
cv.destroyAllWindows()
