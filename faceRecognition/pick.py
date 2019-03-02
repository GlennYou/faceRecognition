'''
利用电脑摄像头进行人脸截图，作为raw pictrure，
截取的照片经过灰度化处理，为数据集做准备
'''

import cv2
import datetime

# 训练识别人脸
out_path = r'F:\GraduationProject\pictures\raw_pictures\wuyifan'


cap = cv2.VideoCapture(0)
faceIndex = 0
while True:
    if faceIndex < 10000:
        if faceIndex % 100 == 0:
            print('已截图 %d 张' %faceIndex)

        # get a frame
        ret, img = cap.read()
        # 需要先转换为灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("capture", gray_img)  # 显示画面

        faceIndex += 1
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # 获取当前时间戳
        cv2.imwrite(out_path +'\\' + str(nowTime) + str(faceIndex) + ".jpg", gray_img)    # 保存图片
        # 延迟一毫秒 将字符转化
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()             # 释放对象
            cv2.destroyAllWindows()   # 销毁窗口
            break
    else:
        print('完成截图')
        break
