import os
import cv2
import time


# 根据输入的文件夹绝对路径，将该文件夹下的所有指定suffix的文件读取存入一个list,
# 该list的第一个元素是该文件夹的名字
def readImg(path, *suffix):
    try:
        s = os.listdir(path)
        files = []
        fileName = os.path.basename(path)
        files.append(fileName)

        for i in s:
            if endwith(i, suffix):
                document = os.path.join(path, i)
                img = cv2.imread(document)
                files.append(img)
    except IOError:
        print("读取照片失败")

    else:
        print ("读取成功")
        return files

# 输入一个字符串一个标签，对这个字符串的后续和标签进行匹配
def endwith(s, *endstring):
   resultArray = map(s.endswith, endstring)
   if True in resultArray:
       return True
   else:
       return False


# 从源路径中读取所有图片放入一个list，然后逐一进行检查，把其中的脸扣下来，存储到目标路径中
def readPicSaveFace(sourcePath,targetPath,*suffix):
    try:

        # 读取照片,第一个元素是文件原始图片路径
        resultArray = readImg(sourcePath, *suffix)


        # 对list中图片逐一进行检查,找出其中的人脸然后写到目标文件夹下
        count = 1
        # 加载xml文件生成级联分类器face_cascade，然后用这个级联分类器对灰度图进行检测
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        for i in resultArray:
            if type(i) != str:
              gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
              # 1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
              faces = face_cascade.detectMultiScale(gray, 1.1, 5)
              for(x, y, w, h) in faces:
                fileName = str(count)   # 以索引作为文件名
                # resize_without_deformation()
                f = cv2.resize(gray[y:(y + h), x:(x + w)], (128, 128))
                cv2.imwrite(targetPath+os.sep+'%s.jpg' % fileName, f)
                count += 1

    except IOError:
        print("读取失败")
    else:
        print('已裁剪'+str(count)+' 张脸部图片'+targetPath)

if __name__ == '__main__':
    '''


    
    readPicSaveFace(r'F:\GraduationProject\pictures\raw_pictures\AndyLou',
                     r'F:\GraduationProject\pictures\dataset\AndyLou',
                     '.jpg')
    
    readPicSaveFace(r'F:\GraduationProject\pictures\raw_pictures\DanielWu',
                     r'F:\GraduationProject\pictures\dataset\DanielWu',
                     '.jpg')

    readPicSaveFace(r'F:\GraduationProject\pictures\raw_pictures\luoyufan',
                     r'F:\GraduationProject\pictures\dataset\luoyufan',
                     '.jpg')

    readPicSaveFace(r'F:\GraduationProject\pictures\raw_pictures\myface',
                    r'F:\GraduationProject\pictures\dataset\youchangxin',
                    '.jpg') 
    '''

    readPicSaveFace(r'F:\GraduationProject\pictures\raw_pictures\wuyifan',
                r'F:\GraduationProject\pictures\dataset\wuyifan',
                '.jpg')



