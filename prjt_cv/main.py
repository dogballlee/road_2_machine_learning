import cv2
import numpy as np

im_path = r'D:\\download\\material\\02.jpg'     #zsanett妹妹真美
# im_path = r'D:\download\\face.jpg'    #边缘检测使用这张图像
things_dtct_path = r'C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
#opencv用以检测物体的自带xml文件(训练好的分类器)，诸如人脸、猫咪、狗子、眼睛等
vid_path = r'F:\\PORN\\chanel preston\\A day with Chanel Preston\\videos\\Scene5-1-b.mp4'       #CHANEL进餐，老美食家了


#读取图片
# img = cv2.imread(im_path)
# cv2.imshow('zsanett',img)
# cv2.waitKey(0)


#q读取视频，此处为MP4
# vid = cv2.VideoCapture(vid_path)      #如果参数填0，就会调用默认的摄像头(如果你的机器有摄像头的话)
#
# while True:
#     success,img = vid.read()
#     cv2.imshow("chanel",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


#改变图片显示方式----------灰度图/高斯模糊/轮廓图/膨胀&腐蚀操作
# kernel = np.ones((5,5),np.uint8)
#
# img = cv2.imread(im_path)
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    #转换为灰度图,opencv的颜色格式是BGR，不是传统的RGB
# imgBlur = cv2.GaussianBlur(imgGray,(9,9),1)    #高斯模糊，需要定义核大小，必须是奇数。核值越大越模糊
# imgCanny = cv2.Canny(imgGray,150,200)    #轮廓图，数值越大轮廓线越少
# imgDilation = cv2.dilate(imgGray,kernel,iterations=1)    #膨胀操作，使用kernel滑过整张图，可以连通白色区域的边界，迭代次数越多，白色区域越大。类似的有腐蚀操作erode可以用来去除毛边
# imgEroded = cv2.erode(imgDilation,kernel,iterations=1)      #腐蚀操作
#
# cv2.imshow("gray_zsanett",imgGray)
# cv2.imshow("blur_zsanett",imgBlur)
# cv2.imshow("canny_zsanett",imgCanny)
# cv2.imshow("dilate_zsanett",imgDilation)
# cv2.imshow("erote_zsanett",imgEroded)
# cv2.waitKey(0)


#图像尺寸,宽高----------从上到下从左到右
# img = cv2.imread(im_path)
# print(img.shape)    #shape显示的是高、宽、通道数
# imgResize = cv2.resize(img,(341,512))   #此处设置是按照宽、高排列，正好相反
# imgcropped = img[80:215, 200:350]    #裁剪图像的一部分，从上到下，从左到右
#
# cv2.imshow("original",img)
# # cv2.imshow("resize",imgResize)
# cv2.imshow("cropped",imgcropped)
# cv2.waitKey(0)


#在图像上绘制图形&输入文本----------按BGR的顺序排列，颜色取值范围在0~255之间
# img = np.zeros((512,512,3),np.uint8)
# # print(img)
# print(img.shape,img.shape[1],img.shape[0])
# # img[0:200,300:500] = 0,0,255    #给指定区域着色
# # img[:] = 0,0,255      #给整张图着色

# cv2.line(img,(0,0),(300,500),(0,126,0),5)     #从左上(0,0)开始画一条直线，到(300,500)结束，先宽后高，颜色是(0,126,0)，线粗细为5
# cv2.rectangle(img,(0,0),(250,350),(0,0,255),cv2.FILLED)    #画一个矩形框，从(0,0)开始，横着走250，纵着走350，红色线，不填充是个框，填充了是个色块
# cv2.circle(img,(400,50),30,(255,255,255),5)     #画个白圈圈，圆心是(400,50)，半径30，线粗细为5
# cv2.putText(img,"OPENCV",(250,250),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)    #输入文字OPENCV，在(250,250)处，使用cv2自带字体，比例尺为1（控制字体大小），蓝色，线条粗细为2
#
# cv2.imshow("img",img)
# cv2.waitKey(0)


#图像拉伸,把歪歪的图拉直
# img = cv2.imread(im_path)
# # print(img.shape)
# width,height = 350,350
# pts1 = np.float32([[39,891],[180,824],[135,976],[283,891]])     #左上，右上，左下，右下
# pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])      #转换
# matrix = cv2.getPerspectiveTransform(pts1,pts2)     #将倾斜的图像拉正，此处样例是地砖
# imgoutput = cv2.warpPerspective(img,matrix,(width,height))
# # img_resize = cv2.resize(img,(341,512))
# cv2.imshow("zsanett",img)
# cv2.imshow("zsanett-w",imgoutput)
# cv2.waitKey(0)


#多图同屏显示
# img = cv2.imread(im_path)
# imgr = cv2.resize(img,(341,512))
# hor = np.hstack((img,img))      #水平放置
# ver = np.vstack((imgr,imgr))      #垂直放置
#
# cv2.imshow("horizontal",hor)
# cv2.imshow("vertical",ver)
# cv2.waitKey(0)


#颜色检测&trackbars(滑窗)将图片从VGR转换到HSV颜色空间
# def empty(a):
#     pass
#
# cv2.namedWindow("trackbars")    #命名一个窗口为trackbar
# cv2.resizeWindow("trackbars",640,240)     #调整窗口trackbar大小
# cv2.createTrackbar("hue_min","trackbars",7,255,empty)      #创建一个trackbar，通过滑动条来改变参数，此行为色度下限，以上数值为运行程序后手工填写的，用来过滤颜色
# cv2.createTrackbar("hue_max","trackbars",18,255,empty)     #此行为色度上限
# cv2.createTrackbar("sat_min","trackbars",49,255,empty)       #此行为饱和度下限
# cv2.createTrackbar("sat_max","trackbars",255,255,empty)     #此行为饱和度上限
# cv2.createTrackbar("val_min","trackbars",102,255,empty)       #此行为值下限
# cv2.createTrackbar("val_max","trackbars",255,255,empty)     #此行为值上限
#
# while True:
#     img = cv2.imread(im_path)
#     imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)    #从BGR到HSV，HSV(色调 Hue,饱和度 Saturation,亮度 Value)，相较于BGR空间，HSV空间的识别的范围更广，更方便
#
#     h_min = cv2.getTrackbarPos("hue_min", "trackbars")      #创建一组滑窗
#     h_max = cv2.getTrackbarPos("hue_max", "trackbars")
#     s_min = cv2.getTrackbarPos("sat_min", "trackbars")
#     s_max = cv2.getTrackbarPos("sat_max", "trackbars")
#     v_min = cv2.getTrackbarPos("val_min", "trackbars")
#     v_max = cv2.getTrackbarPos("val_max", "trackbars")
#
#     lower = np.array([h_min,s_min,v_min])       #创建蒙版下界
#     upper = np.array([h_max,s_max,v_max])     #创建蒙版上界
#     mask = cv2.inRange(imgHSV,lower,upper)      #创建蒙版，原图,下界（低于该值会被置为0）,上界（高于该值会被置为0）
#     imgresult = cv2.bitwise_and(img,img,mask=mask)      #将蒙版叠加在原图上，达到筛选某颜色的目的
#
#     # cv2.imshow("original",img)
#     # cv2.imshow("HSV",imgHSV)
#     cv2.imshow("mask",mask)
#     cv2.imshow("imgresult", imgresult)
#     cv2.waitKey(1)


#形状检测，用于检查图像中的标准图形，诸如正方形、矩形、三角、圆等

# def stackImages(scale,imgArray):        #定义了一个堆叠显示图像的函数
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= np.hstack(imgArray)
#         ver = hor
#     return ver
#
# def getcontours(img):       #定义了一个寻找图像轮廓的函数
#     contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         print(area)
#         if area > 500:
#             cv2.drawContours(imgcontour, cnt, -1, (255, 0, 0), 3)
#             peri = cv2.arcLength(cnt, True)     #图像中各个闭合的标准图形的轮廓长度
#             # print(peri)
#             approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)       #各个轮廓的拐点数
#             print(len(approx))
#             objCor = len(approx)
#             x, y, w, h = cv2.boundingRect(approx)       #用矩形框框选出发现的轮廓
#
#             if objCor == 3:
#                 objectType = "Tri"
#             elif objCor == 4:
#                 aspRatio = w / float(h)
#                 if aspRatio > 0.98 and aspRatio < 1.03:
#                     objectType = "Square"
#                 else:
#                     objectType = "Rectangle"
#             elif objCor > 4:
#                 objectType = "Circles"
#             else:
#                 objectType = "None"
#
#             cv2.rectangle(imgcontour, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(imgcontour, objectType,
#                         (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
#                         (0, 0, 0), 2)
#
# img = cv2.imread(im_path)
# imgcontour = img.copy()
# imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgblur = cv2.GaussianBlur(imggray,(3,3),1)
# imgcanny = cv2.Canny(imgblur,50,50)
# getcontours(imgcanny)
#
# imgblank = np.zeros_like(img)
# imgstack = stackImages(0.5,([img,imggray,imgblur],
#                             [imgcanny,imgcontour,imgblank]))
#
# cv2.imshow("Stack",imgstack)
# cv2.waitKey(0)


#人脸识别
facecascade = cv2.CascadeClassifier(things_dtct_path)       #级联分类器
img = cv2.imread(im_path)
imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = facecascade.detectMultiScale(imggray,1.1,20)     #表示每次图像尺寸减小的比例1.1;每一个目标至少要被检测到20次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow("result",img)
# cv2.imshow("result-G",imggray)
cv2.waitKey(0)


#使用摄像头实现某一色彩的追踪和轮廓检测（我莫得摄像头，不敲了）

