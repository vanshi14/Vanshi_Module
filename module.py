def read_img():
    import cv2 as cv 
    img = cv.imread('Photos/4k.jpg')
    cv.imshow('Img',img)
    cv.waitKey(0)

def read_video():
    import cv2 as cv 
    capture = cv.VideoCapture('Videos/dog.mp4')
    while True:
       isTrue, frame=capture.read()
       cv.imshow('Videos',frame)
       if cv.waitKey(20) & 0Xff==ord('d'):
        break
    capture.release()
    cv.destroyAllWindows()

def resize_img():
   import cv2 as cv 
   img = cv.imread('Photos/4k.jpg')
   cv.imshow('cat',img)
   def rescaleFrame(frame, scale=0.75):
      width = int(frame.shape[0]*scale)
      height = int(frame.shape[1]*scale)
      dimensions =(width,height)
      return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
   resize_img = rescaleFrame(img)
   cv.imshow('Img_resized',resize_img)
   cv.waitKey(0)

def resize_video():
   import cv2 as cv 
   capture = cv.VideoCapture('Videos/dog.mp4')
   def rescaleFrame(frame, scale=0.75):
     width = int(frame.shape[0]*scale)
     height = int(frame.shape[1]*scale)
     dimensions =(width,height)
     return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
   while True:
       isTrue, frame=capture.read()
       frame_resized = rescaleFrame(frame)
       cv.imshow('Video',frame)
       cv.imshow('Resized_Video',frame_resized)
       if cv.waitKey(20)&0XFF==ord('d'):
          break
   capture.release()
   cv.destroyAllWindows()

def blank_img():
   # BLANK IMAGE
    import cv2 as cv
    import numpy as np
    blank=np.zeros((500,500,3),dtype='uint8')
    cv.imshow('Blank',blank)
    cv.waitKey(0)

def paint_img():
   import cv2 as cv
   import numpy as np
   blank=np.zeros((500,500,3),dtype='uint8')
   blank[200:300,300:400]=255,0,0
   cv.imshow('Blue',blank)
   cv.waitKey(0)


def draw_rectangle():
   import cv2 as cv 
   import numpy as np 
   blank=np.zeros((500,500,3),dtype='uint8')
   cv.imshow('Blank',blank)
   blank[:] = 255,255,255
   cv.imshow('Green',blank)
   cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,255,240),thickness=-1)
   cv.imshow('Rectangle',blank)
   cv.waitKey(0)

def draw_circle():
   import cv2 as cv 
   import numpy as np 
   blank=np.zeros((500,500,3),dtype='uint8')
   cv.imshow('Blank',blank)
   blank[200:300,300:400] = 0,155,255
   cv.imshow('Green',blank)
   cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2),40,(255,255,0),thickness=-1)
   cv.imshow('Circle',blank)
   cv.waitKey(0)

def draw_line():
   import cv2 as cv 
   import numpy as np 
   blank=np.zeros((500,500,3),dtype='uint8')
   cv.imshow('Blank',blank)
   blank[200:300,300:400] = 0,0,0
   cv.imshow('Green',blank)
   cv.line(blank,(0,0),(blank.shape[1]//2, blank.shape[0]//2),(0,255,0),thickness=3)
   cv.imshow('Line',blank)
   cv.waitKey(0)

def put_text():
   import cv2 as cv 
   import numpy as np 
   blank=np.zeros((500,500,3),dtype='uint8')
   cv.imshow('Blank',blank)
   blank[:] = 100,0,100
   cv.imshow('Green',blank)
   cv.putText(blank, "Vanshi",(235,255),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,255),3)
   cv.imshow('Text',blank)
   cv.waitKey(0)

def grayscale():
   import cv2 as cv 
   img = cv.imread('Photos/4k.jpg')
   cv.imshow('Img',img)
   gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   cv.imshow('Gray',gray)
   cv.waitKey(0)

def blur():
   import cv2 as cv
   img = cv.imread('Photos/park.jpg')
   blur = cv.GaussianBlur(img,(3,3), cv.BORDER_DEFAULT)
   cv.imshow('blur',blur)
   cv.waitKey(0)

def edge_cascade():
   import cv2 as cv
   img = cv.imread('Photos/park.jpg')
   canny = cv.Canny(img, 125,175)
   cv.imshow('canny',canny)
   cv.waitKey(0)

def dilate():
   import cv2 as cv
   img = cv.imread('Photos/park.jpg')
   canny = cv.Canny(img, 125,175)
   dilated = cv.dilate(canny,(7,7),iterations=2) #iterations=thickness
   cv.imshow('dilated',dilated)
   cv.waitKey(0)

def eroded():
   import cv2 as cv
   img = cv.imread('Photos/4k.jpg')
   canny = cv.Canny(img, 125,175)
   dilated = cv.dilate(canny,(7,7),iterations=2)
   eroded = cv.erode(dilated,(7,7),iterations=2)
   cv.imshow('eroded',eroded)
   cv.waitKey(0)

def resize():
   import cv2 as cv 
   img = cv.imread('Photos/4k.jpg')
   resized = cv.resize(img,(500,500), interpolation=cv.INTER_AREA)
   cv.imshow('resized',resized)
   cv.waitKey(0)

def crop():
   import cv2 as cv 
   img = cv.imread('Photos/4k.jpg')
   cropped = img[50:200, 200:400]
   cv.imshow('Cropped',cropped)
   cv.waitKey(0)

def img_translation():
   import cv2 as cv 
   import numpy as np 
   img = cv.imread('Photos/4k.jpg', 0)
   rows, cols = img.shape
   M = np.float32([[1,0,100],[0,1,50]])
   dst = cv.warpAffine(img,M,(rows,cols))
   cv.imshow('Image',img)
   cv.waitKey(0)

def img_reflectionH():
   import numpy as np
   import cv2 as cv
   img = cv.imread('Photos/park.jpg', 0)
   rows, cols = img.shape
   M = np.float32([[1, 0, 0],[0, -1, rows],[0, 0, 1]])
   reflected_img = cv.warpPerspective(img, M,(int(cols),int(rows)))
   cv.imshow('img', reflected_img)
   cv.imwrite('reflection_out.jpg', reflected_img)
   cv.waitKey(0)
   cv.destroyAllWindows()

def img_reflectionV():
   import numpy as np
   import cv2 as cv
   img = cv.imread('Photos/park.jpg', 0)
   rows, cols = img.shape
   M = np.float32([[-1, 0, cols], [0, 1, 0], [0, 0, 1]])
   reflected_img = cv.warpPerspective(img, M,(int(cols),int(rows)))
   cv.imshow('img', reflected_img)
   cv.imwrite('reflection_out.jpg', reflected_img)
   cv.waitKey(0)
   cv.destroyAllWindows()

def img_rotation():
   import numpy as np
   import cv2 as cv
   img = cv.imread('Photos/park.jpg', 0)
   rows, cols = img.shape
   M = np.float32([[1, 0, 0], [0, -1, rows], [0, 0, 1]])
   img_rotation = cv.warpAffine(img,cv.getRotationMatrix2D((cols/2, rows/2),50, 0.6),(cols, rows))
   cv.imshow('img', img_rotation)
   cv.imwrite('rotation_out.jpg', img_rotation)
   cv.waitKey(0)
   cv.destroyAllWindows()

def scaling_shrinked():
   import numpy as np
   import cv2 as cv
   img = cv.imread('Photos/cats.jpg', 0)
   rows, cols = img.shape
   img_shrinked=cv.resize(img, (250, 200),interpolation=cv.INTER_AREA)
   cv.imshow('img', img_shrinked)
   cv.waitKey(0)
   cv.destroyAllWindows()
     
def scaling_enlarged():
   import numpy as np
   import cv2 as cv
   img = cv.imread('Photos/cats.jpg', 0)
   rows, cols = img.shape
   img_shrinked = cv.resize(img, (250, 200),interpolation=cv.INTER_AREA)
   img_enlarged = cv.resize(img_shrinked, None,fx=1.5, fy=1.5,interpolation=cv.INTER_CUBIC)
   cv.imshow('Img_enlarged',img_enlarged)
   cv.waitKey(0)
   cv.destroyAllWindows()

def img_croping():
    import cv2 as cv
    img=cv.imread('Photos/park.jpg')
    cv.imshow('Cat',img)
    cropped=img[50:200,200:400]
    cv.imshow('Cropped',cropped)
    cv.waitKey(0)

def x_axis():
   import numpy as np
   import cv2 as cv
   img = cv.imread('Photos/park.jpg', 0)
   rows, cols = img.shape
   M = np.float32([[1, 0.5, 0], [0, 1, 0],[0, 0, 1]])
   sheared_img =cv.warpPerspective(img, M,(int(cols*1.5), int(rows*1.5)))
   cv.imshow('img', sheared_img)
   cv.waitKey(0)
   cv.destroyAllWindows()

def y_axis():
   import numpy as np
   import cv2 as cv
   img = cv.imread('Photos/park.jpg', 0)
   rows, cols = img.shape
   M = np.float32([[1, 0, 0], [0.5, 1, 0],[0, 0, 1]])
   sheared_img =cv.warpPerspective(img, M,(int(cols*1.5), int(rows*1.5)))
   cv.imshow('sheared_y-axis_out.jpg', sheared_img)
   cv.waitKey(0)
   cv.destroyAllWindows()

def contours():
   import cv2 as cv 
   import numpy as np 
   image = cv.imread('Photos/4k.jpg')
   cv.waitKey(0)
   gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
   edged = cv.Canny(gray, 30, 200)
   cv.waitKey(0)
   contours, hierarchy = cv.findContours(edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
   cv.imshow('Canny Edges After Contouring', edged)
   cv.waitKey(0)
   print("Number of Contours found = " + str(len(contours)))
   cv.drawContours(image, contours, -1, (0, 0, 0), 3)
   cv.imshow('Contours', image)
   cv.waitKey(0) 
   cv.destroyAllWindows()

def color_spaces():
   import cv2
   image = cv2.imread('Photos/4k.jpg')
   B, G, R = cv2.split(image)
   cv2.imshow("original", image)
   cv2.waitKey(0)
   cv2.imshow("blue", B)
   cv2.waitKey(0)
   cv2.imshow("Green", G)
   cv2.waitKey(0)
   cv2.imshow("red", R)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

def blur2D():
    import cv2 as cv
    import numpy as np 
    # Reading the image
    image = cv.imread('Photos/4k.jpg')
    # Creating the kernel with numpy
    kernel2 = np.ones((5, 5), np.float32)/25
    # Applying the filter
    img = cv.filter2D(src=image, ddepth=-1, kernel=kernel2)
    # showing the image
    cv.imshow('Original', image)
    cv.imshow('Kernel Blur', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def blur_average():
    import cv2 as cv
    image = cv.imread('Photos/park.jpg')
    # Applying the filter
    averageBlur = cv.blur(image, (5, 5))
    # Showing the image
    cv.imshow('Original', image)
    cv.imshow('Average blur', averageBlur)
    cv.waitKey()
    cv.destroyAllWindows() 

def blur_gaussian():
   import cv2 as cv
   image = cv.imread('Photos/park.jpg')
   gaussian = cv.GaussianBlur(image, (3, 3), 0)
   # Showing the image
   cv.imshow('Original', image)
   cv.imshow('Gaussian blur', gaussian)
   cv.waitKey() 
   cv.destroyAllWindows()

def blur_median():
   import cv2 as cv 
   image = cv.imread('Photos/4k.jpg')
   # Applying the filter
   medianBlur = cv.medianBlur(image, 9)
   # Showing the image
   cv.imshow('Original', image)
   cv.imshow('Median blur',medianBlur)
   cv.waitKey()
   cv.destroyAllWindows()

def blur_bilateral():
   import cv2 as cv 
   image = cv.imread('Photos/4k.jpg')
   # Applying the filter
   bilateral = cv.bilateralFilter(image,9, 75, 75)
   # Showing the image
   cv.imshow('Original', image)
   cv.imshow('Bilateral blur', bilateral)
   cv.waitKey()
   cv.destroyAllWindows()

def bitwise_and():
   # import the necessary packages
   import numpy as np
   import cv2
   # draw a rectangle
   rectangle = np.zeros((300, 300), dtype="uint8")
   cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
   cv2.imshow("Rectangle", rectangle)
   # draw a circle
   circle = np.zeros((300, 300), dtype = "uint8")
   cv2.circle(circle, (150, 150), 150, 255, -1)
   cv2.imshow("Circle", circle)
   bitwiseAnd = cv2.bitwise_and(rectangle, circle)
   cv2.imshow("AND", bitwiseAnd)
   cv2.waitKey(0)

def bitwise_or():
   # import the necessary packages
   import numpy as np
   import cv2
   # draw a rectangle
   rectangle = np.zeros((300, 300), dtype="uint8")
   cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
   cv2.imshow("Rectangle", rectangle)
   # draw a circle
   circle = np.zeros((300, 300), dtype = "uint8")
   cv2.circle(circle, (150, 150), 150, 255, -1)
   cv2.imshow("Circle", circle)
   bitwiseOr = cv2.bitwise_or(rectangle, circle)
   cv2.imshow("OR", bitwiseOr)
   cv2.waitKey(0)

def bitwise_xor():
   # import the necessary packages
   import numpy as np
   import cv2
   # draw a rectangle
   rectangle = np.zeros((300, 300), dtype="uint8")
   cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
   cv2.imshow("Rectangle", rectangle)
   # draw a circle
   circle = np.zeros((300, 300), dtype = "uint8")
   cv2.circle(circle, (150, 150), 150, 255, -1)
   cv2.imshow("Circle", circle)
   bitwiseXor = cv2.bitwise_xor(rectangle, circle)
   cv2.imshow("XOR", bitwiseXor)
   cv2.waitKey(0)

def bitwise_nor():
   # import the necessary packages
   import numpy as np
   import cv2
   # draw a rectangle
   rectangle = np.zeros((300, 300), dtype="uint8")
   cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
   cv2.imshow("Rectangle", rectangle)
   # draw a circle
   circle = np.zeros((300, 300), dtype = "uint8")
   cv2.circle(circle, (150, 150), 150, 255, -1)
   cv2.imshow("Circle", circle)
   bitwiseNot = cv2.bitwise_not(circle)
   cv2.imshow("NOT", bitwiseNot)
   cv2.waitKey(0)

def masking():
   import cv2 as cv 
   import numpy as np 
   img = cv.imread('Photos/cats.jpg')
   cv.imshow('Img',img)
   blank = np.zeros(img.shape[:2],dtype='uint8')
   cv.imshow('Blank',blank)
   circle = cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),200,255, -1)
   cv.imshow('Mask',circle)
   masked = cv.bitwise_and(img,img,mask=circle)
   cv.imshow('Masked Image', masked)
   cv.waitKey(0)
   cv.destroyAllWindows()

def alpha_blend():
   import cv2
   img1 = cv2.imread('Photos/cat.jpg')
   img2 = cv2.imread('Photos/park.jpg')
   img2 = cv2.resize(img2, img1.shape[1::-1])
   cv2.imshow("img 1",img1)
   cv2.waitKey(0)
   cv2.imshow("img 2",img2)
   cv2.waitKey(0)
   choice = 1
   while (choice) :
      alpha = float(input("Enter alpha value"))
      dst = cv2.addWeighted(img1, alpha , img2, 1-alpha, 0)
      cv2.imwrite('alpha_mask_.jpg', dst)
      img3 = cv2.imread('alpha_mask_.jpg')
      cv2.imshow("alpha blending 1",img3)
      cv2.waitKey(0)
      choice = int(input("Enter 1 to continue and 0 to exit"))
   
def histogram():
   # importing required libraries of opencv
   import cv2 as cv
   import matplotlib.pyplot as plt
   img = cv.imread('Photos/cats.jpg')
   cv.imshow('Cats',img)
   gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
   cv.imshow('Gray',gray)
   gray_hist = cv.calcHist([gray],[0],None,[256],[0,256])
   plt.figure()
   plt.title('Grayscale Histogram')
   plt.xlabel('Bins')
   plt.ylabel('# of pixels')
   plt.plot(gray_hist)
   plt.xlim([0,256])
   plt.show()
   cv.waitKey(0)







 
 


   
   




 
   





 



   

   









