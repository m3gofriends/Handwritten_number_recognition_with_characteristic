"""                                      
      First publication date : 2021 / 1 / 14
      Author : 張哲銘(CHANG, CHE-MING)
"""
from tkinter import * 
import cv2 as cv
import numpy as np
from PIL import ImageGrab
from skimage import morphology

"""   Tkinter環境設定   """

def activate_paint(position):
    global lastx, lasty
    canvas.bind('<B1-Motion>', paint)
    lastx, lasty = position.x, position.y

def paint(position):
    global lastx, lasty
    canvas.create_line((lastx, lasty, position.x, position.y), fill = 'white', width = 1.0, capstyle = ROUND, smooth = TRUE, splinesteps = 36)
    lastx, lasty = position.x, position.y

def Erase(my_text):
    canvas.delete("all")
    cv.waitKey(1)
    cv.destroyAllWindows()
    my_text.set("Answer : ")

"""   Tkinter環境設定   """


def FindCornerPoint(img, StartPoint, EndPoint, total_pixels, LR_CornerPoint, TB_CornerPoint):

    """   輸入參數分別為(圖片, 數字的起點, 數字的終點, 數字本身的pixel總數, 儲存左右兩側Corner的list, 儲存上下兩側Corner的list)
          先複製一張原圖(因為要對圖片進行修改)
          創造一個大小為4的bool陣列(紀錄pixel的移動路徑)
          創造兩個陣列分別記錄左右兩側和上下兩側的移動方向
          創造兩個整數變數分別記錄左右兩側和上下兩側從EndPoint到CornerPoint或CornerPoint到另一個CornerPoint的移動步數
          並兩個陣列將步數儲存下來(無法從CornerPoint的方向去判斷時才會使用它)
          

          先將StartPoint的值給NowPoint
          
          接著把自己的位置變成黑色, 然後去搜尋以自己為中心的3x3陣列不為黑色pixel的位置(因為EndPoint以自己為中心的3x3陣列只會有2個值不為黑色)
          如果此pixel的位置為rows = 0且cols = 0, 表示正在往左上方移動, 為rows = 0且cols = 2, 表示正在往右上方移動, 為rows = 2且cols = 0, 表示正在往左下方移動, 為rows = 2且cols = 2, 表示正在往右下方移動
          接著bool陣列會變True(0為左上, 1為右上, 2為左下, 3為右下)
          如果此bool陣列有兩個值為True即可判斷此Corner的方向(例如是由[True, False, False, False]變成[True, False, True, False], 表示pixel是先經過左上再往左下, 因此可以得知這是一個Top方向的CornerPoint)
          做完以後將自己的位置設為True其他為False, 表示此pixel下一個Corner是由此方向開始的
          如果步數 比 數字本身的pixel總數 // 15 還要小的話就捨棄(為了要去除太過接近的CornerPoint)
          然後把合格的步數放進自己方向的步數陣列裡面
          並將步數變數初始化為0, 記錄著下一個Corner的步數

          當NowPoint等於EndPoint時表示整個數字都被搜尋過了, 即可結束   """
    
    copyimg = img.copy()
    bool_direction = np.zeros(4, dtype = bool)
    LR_direction = []
    LR_StepCounter = 0
    LR_StepCounter_list = []
    TB_direction = []
    TB_StepCounter = 0
    TB_StepCounter_list = []
    NowPoint = StartPoint
    
    while NowPoint != EndPoint:
        TB_StepCounter = TB_StepCounter + 1
        LR_StepCounter = LR_StepCounter + 1
        copyimg[NowPoint] = 0
        rows = np.nonzero(copyimg[NowPoint[0] - 1:NowPoint[0] + 2, NowPoint[1] - 1:NowPoint[1] + 2])[0][0]
        cols = np.nonzero(copyimg[NowPoint[0] - 1:NowPoint[0] + 2, NowPoint[1] - 1:NowPoint[1] + 2])[1][0]
        #UpperLeft
        if rows== 0 and cols==0:
            bool_direction[0] = True
            if len(np.nonzero(bool_direction)[0]) == 2:
                if bool_direction[1] == True and LR_StepCounter > total_pixels // 15:
                    LR_CornerPoint.append((NowPoint[0] + int(rows) - 1, NowPoint[1] + int(cols) - 1))
                    LR_direction.append("RIGHT")
                    LR_StepCounter_list.append(LR_StepCounter)
                    LR_StepCounter = 0                    
                elif bool_direction[2] == True and TB_StepCounter > total_pixels // 15:
                    TB_CornerPoint.append((NowPoint[0] + int(rows) - 1, NowPoint[1] + int(cols) - 1))
                    TB_direction.append("BOTTOM")
                    TB_StepCounter_list.append(TB_StepCounter)
                    TB_StepCounter = 0  
                bool_direction = [True, False, False, False]
        #UpperRight 
        elif rows== 0 and cols==2:
            bool_direction[1] = True
            if len(np.nonzero(bool_direction)[0]) == 2:
                if bool_direction[0] == True and LR_StepCounter > total_pixels // 15:
                    LR_CornerPoint.append((NowPoint[0] + int(rows) - 1, NowPoint[1] + int(cols) - 1))
                    LR_direction.append("LEFT")
                    LR_StepCounter_list.append(LR_StepCounter)
                    LR_StepCounter = 0
                elif bool_direction[3] == True and TB_StepCounter > total_pixels // 15:
                    TB_CornerPoint.append((NowPoint[0] + int(rows) - 1, NowPoint[1] + int(cols) - 1))
                    TB_direction.append("BOTTOM")
                    TB_StepCounter_list.append(TB_StepCounter)
                    TB_StepCounter = 0
                bool_direction = [False, True, False, False]
        #BottomLeft 
        elif rows== 2 and cols==0:
            bool_direction[2] = True
            if len(np.nonzero(bool_direction)[0]) == 2:
                if bool_direction[3] == True and LR_StepCounter > total_pixels // 15:
                    LR_CornerPoint.append((NowPoint[0] + int(rows) - 1, NowPoint[1] + int(cols) - 1))
                    LR_direction.append("RIGHT")
                    LR_StepCounter_list.append(LR_StepCounter)
                    LR_StepCounter = 0
                elif bool_direction[0] == True and TB_StepCounter > total_pixels // 15:
                    TB_CornerPoint.append((NowPoint[0] + int(rows) - 1, NowPoint[1] + int(cols) - 1))
                    TB_direction.append("TOP")
                    TB_StepCounter_list.append(TB_StepCounter)
                    TB_StepCounter = 0
                bool_direction = [False, False, True, False]
        #BottomRight
        elif rows== 2 and cols==2:
            bool_direction[3] = True
            if len(np.nonzero(bool_direction)[0]) == 2:
                if bool_direction[2] == True and LR_StepCounter > total_pixels // 15:
                    LR_CornerPoint.append((NowPoint[0] + int(rows) - 1, NowPoint[1] + int(cols) - 1))
                    LR_direction.append("LEFT")
                    LR_StepCounter_list.append(LR_StepCounter)
                    LR_StepCounter = 0
                elif bool_direction[1] == True and TB_StepCounter > total_pixels // 15:
                    TB_CornerPoint.append((NowPoint[0] + int(rows) - 1, NowPoint[1] + int(cols) - 1))
                    TB_direction.append("TOP")
                    TB_StepCounter_list.append(TB_StepCounter)
                    TB_StepCounter = 0
                bool_direction = [False, False, False, True]
                
        NowPoint = (NowPoint[0] + int(rows) - 1, NowPoint[1] + int(cols) - 1)
    
    LR_StepCounter_list.append(LR_StepCounter)
    TB_StepCounter_list.append(TB_StepCounter)
    return LR_direction, TB_direction, LR_StepCounter_list, TB_StepCounter_list

def number_of_hole(img, hole_img, hole_counter):
    
    """   判斷hole的數量去執行相對應的函式
          0個hole執行zero_of_hole
          1個hole執行one_of_hole
          2個hole執行my_text.set("Answer : 8")
          大於2個hole則執行my_text.set("Error : holes number = " + str(hole_counter) + "( > 2 )"))   """
          
    switcher = {
        0:zero_of_hole, 
        1:one_of_hole, 
        2:lambda x1, x2:my_text.set("Answer : 8") #參數x1, x2從未使用, 為了return function
        }
    func = switcher.get(hole_counter, lambda x1, x2:my_text.set("Error : holes number = " + str(hole_counter) + "( > 2 )")) #參數x1, x2從未使用, 為了return function
    return func(img, hole_img)
    
def zero_of_hole(img, hole_img = None):
    h, w = img.shape[:2]
    EndPoint = []
    HarrisCornerPoint = []
    LR_CornerPoint = []
    TB_CornerPoint = []


    """   搜尋整張圖, 找到白點(即數字上的點)
          尋找以自己為中心的3x3陣列上白色的點個數
          個數為2的話表示此點有可能是EndPoint
          不為2則把此點變成黑色, 然後去判斷圖片上的連通圖個數
          連通圖個數為1表示這點是多餘的
          否則把此點變回白色   """

    for i in range(h):
        for j in range(w):
            if img[i, j] != 0:
                if len(np.nonzero(img[i - 1:i + 2, j - 1:j + 2])[0]) == 2:
                    copyimg = img.copy()
                    cv.circle(copyimg, (j, i), w // 4, (0, 0, 0), - 1) #用跟背景一樣顏色(黑色)的實心circle，覆蓋在該點上，為了要檢測此點是否為Endpoint
                    if cv.connectedComponents(copyimg)[0] - 1 == 1: #判斷當此點被黑色circle覆蓋時會不會產生新的Connected Component，如果不會則表示此點為Endpoint
                        EndPoint.append((i, j))
                    else:
                        while len(np.nonzero(img[i - 1:i + 2, j - 1:j + 2])[0]) < 3:
                            img[i, j] = 0
                            i = i + int(np.nonzero(img[i - 1:i + 2, j - 1:j + 2])[0][0]) - 1
                            j = j + int(np.nonzero(img[i - 1:i + 2, j - 1:j + 2])[1][0]) - 1
                        cv.imshow("Cleared img", img)
                else:
                    img[i, j] = 0
                    if cv.connectedComponents(img)[0] - 1 != 1:
                        img[i, j] = 255
    
    total_pixels = len(np.nonzero(img)[0])
    BGR_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)


    """   找HarrisCornerPoint   """

    dst = cv.cornerHarris(img, 15, 5, 0.05)
    max_dst = cv.dilate(dst, np.ones((10, 10)))   
    dst = dst * (dst == max_dst)
    sortIdx = np.argsort(dst.flatten())[:: -1][:5]


    """   將HarrisCornerPoint做過濾
          如果太靠近Endpoint或者HarrisCornerPoint彼此之間太過接近
          則此點不採用   """

    for k in sortIdx:
        TempPoint_count = 0
        EndPoint_count = 0
        if len(HarrisCornerPoint) == 0:
            for i in range(len(EndPoint)):
                if max(abs(int(k % w) - EndPoint[i][1]), abs(int(k / w) - EndPoint[i][0])) > w / 3:
                    EndPoint_count = EndPoint_count + 1
                    if EndPoint_count == len(EndPoint):
                        HarrisCornerPoint.append((int(k % w), int(k / w)))
        else:
            for i in range(len(HarrisCornerPoint)):
                if max(abs(int(k % w) - HarrisCornerPoint[i][0]), abs(int(k / w) - HarrisCornerPoint[i][1])) > w / 3:
                    TempPoint_count = TempPoint_count + 1
            for i in range(len(EndPoint)):
                if max(abs(int(k % w) - EndPoint[i][1]), abs(int(k / w) - EndPoint[i][0])) > w / 3:
                    EndPoint_count = EndPoint_count + 1
            if TempPoint_count == len(HarrisCornerPoint) and EndPoint_count == len(EndPoint):
                HarrisCornerPoint.append((int(k % w), int(k / w)))

    LR_direction, TB_direction, LR_StepCounter_list, TB_StepCounter_list = FindCornerPoint(img, EndPoint[0], EndPoint[1], total_pixels, LR_CornerPoint, TB_CornerPoint) #找左右側和上下側的CornerPoint

    for (i, j) in HarrisCornerPoint:
        cv.circle(BGR_img, (i, j), 5, (0, 255, 255), -1) #畫出HarrisCornerPoint

    for (i, j) in EndPoint:
        cv.circle(BGR_img, (j, i), 3, (0, 255, 0), -1) #畫出EndPoint

    for (i, j) in LR_CornerPoint:
        cv.circle(BGR_img, (j, i), 3, (255, 0, 0), -1) #畫出LR_CornerPoint
        
    for (i, j) in TB_CornerPoint:
        cv.circle(BGR_img, (j, i), 3, (0, 0, 255), -1) #畫出TB_CornerPoint

    print("HarrisCornerPoint: %d, LR_CornerPoint: %d, TB_CornerPoint: %d" % (len(HarrisCornerPoint), len(LR_CornerPoint), len(TB_CornerPoint)))
    cv.imshow("Find EndPoint & CornerPoint", BGR_img)


    """   用三種CornerPoint去判斷是什麼數字   """

    if len(EndPoint) != 2:
        my_text.set("Error : Number not found (0 hole)")
        root.mainloop()

    elif len(HarrisCornerPoint) == 0 and len(TB_CornerPoint) == 0:
        my_text.set("Answer : 1")
        root.mainloop()

    elif len(HarrisCornerPoint) == 1:
        copyimg = img.copy()
        cv.line(copyimg, (EndPoint[0][1], EndPoint[0][0]), (EndPoint[1][1], EndPoint[1][0]), (255, 255, 255), 1)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv.floodFill(copyimg, mask, (0, 0), 255)
        hole_img = cv.bitwise_not(copyimg)
        if cv.connectedComponents(hole_img)[0] - 1 == 1:
            my_text.set("Answer : 7")
            cv.imshow("Find 7", hole_img)
            root.mainloop()
    
    if len(TB_CornerPoint) < 2 or len(TB_CornerPoint) > 3:        
        if LR_direction == ['RIGHT', 'LEFT']:
            my_text.set("Answer : 2")
            root.mainloop()

        elif LR_direction == ['LEFT', 'RIGHT']:
            my_text.set("Answer : 5")
            root.mainloop()

        else:
            my_text.set("Answer : 3")
            root.mainloop()

    elif len(LR_CornerPoint) < 2 or len(LR_CornerPoint) > 3:
        if TB_direction == ['BOTTOM', 'TOP'] and w - EndPoint[0][1] < EndPoint[0][1] - 0:
            my_text.set("Answer : 2")
            root.mainloop()

        elif TB_direction == ['BOTTOM', 'TOP'] and w - EndPoint[0][1] > EndPoint[0][1] - 0:
            my_text.set("Answer : 5")
            root.mainloop()

        else:
            my_text.set("Answer : 3")
            root.mainloop()

    elif min(LR_StepCounter_list) > min(TB_StepCounter_list):
        if LR_direction == ['RIGHT', 'LEFT']:
            my_text.set("Answer : 2")
            root.mainloop()

        elif LR_direction == ['LEFT', 'RIGHT']:
            my_text.set("Answer : 5")
            root.mainloop()

        else:
            my_text.set("Answer : 3")
            root.mainloop()

    else:
        if TB_direction == ['BOTTOM', 'TOP'] and w - EndPoint[0][1] < EndPoint[0][1] - 0:
            my_text.set("Answer : 2")
            root.mainloop()

        elif TB_direction == ['BOTTOM', 'TOP'] and w - EndPoint[0][1] > EndPoint[0][1] - 0:
            my_text.set("Answer : 5")
            root.mainloop()

        else:
            my_text.set("Answer : 3")
            root.mainloop()
            
    my_text.set("Error : Number not found (0 hole)") #什麼都沒找到會輸出這個

def one_of_hole(img, hole_img):
    line_count = 0
    left_line = 0
    right_line = 0
    top_hole = 0
    bottom_hole = 0

    """   把hole image膨脹然後用矩形框起來填滿黑色
          就可以找到其他的line
          然後用hole跟line的關係去判斷數字為何   """
    
    hole_img = cv.dilate(hole_img, np.ones((3, 3), np.uint8), iterations = 9)
    contours, _ = cv.findContours(hole_img, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
    rect = cv.minAreaRect(contours[0])
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img, [box], -1, (0, 0, 0), -1)
    center_point = rect[0]
    diameter_divided_by_3 = max(rect[1][0], rect[1][1]) / 3 #找hole直徑再除以3
         
    x, y, w, h = cv.boundingRect(contours[0])
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

    line_counter, _ = cv.connectedComponents(img)
    line_contours, _ = cv.findContours(img, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
    
    if (center_point[1] - (img.shape[0] / 2)) < 0: #判斷hole的中心點在圖片中的相對位置
        top_hole = top_hole + 1
    else:
        bottom_hole = bottom_hole + 1
    for cnt in line_contours:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(img, [box], -1, (255, 255, 255), 2)
        if diameter_divided_by_3 < max(rect[1][0], rect[1][1]): #直徑除3跟直線長度比較，直線比半徑長的話該線才取，否則捨棄
            line_count = line_count + 1
            if top_hole == 1:
                if abs(box[2][0] - x) < abs(box[2][0] - (x + w)):
                    left_line = left_line + 1
                else:
                    right_line = right_line + 1
            if bottom_hole == 1: 
                if abs(box[0][0] - x) < abs(box[0][0] - (x + w)):
                    left_line = left_line + 1
                else:
                    right_line = right_line + 1
      
    print("Number of top hole : " + str(top_hole))
    print("Number of bottom hole : " + str(bottom_hole))
    print("Number of line : " + str(line_count))
    print("Number of left line : " + str(left_line))
    print("Number of right line : " + str(right_line))
          
    if line_count == 0:
        my_text.set("Answer : 0")
    elif line_count == 2:
        my_text.set("Answer : 4")
    elif line_count == 1 and left_line == 1 and bottom_hole == 1:
        my_text.set("Answer : 6")
    elif line_count == 1 and right_line == 1 and top_hole == 1:
        my_text.set("Answer : 9")
    else:
        my_text.set("Error : Number not found (1 hole)")
        
    cv.imshow("Find line", img)
    cv.imshow("Dilated hole", hole_img)

def classify(img):

    """   把圖片填滿白色用來判斷有幾個hole
          再依據hole的數量去做分類   """
    
    h, w = img.shape[:2]
    copyimg = img.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(copyimg, mask, (0, 0), 255)
    hole_img = cv.bitwise_not(copyimg)
    hole_counter, _ = cv.connectedComponents(hole_img)
    small_hole = 0
    contours, _ = cv.findContours(hole_img, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) < 40:
            img = cv.add(img, hole_img)
            small_hole = small_hole + 1        
    print("-----------------")
    print("Number of hole: " + str(hole_counter - 1 - small_hole)) 
    number_of_hole(img, hole_img, hole_counter - 1 - small_hole)
    cv.imshow("Find hole", hole_img)

def rotate_and_crop(img, rect):

    """   先做膨脹再旋轉圖片(因為旋轉圖片會變形圖片且顏色會些微改變)
          用二值化恢復顏色
          再來做骨架化
          最後裁切圖片於適當大小   """
    
    h, w = img.shape[:2]
    img = cv.dilate(img, np.ones((3, 3), np.uint8), iterations = 1)
    if rect[1][0] < rect[1][1]:
        rotated_img = cv.warpAffine(img, cv.getRotationMatrix2D(rect[0], rect[2], 1.0), (w, h))
    else:
        rotated_img = cv.warpAffine(img, cv.getRotationMatrix2D(rect[0], rect[2] - 270, 1.0), (w, h))
    _, rotated_img = cv.threshold(rotated_img, 0, 255, cv.THRESH_BINARY) #旋轉後顏色會些微改變，這邊採用二值化恢復，不然數字會有虛線造成hole數量辨識錯誤
    rotated_img[rotated_img==255] = 1
    skeleton_img = morphology.skeletonize(rotated_img, method = 'lee')
    rotated_img = skeleton_img.astype(np.uint8) * 255
    contours, _ = cv.findContours(rotated_img, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv.boundingRect(max(contours, key = cv.contourArea))
    cropped_img = np.zeros((h + 20, w + 20), np.uint8)
    cropped_img[10: -10, 10: -10] = rotated_img[y:y + h, x:x + w]
    return cropped_img
    
def Execute():

    """   找最大連通圖然後把圖片二值化
          再來經過旋轉和裁切後進行分類動作   """
    
    ImageGrab.grab().crop((canvas.winfo_rootx() + 2, canvas.winfo_rooty() + 2, canvas.winfo_rootx() + 640, canvas.winfo_rooty() + 480)).save("my_drawing.png", quality = 100) #裁切canvas的畫面
    img = np.array(cv.imread("my_drawing.png"), dtype = np.uint8)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    LCC_img = np.zeros_like(gray_img)   #LCC = largest connected component                                   
    for val in np.unique(gray_img)[1:]:                                      
        mask = np.uint8(gray_img == val)                                     
        labels, stats = cv.connectedComponentsWithStats(mask, 8)[1:3] 
        largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])      
        LCC_img[labels == largest_label] = val                         

    _, thresh = cv.threshold(LCC_img, 0, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)

    rect = cv.minAreaRect(contours[0])
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img, [box], -1, (0, 255, 0), 1)
    rotated_img = rotate_and_crop(thresh, rect)
    classify(rotated_img)

    cv.imshow("Largest connected component", LCC_img)
    cv.imshow("Rotate and crop", rotate_and_crop(thresh, rect))
    cv.imshow("Origin image", img)


"""   Tkinter環境設定   """
     
root = Tk()
root.title("Handwritten number recognition with characteristic")
lastx, lasty = None, None
my_text = StringVar()
my_text.set("Answer : ")
canvas = Canvas(root, width = 640, height = 480, bg = 'black')
canvas.bind('<1>', activate_paint)
canvas.pack(expand = YES, fill = BOTH)
button_frame = Frame(root)
button_frame.pack()
Button(button_frame, text = "Erase", command = lambda:Erase(my_text)).pack(side = LEFT)
Button(button_frame, text = "Execute", command = Execute).pack(side = LEFT)
Label(root, textvariable = my_text).pack()
root.mainloop()

"""   Tkinter環境設定   """
