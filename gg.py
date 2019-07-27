import cv2
from PIL import ImageGrab,Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from LLModel import Models
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import pyautogui
from pynput.keyboard import Key,Listener
pyautogui.FAILSAFE = True
class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y

SEGIMGROW_H = 35  # //分割出来的小块的长度
SEGIMGCOL_W = 31  # //分割出来的小块的宽度
COL_NUM = 19
ROW_NUM = 11
CONEROFPIC_LT_X = 570  # // 桌面左上角x坐标，依据显示器分辨率调整。
CONEROFPIC_LT_Y = 400  # //桌面左上角y坐标，依据显示器分辨率调整。
CONEROFPIC_RB_X = 1165  # //桌面右下角x坐标，依据显示器分辨率调整。
CONEROFPIC_RB_Y = 785  # //桌面右下角y坐标，依据显示器分辨率调整。
data_dir = "E:/[10]_Dataset/llexp/"

data_transform = transforms.Compose([transforms.Resize([36,36]), #括号出错了
                                        transforms.RandomCrop(36, padding=2),
                                        transforms.ToTensor()
                                        ])
image_datasets = torchvision.datasets.ImageFolder(root = data_dir,
                                                    transform = data_transform
                                                    )

batch_size = 108
dataloader = torch.utils.data.DataLoader(dataset = image_datasets,batch_size = batch_size,shuffle = True)
BKGRND_ID = image_datasets.class_to_idx["background"]


def getImageAt(rowy,colx,img):
    if (rowy>ROW_NUM) | (rowy <0):
        return -1
    if (colx > COL_NUM) | (colx <0):
        return -1
    image = img[rowy*SEGIMGROW_H : (rowy+1)*SEGIMGROW_H, colx*SEGIMGCOL_W : (colx+1)*SEGIMGCOL_W]
    image = cv2.resize(image,(36,36))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def imageCv2Torch(images):
    images = np.array(images)
    images = images.transpose(0,3,1,2)
    images = images.astype(np.float32)
    return torch.from_numpy(images)

def getMatrix():
    im = ImageGrab.grab()
    # im = Image.open("images/3.png","r")
    # plt.imshow(im)
    # plt.show()
    img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    # cv2.imshow("1",img)
    # cv2.waitKey(0)
    # target_img = cv2.crea
    target_img = img[CONEROFPIC_LT_Y:CONEROFPIC_RB_Y,CONEROFPIC_LT_X : CONEROFPIC_RB_X]
    # cv2.imshow("0", target_img)
    # cv2.imshow("2", getImageAt(10,18,target_img))
    # cv2.waitKey(0)
    images = []
    for colx in range(COL_NUM):
        for rowy in range(ROW_NUM):
            img = getImageAt(rowy,colx,target_img)
            images.append(img/255.0)
    images = imageCv2Torch(images)

    img = torchvision.utils.make_grid(images)
    img = img.numpy().transpose(1, 2, 0)

    model = Models().cuda()
    model.load_state_dict(torch.load("E:\[3]_Master\[1]_Pytorch\LLlook\ll5.pth"))
    images = Variable(images).cuda()
    y_pred = model(images)
    _, y_pred_class = torch.max(y_pred, 1)
    y_pred_class = y_pred_class.cpu()
    mat_t = np.zeros([11*19,1])
    for i, it in enumerate(y_pred_class, 0):
        mat_t[i] = int(it.data.numpy())
    mat_t = mat_t.astype(np.int32)
    mat_t =mat_t.reshape([19,11]).T
    # mat_pad = np.zeros([13, 21]).astype(np.int32) + BKGRND_ID
    # for x in range(11):
    #     for y in range(19):
    #         mat_pad[x + 1][y + 1] = mat_t[x][y]
    # # print(mat_pad)
    print(mat_t)
    return mat_t
    # str_=""
    # for i,it in enumerate(y_pred_class,0):
    #     print(str(image_datasets.classes[it.data.numpy()]))
    #
    # plt.imshow(img)
    # plt.show()
    # cv2.waitKey(0)
#######################
######################
####################
##################
#Point 形式是tuple 且对应为(x,y)的坐标值，opencv坐标
def detectVertical(Point_s,Point_e,mat_pad):
    if (Point_e.x != Point_s.x):
        return False
    heigherPoint = Point(Point_s.x,Point_s.y)
    lowerPoint = Point(Point_e.x,Point_e.y)
    if Point_e.y > Point_s.y:
        heigherPoint = Point(Point_e.x,Point_e.y)
        lowerPoint = Point(Point_s.x,Point_s.y)
    heigherPoint.y -= 1
    while(heigherPoint.y>=0):
        if(lowerPoint.y==heigherPoint.y):
            return True
        elif mat_pad[heigherPoint.y][heigherPoint.x] != BKGRND_ID:
            break
        else:
            heigherPoint.y -=1
    return False

def detectHorizental(Point_s,Point_e,mat_pad):
    if (Point_e.y != Point_s.y):
        return False
    rightPoint = Point(Point_s.x,Point_s.y)
    leftPoint = Point(Point_e.x,Point_e.y)
    if Point_e.x > Point_s.x:
        leftPoint  = Point(Point_s.x,Point_s.y)
        rightPoint = Point(Point_e.x,Point_e.y)
    rightPoint.x -= 1
    while (rightPoint.x >= 0):
        if rightPoint.x == leftPoint.x:
            return True
        elif mat_pad[rightPoint.y][rightPoint.x] != BKGRND_ID:
            break
        else:
            rightPoint.x -= 1
    return False

def detectOneNode(Point_s,Point_e,mat_pad):
    flag1 = False
    flag2 = False
    if mat_pad[Point_e.y][Point_s.x] == BKGRND_ID:
        flag1 = True
        bridge_c1 = Point(Point_s.x,Point_e.y)
    if mat_pad[Point_s.y][Point_e.x] == BKGRND_ID:
        flag2 = True
        bridge_c2 = Point(Point_e.x,Point_s.y)
    if(flag1 and detectHorizental(bridge_c1,Point_e,mat_pad) and detectVertical(bridge_c1,Point_s,mat_pad)):
        return True
    elif(flag2 and detectHorizental(bridge_c2,Point_s,mat_pad) and detectVertical(bridge_c2,Point_e,mat_pad)):
        return True
    else:
        return False

def detectTwoNode(Point_s,Point_e,mat_pad):
    for colx in range(COL_NUM):
        for rowy in range(ROW_NUM):
            if(mat_pad[rowy][colx] == BKGRND_ID):
                temp = Point(colx,rowy)
                if detectHorizental(temp,Point_s,mat_pad) and detectOneNode(temp,Point_e,mat_pad):
                    return True
                elif detectVertical(temp,Point_s,mat_pad) and detectOneNode(temp,Point_e,mat_pad):
                    return True
                elif detectOneNode(temp,Point_s,mat_pad) and detectHorizental(temp,Point_e,mat_pad):
                    return True
                elif detectOneNode(temp,Point_s,mat_pad) and detectVertical(temp,Point_e,mat_pad):
                    return True
    return False

def vanish(Point_s,Point_e,mat_pad):
    mat_pad[Point_s.y][Point_s.x] = BKGRND_ID
    mat_pad[Point_e.y][Point_e.x] = BKGRND_ID

def canVanish(Point_s,Point_e,mat_pad):
    if Point_s == Point_e:
        return False
    #已排除相同点的情况
    if(detectVertical(Point_s,Point_e,mat_pad) or detectHorizental(Point_s,Point_e,mat_pad)):
        vanish(Point_s,Point_e,mat_pad)
        return True
    if(detectOneNode(Point_s,Point_e,mat_pad)):
        vanish(Point_s, Point_e, mat_pad)
        return True
    if(detectTwoNode(Point_s,Point_e,mat_pad)):
        vanish(Point_s, Point_e, mat_pad)
        return True
    return False

def findPairs(mat_pad):
    count = 0
    rtn ={}
    for i in range(COL_NUM):
        for j in range(ROW_NUM):
            if(mat_pad[j][i] != BKGRND_ID):
                count +=1
                if rtn.__contains__(mat_pad[j][i]):
                    rtn[mat_pad[j][i]].append(Point(i,j))
                else:
                    rtn[mat_pad[j][i]] = []
                    rtn[mat_pad[j][i]].append(Point(i,j))
    return count/2,rtn

def test_VanishFuc():
    mat_pad = getMatrix()
    pairNum, pairs = findPairs(mat_pad)
    print(pairNum)
    if canVanish(Point(1,1),Point(4,4 ),mat_pad):
        print("yes")
    else:
        print("No")

def play():
    print("开始获得图像分类矩阵...")
    mat_pad = getMatrix()
    print("已经获得图像分类矩阵，准备配对...")
    pairNum,pairs = findPairs(mat_pad)
    print("一共匹配到"+ str(int(pairNum)) +"对")
    print("现在开始寻找解决方案...")
    ans =[]
    all_pairs = pairNum
    last_pairNum = pairNum
    while(pairNum > 0):
        for item in pairs:
            for idx in pairs[item]:
                if idx.x != -1:
                    for idx2 in pairs[item]:
                        if idx2.x  != -1:
                            if(canVanish(idx,idx2,mat_pad)):
                                print("找到一对{}/{}：点({},{}) 与 点({},{}) 可以消除".format(int(all_pairs)-int(pairNum) + 1,
                                                                                  int(all_pairs),
                                                                                  idx.x,idx.y,idx2.x,idx2.y))
                                pairNum -=1
                                ans.append((Point(idx.x,idx.y),Point(idx2.x,idx2.y)))
                                idx.x = -1
                                idx2.x = -1
        if last_pairNum == pairNum:
            break
        else:
            last_pairNum = pairNum
    print("解析完毕,剩余{}对还未匹配...".format(int(pairNum)))
    return ans

def idex2piexl(point):
    return (point.x) * SEGIMGCOL_W + CONEROFPIC_LT_X + SEGIMGCOL_W /2,\
           (point.y) * SEGIMGROW_H + CONEROFPIC_LT_Y + SEGIMGROW_H /2

def click(key):
    # width, height = pyautogui.size()
    # screen_center_x = width/2
    # screen_center_y = height/2
    # rect_len = 100
    # for i in range(3):
    #     pyautogui.moveTo(screen_center_x - rect_len, screen_center_y - rect_len, duration=0.25)
    #     pyautogui.moveTo(screen_center_x + rect_len, screen_center_y - rect_len, duration=0.25)
    #     pyautogui.moveTo(screen_center_x + rect_len, screen_center_y + rect_len, duration=0.25)
    #     pyautogui.moveTo(screen_center_x - rect_len, screen_center_y + rect_len, duration=0.25)
    # pyautogui.click(screen_center_x - rect_len, screen_center_y + rect_len,button='right')
    ans = play()
    all = len(ans)
    count = 1
    RUN_FLAG = True
    for item in ans:
        if RUN_FLAG == True:
            pyautogui.click(idex2piexl(item[0]), button='left',duration= 0.00001)
            pyautogui.click(idex2piexl(item[1]), button='left',duration= 0.00001)
            print("已消除{}/{}：点({},{}) 与 点({},{}) 已消除".format(count,
                                                              (all),
                                                             item[0].x, item[0].y, item[1].x, item[1].y))
            count += 1
        else:
            break
    print("点击完毕...")

def on_press(key):
    try:
        # print("正在按压:",format(key.char))
        if(key.char == 'q'):
            # Listener.stop()
            print("启动连连外挂2.0...")
            click(key)
    except AttributeError:
        print("正在按压:",format(key))

def start_listen():
    with Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == '__main__':
    while(1):
        print("请按Q键开始...")
        start_listen()