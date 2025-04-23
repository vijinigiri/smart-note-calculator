import cv2
import numpy as np
import os
from keras.saving import load_model
import re

active_options = [1,1]
def select_option(x1,y1):
    global img,img_output,black_img,active_options
    
    if y1<50:
        line_img[:] = black_img
        if x1<100:
            print("erase all")
            img[:] = black_img
            var_lst.clear()
            img_output[:] = black_img
        elif x1<200:
            print("erase")
            dct["thickness"] = 20
            dct["parameters"] = "erase"
            active_options[0] = 2
            active_options[1] = 1
        elif x1<280:
            print("marker")
            dct["thickness"] = 5
            dct["parameters"] = "marker"
            active_options[1] = 2
            active_options[0] = 1
        elif x1<400:
            if len(prev_imgs):
                img[:] = prev_imgs.pop()
            img_output[:] = black_img
            print("undo")


def nav_bar(nav):
    global thickness_ball,nav_img,active_options
    if nav:
        nav_img[:]=(0,0,0)
        color = (0,255,255)
        cv2.putText(nav_img, f'earse all', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,1, cv2.LINE_AA)
        cv2.putText(nav_img, f'erase', (130,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,1*active_options[0], cv2.LINE_AA)
        cv2.putText(nav_img, f'marker', (200,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,1*active_options[1], cv2.LINE_AA)
        cv2.putText(nav_img, f'undo', (280,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,1, cv2.LINE_AA)

        cv2.putText(nav_img, f'thickness :{dct['thickness']}', (380,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1, cv2.LINE_AA)
        cv2.line(nav_img,(500,40),(800,40),(255,255,255),2)
        cv2.circle(nav_img,(thickness_ball,40),10,(0,0,255),-1)

        cv2.line(nav_img,(0,100),(width,100),(255,255,255),2)

    return  (0,nav_img)

thickness_ball = 500
def thickness_bar(x1,y1):
    global thickness_ball
    if y1<50 and x1>500 and x1< 800:
        thickness_ball = x1
        thickness = np.abs(int((x1-400)/20))
        dct['thickness'] = thickness




def get_final_img_output(text,nums):
    if len(text) > 1:
        first_digit = nums[text[0]]
        for i in range(1,len(text)):
            second_digit = nums[text[i]]
            h1, h2 = first_digit.shape[0], second_digit.shape[0]
            if text[i-1] == "-":
                pad_top = (h2 - h1) // 2
                pad_bottom = h2 - h1 - pad_top
                first_digit = cv2.copyMakeBorder(first_digit, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            elif text[i] == "." :
                pad_top = (h1 - h2) // 2
                pad_bottom = h1 - h2 - pad_top
                second_digit = cv2.copyMakeBorder(second_digit, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                if h1 < h2:
                    new_width = int(second_digit.shape[1] * (h1 / h2))
                    second_digit = cv2.resize(second_digit, (new_width, h1))
                elif h2 < h1:
                    new_width = int(first_digit.shape[1] * (h2 / h1))
                    first_digit = cv2.resize(first_digit, (new_width, h2))

            padding = np.zeros((first_digit.shape[0], 10, first_digit.shape[2]), dtype=np.uint8)
            first_digit = cv2.hconcat([first_digit,padding, second_digit])
        return first_digit
    
    return nums[text]



def show_answer(output_text,nums,i,j):
    global img_output,width
    i=i+50
    j=j+10
    final_output = get_final_img_output(output_text,nums)
    limit = width-j
    if final_output.shape[1]>limit and final_output.shape[1]<limit*1.4:
        new_height = int(final_output.shape[0] * (limit / final_output.shape[1]))
        final_output = cv2.resize(final_output,(limit,new_height))
    elif final_output.shape[1]>limit and final_output.shape[1]>limit*1.4:
        i=i+150
        j=j-final_output.shape[1]
    end_i = min(i + final_output.shape[0], img_output.shape[0])
    end_j = min(j + final_output.shape[1], img_output.shape[1])
    rows = end_i - i
    cols = end_j - j
    img_output[i:end_i, j:end_j] = final_output[:rows, :cols]

def insert_multiplication(expr):
    expr = re.sub(r'(\d)(\()', r'\1*\2', expr)
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])(\()', r'\1*\2', expr)
    return expr


def clean_variables(text,indices):
    global nav_img,var_lst
    lines = text.strip().split(',')
    # pattern = r"[a-zA-Z]+\s*=\s*\d+"
    pattern = r'^[xy]\d*=\d+$'
    unmatched = []
    var_lst = []
    required_indices = []
    for i in range(len(lines)):
        if re.fullmatch(pattern, lines[i]):
            exec(lines[i],globals())
            var_lst.append(lines[i])
        else:
            if lines[i]!="" and lines[i][-1]=="=":
                required_indices.append(indices[i])
            unmatched.append(lines[i])
    return unmatched,required_indices


def warn(e,no):
    img_output[:] = black_img
    cv2.putText(nav_img, "Please draw correcly", (650,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),1, cv2.LINE_AA)
    print("calculate"+no)
    print(e)

def calc_output(text):
    output_text = eval(text[:-1])
    if output_text == int(output_text):
        output_text = str(int(output_text))
    else:
        output_text = format(output_text, '.1f')
    return output_text
    

def calculate(text,nums,indices):
    global img_output
    nav_bar(1)
    expressions,indices = clean_variables(text,indices)
    full_text = ""
    for i in range(len(expressions)):
        text = insert_multiplication(expressions[i])
        full_text = full_text+text
        tag,co = 1,(0,255,255)
        try:
            if len(text) >2 and text[-1] == "="  :
                if len(expressions)==1:
                    img_output[:] = black_img
                output_text = calc_output(text)
                text = text+output_text
                full_text = full_text+output_text+" "
                show_answer(output_text,nums,indices[i][0],indices[i][1])
                tag = 0
        except Exception as e:
            warn(e,"1")
        try:
            text = "".join(expressions)
            if "=" in text and text[0]!= "=" and text[-1]!="=" and tag:
                img_output[:] = black_img
                if eval(insert_multiplication(text[:text.index("=")])) == eval(insert_multiplication(text[text.index("=")+1:])):
                    co = (0,255,0)
                else:
                    co = (0,0,255)
                    cv2.putText(nav_img, "not equal", (600,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, co,1, cv2.LINE_AA)
        except Exception as e:
            warn(e,"2")
        print(full_text)
        cv2.putText(nav_img, full_text, (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, co,1, cv2.LINE_AA)

def process_img(digit):
    gray_digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    pad_digit = np.pad(gray_digit, pad_width=((20, 20), (20, 20)), mode='constant', constant_values=0)
    resized_digit = cv2.resize(pad_digit,(28,28))
    fin_digit = np.where(resized_digit>50,255,0)
    return fin_digit
    
# def sort_contours(contours, row_threshold=10):
#     bounding_boxes = [cv2.boundingRect(c) for c in contours]
#     rows = []
#     for i, box in enumerate(bounding_boxes):
#         x, y, w, h = box
#         placed = False
#         for row in rows:
#             if abs(row[0][1] - y) < row_threshold:
#                 row.append((x, y, i))
#                 placed = True
#                 break
#         if not placed:
#             rows.append([(x, y, i)])
#     sorted_indices = []
#     for row in sorted(rows, key=lambda r: r[0][1]):
#         row_sorted = sorted(row, key=lambda x: x[0])
#         sorted_indices.extend([x[2] for x in row_sorted])
#     return [contours[i] for i in sorted_indices]

def sort_bowding_boxes(bounding_boxes,row_threshold=0):
    x5,y5,w5,h5 = bounding_boxes[0]
    min = np.sqrt((x5**2)*0.1+(y5**2))
    for x6,y6,_,h6 in bounding_boxes:
        if np.sqrt((x6**2)*0.1+(y6**2))<min:
            x5,y5,h5 = x5,y6,h6
    
    thresold = y5+h5-row_threshold
    # cv2.line(line_img,(0,thresold+100),(width,thresold+100),(50,50,50),1)
    lst1,lst2 = [],[]
    for i in range(len(bounding_boxes)):
        if bounding_boxes[i][1]<thresold:
            lst1.append(bounding_boxes[i])
        else:
            lst2.append(bounding_boxes[i])
    lst1 = sorted(lst1, key=lambda x: x[0])
    if len(lst2) ==0:
        return lst1
    lst1.extend(sort_bowding_boxes(lst2,row_threshold=0))
    return lst1


def detect_img(img1): 
    global digit5,nav,nums,fin_digit
    bounding_boxes,lst,prev = 0, [], " "

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    try:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        outer_contours = [c for i, c in enumerate(contours) if hierarchy[i][3] == -1]
        bounding_boxes = sort_bowding_boxes([cv2.boundingRect(c) for c in outer_contours])
    except:
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = sort_bowding_boxes([cv2.boundingRect(c) for c in contours])
    indices = []
    for x2, y2, w, h in bounding_boxes:
        digit5 = img1[y2:y2+h, x2:x2+w]
        fin_digit = process_img(digit5)
        pred = model.predict(fin_digit.reshape(1,28,28))    
        y_pred = str(np.argmax(pred))
        try:

            if prev == '-' and symbles.get(y_pred)=='-':
                lst.pop()
                lst.append('=')
                y_pred = "="
                indices.append((y2,x2+w))
            elif int(y_pred)>=10:
                lst.append(symbles[y_pred]) 
                y_pred = symbles[y_pred]
            else:
                lst.append(y_pred)
                if np.max(pred)==1:
                    pass
                    # nums[y_pred] = digit5
            prev = y_pred
        except Exception as e:
            print("detect_img")
            print(e)
    text = "".join(lst) 
    calculate(text,nums,indices)

x1_start,y1_start = 0,0
trigger,nav = 0,1
prev_x,prev_y = 0,0
def mouse_tracking(event,x,y,flags,param):
    global x1_start, y1_start
    global img,img_show,prev_img,img_output
    global background_color
    global dct,prev_x,prev_y,trigger
    global nav,img_nav_bar,line_img


    if event==1:
        x1_start,y1_start,trigger = x,y,1
        select_option(x1_start,y1_start)
    elif event == 4:
        if dct["parameters"]=="marker" and y1_start>100:
            prev_imgs.append(img.copy())
            detect_img(img[102:,])
        trigger=0
        prev_x, prev_y = 0, 0 
    if trigger:
        if y1_start<100:
            nav = 1
            thickness_bar(x,y)
        if dct["parameters"]=="marker":
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            cv2.line(img, (prev_x, prev_y), (x, y), dct['color'], dct['thickness'])
            prev_x, prev_y = x, y
        elif dct["parameters"] == "erase":
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            cv2.line(img_output, (prev_x, prev_y), (x, y), (0,0,0), dct['thickness'])
            cv2.line(img, (prev_x, prev_y), (x, y), (0,0,0), dct['thickness'])
            prev_x, prev_y = x, y

    cv2.putText(nav_img, ",".join(var_lst), (400,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),1, cv2.LINE_AA)
    nav,img_nav_bar = nav_bar(nav)
    img[:102] = img_nav_bar
    img_show = cv2.add(img_output,img)
    # img_show = cv2.add(img_show,line_img)
    
model = load_model("new_num.keras")
symbles = {"10":'(',"11":')',"12":'/',"13":'*',"14":'+',
           "15":'-',"16":",","17":".","18":"y","19":"x"}
nums,var_lst ={}, []
for i in os.listdir("nums"):
    nums[i[0]] = cv2.imread(f"nums/{i}")
height,width = 800-50,1050
background_color = (0,0,0)
dct = {"parameters" : "marker","thickness":5,"color":(255,255,255)}
cv2.namedWindow("drawing_pad")
cv2.setMouseCallback("drawing_pad",mouse_tracking)
img = np.full((height,width,3),background_color,dtype=np.uint8)
img_show,prev_img = img.copy(),img.copy()
img_output,black_img = img.copy(),img.copy()
line_img = img.copy()
nav_img = np.full((102,width,3),background_color,dtype=np.uint8)
prev_imgs = [np.full((height,width,3),background_color,dtype=np.uint8)]
count = 0
fin_digit =0 
while True:
    key = cv2.waitKey(1)
    cv2.imshow("drawing_pad",img_show)
    if key == 0:
        break
    if key == ord("s"):
        
        cv2.imwrite(f"one/{count}.jpg",fin_digit)
        count = count+1
        print(fin_digit.shape)
        print("one ",count," saved")
        img = np.full((height,width,3),background_color,dtype=np.uint8)
cv2.destroyAllWindows()
