from ultralytics import YOLO
import cv2
import numpy as np
from matplotlib import pyplot as plt
import easyocr
import math

class model:
    def __init__(self,model_path = "best.pt",):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.reader = easyocr.Reader(['en'],gpu = True) 

    def pred(self,image_path):
        result = self.model.predict(image_path)
        data = result[0].boxes.data
        contours = np.array(data.cpu())
        return contours
    
    def show_pred(self,image_path,read_pred = False,low = 113,clean_text = True,save_pred = False,show_on_image = False,auto_low = False):
        result = self.model.predict(image_path,verbose = False)
        data = result[0].boxes.data
        contours = np.array(data.cpu())
        image = cv2.imread(image_path)
        if contours is None:
            print("no num plate found")
            return
        height, width, _ = image.shape

        thickness = math.ceil(min(width, height) * 1e-3)+1
        for (x1,y1,x2,y2,conf,_) in contours:
            x1 , y1, x2, y2 = int(x1),int(y1) , int(x2) , int(y2)
            if(read_pred):
                num_plate = image[y1:y2,x1:x2]
                text_final = ""
                if(auto_low):
                    text_final,conf,low=self.auto_low_preprocess(num_plate)
                    print(low)
                else:
                    num_plate = self.preprocess(num_plate,low)
                    text = self.reader.readtext(num_plate)
                    for(_,st,conf) in text:
                        text_final+=st;
                if(clean_text):
                    out = ""
                    text_final = text_final.upper()
                    text_final = text_final.replace(" ","")
                    text_final = text_final.replace("IND","")
                    text_final = text_final.replace("IN","")
                    text_final = text_final.replace("ND","")
                    text_final = text_final.replace("ID","")
                    last_four = text_final[-4:]
                    text_list = list(text_final)
                    for i in range(0,4):
                        char = last_four[i]
                        if(char == 'Z'):
                            text_list[i-4] = '4'
                    if(text_final[0] == '0' or text_final[0] == 'O'):
                        text_list[0] = 'D'  
                    if(text_final[2] == 'Z'):
                        text_list[2] = '2'  
                    for i in text_list:
                        out+=i
                print(out)
            width = abs(x2-x1)/2
            if(show_on_image):
                text = out
            else:
                text = str(conf)[:5]
            scale = self.font_scale(text,width)
            if(conf >= 0.5):
                text_color = (0,255,0)
            else:
                text_color = (0,0,255)
            cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),thickness=thickness,color= text_color)
            cv2.putText(image,text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,scale, text_color, thickness)
        if(save_pred):
            cv2.imwrite(out+"read.jpg",image)
        cv2.imshow("results",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def font_scale(self,text, width):
        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
            new_width = textSize[0][0]
            if (new_width <= width):
                return scale/10
        return 1
    
    def preprocess(self,image,low = 113):
        bw = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(bw,low,255,cv2.THRESH_BINARY_INV)
        return thresh
    
    def auto_low_preprocess(self,num_plate):
        bw = cv2.cvtColor(num_plate,cv2.COLOR_BGR2GRAY)
        final_text = ""
        final_conf = 0
        low = 0
        for i in range(95,120):
            _,thresh = cv2.threshold(bw,i,255,cv2.THRESH_BINARY_INV)
            text = self.reader.readtext(thresh)
            text_final = ""
            count = 0
            conff = 0
            conf_sum = 0
            for(_,st,conf) in text:
                text_final+=st;
                count +=1
                conf_sum+=conf
            conff = conf_sum/count
            
            if(conff>final_conf):
                final_conf = conff
                final_text = text_final
                low = i
        return final_text,final_conf,low


    


