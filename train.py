import tkinter as tk
from tkinter import Message, Text
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("Face_Recogniser")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
#answer = messagebox.askquestion(dialog_title, dialog_text)

window.geometry('1180x650')
window.configure(background="white")

#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


message = tk.Label(window, text="AUTOMATIC ATTENDANCE MANAGEMENT USING FACE RECOGNITION", font=(
    "times new roman", 25), bg="red", fg="white", height=2)

message.place(x=0, y=0)

lbl = tk.Label(window, text="Enter ID", width=20, height=2,
               fg="white", bg="Blue", font=('times new roman', 15, ' bold '))
lbl.place(x=0, y=100)

txt = tk.Entry(window, width=20, bg="white",
               fg="black", font=('times new roman', 15, ' bold '))
txt.place(x=270, y=110)

lbl2 = tk.Label(window, text="Enter Name", width=20, fg="white",
                bg="blue", height=2, font=('times new roman', 15, ' bold '))
lbl2.place(x=0, y=200)

txt2 = tk.Entry(window, width=20, bg="white",
                fg="black", font=('times new roman', 15, ' bold '))
txt2.place(x=270, y=210)

lbl3 = tk.Label(window, text="Notification : ", width=20, fg="white",
                bg="blue", height=2, font=('times new roman', 15, ' bold underline '))
lbl3.place(x=0, y=310)

message = tk.Label(window, text="", bg="light blue", fg="white", width=30,
                   height=2, font=('times new roman', 15, ' bold '))
message.place(x=270, y=310)

lbl3 = tk.Label(window, text="Attendance : ", width=20, fg="white",
                bg="blue", height=2, font=('times', 15, ' bold  underline'))
lbl3.place(x=0, y=560)


message2 = tk.Label(window, text="", fg="white", bg="blue",
                    activeforeground="green", width=30, height=2, font=('times', 15, ' bold '))
message2.place(x=270, y=560)


def clear():
    txt.delete(0, 'end')
    res = ""
    message.configure(text=res)


def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def TakeImages():
    Id = (txt.get())
    name = (txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum+1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name + "."+Id + '.' +
                            str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
                # display the frame
                cv2.imshow('frame', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id + " Name : " + name
        row = [Id, name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text=res)


def TrainImages():
    # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"  # +",".join(str(f) for f in Id)
    message.configure(text=res)


def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)

    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        # When translating a color image to black and white (mode “L”), the library uses the ITU-R 601-2 luma transform:  L = R * 299/1000 + G * 587/1000 + B * 114/1000
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for(x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            print(Id, conf)
            confidence = int(100*(1-(conf)/300))
            if(confidence > 75):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(
                    ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]

            else:
                Id = 'Unknown'
                tt = str(Id)
                noOfFile = len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) +
                            ".jpg", im[y:y+h, x:x+w])
            cv2.putText(im, str(tt), (x, y+h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    # print(attendance)
    res = attendance
    message2.configure(text=res)


clearButton = tk.Button(window, text="Clear", command=clear, fg="white", bg="blue",
                        width=15, height=2, font=('times new roman', 15, ' bold '))
clearButton.place(x=550, y=100)
clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="white", bg="blue",
                         width=15, height=2, font=('times new roman', 15, ' bold '))
clearButton2.place(x=550, y=200)
takeImg = tk.Button(window, text="Take Images", command=TakeImages, fg="white", bg="blue",
                    width=20, height=3, font=('times', 15, ' bold '))
takeImg.place(x=0, y=400)
trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="white",
                     bg="blue", width=20, height=3, font=('times', 15, ' bold '))
trainImg.place(x=280, y=400)
trackImg = tk.Button(window, text="Track Images", command=TrackImages, fg="white",
                     bg="blue", width=20, height=3, font=('times', 15, ' bold '))
trackImg.place(x=560, y=400)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="white", bg="blue",
                       width=20, height=3, font=('times', 15, ' bold '))
quitWindow.place(x=900, y=400)
copyWrite = tk.Text(window, background=window.cget(
    "background"), borderwidth=0, font=('times', 30, 'italic bold underline'))

window.mainloop()
