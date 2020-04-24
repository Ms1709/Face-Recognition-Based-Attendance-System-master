import tkinter as tk
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import cv2
import os
import shutil
import csv
import datetime
import time
import glob
import timeStamp

root = tk.Tk()
root.title("Attendance System")

frame_main = tk.Frame(root, bg="#E8E8E8")
frame_main.rowconfigure(0, minsize=400)
frame_main.columnconfigure([0, 1], minsize=350)
frame_main.grid(row=0, column=0)

# Sections
frame_attendance = tk.Frame(frame_main, bg="#FBFBFB", width=350, height=400)
frame_attendance.grid(row=0, column=0, padx=8, pady=8)
frame_register = tk.Frame(frame_main, bg="#FBFBFB", width=350, height=400)
frame_register.grid(row=0, column=1, padx=8, pady=8)
frame_notification = tk.Frame(frame_main, bg="#FBFBFB", height=50, width=716)
frame_notification.grid(row=1, column=0, columnspan=2, pady=8)

# Section Titles
lbl_attend = tk.Label(master=frame_attendance,
                      text="Attendance", font=("Helvetica", 20, 'bold'), fg="#212121", bg="#FBFBFB")
lbl_attend.place(x=95, y=10)

lbl_reg = tk.Label(master=frame_register,
                   text="Register Student", font=("Helvetica", 20, 'bold'), fg="#212121", bg="#FBFBFB")
lbl_reg.place(x=60, y=10)

lbl_noti = tk.Label(master=frame_notification, text="Notification:", font=(
    "Helvetica", 11, 'bold'), fg="#212121", bg="#FBFBFB")
lbl_noti.place(x=5, y=15)


lbl_ID = tk.Label(frame_register, text="Id",
                  font=("Helvetica", 13), bg="#FBFBFB")
lbl_ID.place(x=40, y=70)
ent_ID = tk.Entry(frame_register, width=20,
                  font=("Helvetica", 18), bg="#DCEBFF", borderwidth=1, relief=tk.GROOVE)
ent_ID.place(x=40, y=95)


lbl_name = tk.Label(frame_register, text="Name",
                    font=("Helvetica", 13), bg="#FBFBFB")
lbl_name.place(x=40, y=150)
ent_name = tk.Entry(frame_register, width=20,
                    font=("Helvetica", 18), bg="#DCEBFF", borderwidth=1, relief=tk.GROOVE)
ent_name.place(x=40, y=175)


lbl_noti_text = tk.Label(
    frame_notification, text="", anchor=tk.W, fg="#212121", bg="#FBFBFB", width=65, font=("Helvetica", 12))
lbl_noti_text.place(x=100, y=15)


frm_attendance = tk.Frame(master=frame_attendance,
                          width=334, height=230, bg="#F8EAD6")
frm_attendance.place(x=6, y=160)

frm_detail = tk.Label(frm_attendance, text="",
                      anchor=tk.W, width=35, bg="#F8EAD6")
frm_detail.place(x=7, y=8)


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(value)
        return True
    except (TypeError, ValueError):
        pass

    return False


def TakeImages():
    Id = (ent_ID.get())
    name = (ent_name.get())
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
                cv2.imwrite("TrainingImage/ "+name + "."+Id + '.' +
                            str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
                # display the frame
                cv2.imshow('Capturing Face...', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 20:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id + " Name : " + name
        row = [Id, name]
        with open('StudentDetails/StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        lbl_noti_text.configure(text=res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            lbl_noti_text.configure(text=res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            lbl_noti_text.configure(text=res)


def TrainImages():
    # $cv2.createLBPHFaceRecognizer()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    res = "Image Trained"  # +",".join(str(f) for f in Id)
    lbl_noti_text.configure(text=res)


def getImagesAndLabels(path):
    # get the path of all the files in the folder
    # imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    imagePaths = [f for f in glob.glob(path+'/*.jpg')]

    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
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
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails/StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for(x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if(conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(
                    ts).strftime('%H:%M')
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]

            else:
                Id = 'Unknown'
                tt = str(Id)
            if(conf > 75):
                noOfFile = len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown/Image"+str(noOfFile) +
                            ".jpg", im[y:y+h, x:x+w])
            cv2.putText(im, str(tt), (x, y+h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('Detecting Student...', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance/Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    # print(attendance)
    res = attendance
    frm_detail.configure(text=res)


btn_capture = tk.Button(master=frame_register, command=TakeImages, text="Capture Images", background="#7EEE7C", font=(
    "Helvetica", 14), fg="#212121", width=24, height=2, borderwidth=0)
btn_capture.place(x=40, y=240)

btn_train = tk.Button(master=frame_register, command=TrainImages, text="Train Model", background="#7CFE99", font=(
    "Helvetica", 14, 'bold'), fg="#212121", width=22, height=2, borderwidth=0)
btn_train.place(x=40, y=320)

btn_track = tk.Button(master=frame_attendance, command=TrackImages, text="Track Image", background="#7EEE7C", font=(
    "Helvetica", 14), fg="#212121", width=25, height=2, borderwidth=0, activebackground="#94f0bb")
btn_track.place(x=34, y=80)

root.mainloop()
