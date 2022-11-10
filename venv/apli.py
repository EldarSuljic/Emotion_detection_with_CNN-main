#importovanje osnovnih biblioteka i dependencies
from keras.models import model_from_json
from tkinter import *
from tkinter import filedialog
from shutil import copyfile
from PIL import ImageTk, Image
import numpy as np
import cv2

#OVAJ MOJ KOD
emotion_dict = {0: "Ljut", 1: "Zgadjen", 2: "Prestrasen", 3: "Sretan", 4: "Neutralan", 5: "Tuzan", 6: "Iznenadjen"}

# load json and create model
json_file = open('C:\\Users\\Korisnik\\Desktop\\Emotion_detection_with_CNN-main\\model\\emocije_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("C:\\Users\Korisnik\\Desktop\\Emotion_detection_with_CNN-main\\model\\emocije_model.h5")
print("Loaded model from disk")


#definisanje root (osnovnog) okvira
root = Tk()
root.geometry("600x500+422+128")
root.title("Detekcija emocija na licu")
root.configure(background="#2B2B2B")
root.configure(cursor="arrow")
root.resizable(0, 0)
#definisanje fontova koji se koriste u programu
font10 = "-family Georgia -size 11 -weight normal -slant roman" \
     " -underline 0 -overstrike 0"
font11 = "-family {Segoe UI} -size 7 -weight normal -slant " \
"roman -underline 0 -overstrike 0"

#definisanje glavnog okvira
mainFrame = Frame(root, width=600, height=150)
mainFrame.pack()
mainFrame.configure(background="#2B2B2B")
mainFrame.configure(highlightbackground="#2B2B2B")
mainFrame.configure(highlightcolor="black")

#definisanje lijevog okvira
leftFrame = Frame(root, width=190, height=200)
leftFrame.place(relx=0.03, rely=0.27, height=250, width=250)
leftFrame.configure(background="#2B2B2B")
leftFrame.configure(highlightbackground="#d9d9d9")
leftFrame.configure(highlightcolor="black")
leftFrame.pack_propagate(False)

#definisanje desnog okvira
rightFrame = Frame(root, width=190, height=200)
rightFrame.place(relx=0.55, rely=0.27, height=250, width=250)
rightFrame.configure(background="#2B2B2B")
rightFrame.configure(highlightbackground="#d9d9d9")
rightFrame.configure(highlightcolor="black")
rightFrame.pack_propagate(False)

#WEB KAMERA
def WebCamEmocije():

    cap = cv2.VideoCapture(0)

    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('C:\\Users\\Korisnik\\Desktop\\Emotion_detection_with_CNN-main\\haarcascades\\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cap.release()
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cap.release()
    cv2.destroyAllWindows()

#VIDEO
def VideoEmocije():

    cap = cv2.VideoCapture("C:\\Users\Korisnik\\video11.mp4")

    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('C:\\Users\\Korisnik\\Desktop\\Emotion_detection_with_CNN-main\\haarcascades\\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#SLIKA
def otkrijOsmijeh():
    for widget in rightFrame.winfo_children():
        widget.destroy()

        cap = cv2.imread("test.png")
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('C:\\Users\\Korisnik\\Desktop\\Emotion_detection_with_CNN-main\\haarcascades\\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        c = cv2.waitKey(0)

#dugme za odabir slike
button_load = Button(mainFrame, text="Odaberi sliku", command='')
button_load.configure(activebackground="#2f72ad")
button_load.configure(activeforeground="white")
button_load.configure(activeforeground="#000000")
button_load.configure(background="#515659")
button_load.configure(foreground="#B2B6B9")
button_load.configure(relief=FLAT)
button_load.configure(font=font10)
button_load.place(relx=0.03, rely=0.1, height=30, width=130)
button_load.configure(cursor="hand2")

#dugme za odabir videa
button_load = Button(mainFrame, text="Odaberi video", command='')
button_load.configure(activebackground="#2f72ad")
button_load.configure(activeforeground="white")
button_load.configure(activeforeground="#000000")
button_load.configure(background="#515659")
button_load.configure(foreground="#B2B6B9")
button_load.configure(relief=FLAT)
button_load.configure(font=font10)
button_load.place(relx=0.53, rely=0.1, height=30, width=130)
button_load.configure(cursor="hand2")

#dugme za detekciju osmijeha na slici
button_execute = Button(mainFrame, text="Detektuj osmijeh", command='')
button_execute.configure(activebackground="#2f72ad")
button_execute.configure(activeforeground="white")
button_execute.configure(activeforeground="#000000")
button_execute.configure(background="#515659")
button_execute.configure(foreground="#B2B6B9")
button_execute.configure(relief=FLAT)
button_execute.configure(font=font10)
button_execute.place(relx=0.25, rely=0.1, height=30, width=130)
button_execute.configure(cursor="hand2")

#dugme za detekciju osmijeha na videu
button_load = Button(mainFrame, text="Detektuj osmijeh", command=VideoEmocije)
button_load.configure(activebackground="#2f72ad")
button_load.configure(activeforeground="white")
button_load.configure(activeforeground="#000000")
button_load.configure(background="#515659")
button_load.configure(foreground="#B2B6B9")
button_load.configure(relief=FLAT)
button_load.configure(font=font10)
button_load.place(relx=0.75, rely=0.1, height=30, width=130)
button_load.configure(cursor="hand2")

#dugme za detekciju osmijeha na webcam
button_load = Button(mainFrame, text="Webcam", command=WebCamEmocije)
button_load.configure(activebackground="#2f72ad")
button_load.configure(activeforeground="white")
button_load.configure(activeforeground="#000000")
button_load.configure(background="#515659")
button_load.configure(foreground="#B2B6B9")
button_load.configure(relief=FLAT)
button_load.configure(font=font10)
button_load.place(relx=0.39, rely=0.31, height=30, width=130)
button_load.configure(cursor="hand2")

#podaci o autoru "credits"
credits = Label(root)
credits.place(relx=0.37, rely=0.96, height=21, width=434)
credits.configure(background="#2B2B2B")
credits.configure(font=font11)
credits.configure(foreground="#bababa")
credits.configure(text='''''Detekcija emocija na licu primjenom dubokog ucenja---Eldar Suljic''')
credits.configure(width=434)

root.mainloop()