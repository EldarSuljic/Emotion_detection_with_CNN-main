# importovanje osnovnih biblioteka i dependencies
from types import FrameType
from keras.models import model_from_json
from tkinter import *
from tkinter import filedialog
from shutil import copyfile
from PIL import ImageTk, Image
import numpy as np
import cv2

# kreiranje mape emocija
emotion_dict = {0: "Ljut", 1: "Zgrozen", 2: "Prestrasen", 3: "Sretan", 4: "Neutralan", 5: "Tuzan", 6: "Iznenadjen"}

# ucitavanje json i istreniranog modela
json_file = open('C:\\Users\\Korisnik\\Desktop\\Emotion_detection_with_CNN-main\\model\\emocije_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# ucitavanje tezina u novi model
emotion_model.load_weights("C:\\Users\Korisnik\\Desktop\\Emotion_detection_with_CNN-main\\model\\emocije_model.h5")
print("Loaded model from disk")


# definisanje root (osnovnog) okvira
root = Tk()
root.geometry("700x400+422+128")
root.title("Detekcija emocija na licu")
root.configure(background="#D3D3D3")
root.configure(cursor="arrow")
root.resizable(0, 0)

# definisanje fontova koji se koriste u programu
font10 = "-family Georgia -size 11 -weight normal -slant roman" \
     " -underline 0 -overstrike 0"
font11 = "-family {Segoe UI} -size 7 -weight normal -slant " \
"roman -underline 0 -overstrike 0"

# definisanje glavnog okvira
mainFrame = Frame(root, width=715, height=150)
mainFrame.pack()
mainFrame.configure(background="#A9A9A9")
mainFrame.configure(highlightbackground="#2B2B2B")
mainFrame.configure(highlightcolor="black")

# modul za odabir slike koja ce se ispitivati
def odaberiSliku():

    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
    filetypes=(("jpg files", "*.jpg"), ("mp4 files", "*.mp4"), ("all files", "*.*")))
    vid = (root.filename)
    copyfile(vid, "test.mp4")

# modul za odabir videa koji ce se ispitivati
def odaberiVideo():

    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
    filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
    vid = (root.filename)
    copyfile(vid, "test.mp4")

# WEB KAMERA
def WebCamEmocije():

    cap = cv2.VideoCapture(0)

    while True:
        # pronalazenje Haarove kaskade za iscrtavanje okvira oko lica
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('C:\\Users\\Korisnik\\Desktop\\Emotion_detection_with_CNN-main\\haarcascades\\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detekcija lica dostupnih na kameri
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # uzimanje svakog lica dostupnog na slici i predprocesiranje
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predikcija emocija
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# VIDEO
def VideoEmocije():

    cap = cv2.VideoCapture('test.mp4')

    while True:
        # pronalazenje Haarove kaskade za iscrtavanje okvira oko lica
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('C:\\Users\\Korisnik\\Desktop\\Emotion_detection_with_CNN-main\\haarcascades\\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detekcija lica dostupnih na kameri
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # uzimanje svakog lica dostupnog na slici i predprocesiranje
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predikcija emocija
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# SLIKA
def slikaEmocija():
    cap = cv2.VideoCapture('test.mp4')

    while True:
        # pronalazenje Haarove kaskade za iscrtavanje okvira oko lica
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, 500))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('C:\\Users\\Korisnik\\Desktop\\Emotion_detection_with_CNN-main\\haarcascades\\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

         # detekcija lica dostupnih na kameri
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # uzimanje svakog lica dostupnog na slici i predprocesiranje
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predikcija emocija
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# WEB KAMERA-SLIKA
def WebCamShot():

    cap = cv2.VideoCapture(0)

    while True:
        # pronalazenje Haarove kaskade za iscrtavanje okvira oko lica
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('C:\\Users\\Korisnik\\Desktop\\Emotion_detection_with_CNN-main\\haarcascades\\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detekcija lica dostupnih na kameri
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # uzimanje svakog lica dostupnog na slici i predprocesiranje
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predikcija emocija
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cap.release()
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cap.release()
    cv2.destroyAllWindows()

# dugme za odabir slike
button_load = Button(mainFrame, text="ODABERI SLIKU", command=odaberiSliku)
button_load.configure(activebackground="#2f72ad")
button_load.configure(activeforeground="white")
button_load.configure(activeforeground="#000000")
button_load.configure(background="#2F4F4F")
button_load.configure(foreground="#B2B6B9")
button_load.configure(relief=FLAT)
button_load.configure(font=font10)
button_load.place(relx=0.02, rely=0.1, height=30, width=130)
button_load.configure(cursor="hand2")

# dugme za odabir videa
button_load = Button(mainFrame, text="ODABERI VIDEO", command=odaberiVideo)
button_load.configure(activebackground="#2f72ad")
button_load.configure(activeforeground="white")
button_load.configure(activeforeground="#000000")
button_load.configure(background="#2F4F4F")
button_load.configure(foreground="#B2B6B9")
button_load.configure(relief=FLAT)
button_load.configure(font=font10)
button_load.place(relx=0.56, rely=0.1, height=30, width=130)
button_load.configure(cursor="hand2")

# dugme za detekciju osmijeha na slici
button_execute = Button(mainFrame, text="DETEKTUJ EMOCIJU", command=slikaEmocija)
button_execute.configure(activebackground="#2f72ad")
button_execute.configure(activeforeground="white")
button_execute.configure(activeforeground="#000000")
button_execute.configure(background="#2F4F4F")
button_execute.configure(foreground="#B2B6B9")
button_execute.configure(relief=FLAT)
button_execute.configure(font=font10)
button_execute.place(relx=0.21, rely=0.1, height=30, width=160)
button_execute.configure(cursor="hand2")

# dugme za detekciju osmijeha na videu
button_load = Button(mainFrame, text="DETEKTUJ EMOCIJU", command=VideoEmocije)
button_load.configure(activebackground="#2f72ad")
button_load.configure(activeforeground="white")
button_load.configure(activeforeground="#000000")
button_load.configure(background="#2F4F4F")
button_load.configure(foreground="#B2B6B9")
button_load.configure(relief=FLAT)
button_load.configure(font=font10)
button_load.place(relx=0.75, rely=0.1, height=30, width=160)
button_load.configure(cursor="hand2")

# dugme za detekciju osmijeha na webcam
button_load = Button(mainFrame, text="WEBCAMLIVE", command=WebCamEmocije)
button_load.configure(activebackground="#2f72ad")
button_load.configure(activeforeground="white")
button_load.configure(activeforeground="#000000")
button_load.configure(background="#2F4F4F")
button_load.configure(foreground="#B2B6B9")
button_load.configure(relief=FLAT)
button_load.configure(font=font10)
button_load.place(relx=0.50, rely=0.51, height=30, width=130)
button_load.configure(cursor="hand2")

# dugme za detekciju osmijeha na webcam
button_load = Button(mainFrame, text="WEBCAMSHOT", command=WebCamShot)
button_load.configure(activebackground="#2f72ad")
button_load.configure(activeforeground="white")
button_load.configure(activeforeground="#000000")
button_load.configure(background="#2F4F4F")
button_load.configure(foreground="#B2B6B9")
button_load.configure(relief=FLAT)
button_load.configure(font=font10)
button_load.place(relx=0.31, rely=0.51, height=30, width=130)
button_load.configure(cursor="hand2")

# podaci
credits = Label(root)
credits.place(relx=0.17, rely=0.92, height=35, width=465)
credits.configure(background="#696969")
credits.configure(font=font10)
credits.configure(foreground="#bababa")
credits.configure(text='''''Detekcija emocija na licu primjenom dubokog ucenja"---Eldar Suljić''')
credits.configure(width=480)

root.mainloop()