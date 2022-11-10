#modul za odabir slike koja ce se ispitivati
def odaberiSliku():
    for widget in leftFrame.winfo_children():
        widget.destroy()
    for widget in rightFrame.winfo_children():
        widget.destroy()
    root.filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    img = (root.filename)
    copyfile(img,"test.jpg")
    photo = ImageTk.PhotoImage(Image.open("test.jpg"))
    labela_1 = Label(leftFrame, image=photo)
    labela_1.image = photo
    labela_1.pack(fill=BOTH, expand=True)

#SLIKA
def otkrijOsmijeh():
    for widget in rightFrame.winfo_children():
        widget.destroy()

        img = cv2.imread("test.jpg")
        # Find haar cascade to draw bounding box around face
        #ret, frame = img.read()
        #frame = cv2.resize(frame, (1280, 720))
        #if not ret:
        #    break
        faceCascade = cv2.CascadeClassifier('C:\\Users\\Korisnik\\Desktop\\Emotion_detection_with_CNN-main\\haarcascades\\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(img, emotion_dict[maxindex], (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        #cv2.imshow('Emotion Detection', img)
        cv2.imwrite("Emotion Detection.jpg",img)

        photo_2 = ImageTk.PhotoImage(Image.open("Emotion Detection.jpg"))
        labela_2 = Label(rightFrame, image=photo_2)
        labela_2.image = photo_2
        labela_2.pack(fill=BOTH, expand=True)

        c = cv2.waitKey(0)