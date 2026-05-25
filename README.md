# 🎱 Bingo Vision System

Bingo is a computer vision project that combines multiple deep learning models to automatically detect, recognize, and process bingo cards in real time through a camera interface.

The system integrates object detection and image classification techniques to identify bingo cards, detect every numbered circle, and recognize the corresponding numbers automatically.

---

# 🚀 Features

* Real-time bingo card detection
* Automatic circle localization
* CNN-based number recognition
* Camera-based registration system
* Interactive bingo interface
* End-to-end computer vision pipeline

---

# 🧠 Models Used

## 1. Card Detection — `detectCard.pt`

YOLO-based model trained to detect bingo cards inside the camera frame.

## 2. Circle Detection — `detectCircles.pt`

YOLO-based model trained to detect every numbered circle within a bingo card.

## 3. Number Recognition — `cnnModel.pth`

Custom Convolutional Neural Network (CNN) built from scratch to classify the detected numbers.

---

# 🔄 Pipeline

```text
┌────────────────┐
│ Card Detection │
└───────┬────────┘
        ↓
┌──────────────────┐
│ Circles Detection│
└───────┬──────────┘
        ↓
┌────────────────────────┐
│ Number Recognition CNN │
└────────────────────────┘
```

The pipeline works sequentially:

1. Detect the bingo card from the camera input.
2. Detect all circles inside the detected card.
3. Crop each detected circle.
4. Predict the number contained in each crop using the CNN classifier.

---

# 📂 Project Structure

```bash
Bingo/
│
├── app.py                 # Main application and interface
├── processInput.py        # Image preprocessing and crop extraction
├── cnn.py                 # CNN architecture
├── detectCard.pt          # YOLO model for card detection
├── detectCircles.pt       # YOLO model for circle detection
├── cnnModel.pth           # Trained CNN weights
├── requirements.txt
└── README.md
```

---

# 🖥️ How It Works

The application interface is divided into two main stages:

## 1. Card Registration

The system captures bingo cards using a camera.

During this stage:

* The card is detected after pressing space bar
* Every numbered circle is localized
* Each circle is cropped individually
* The CNN predicts the corresponding number

`processInput.py` is responsible for processing the detections and generating the crops used by the CNN classifier.

## 2. Bingo Game Interface

Once the cards are registered, the interface allows the bingo game to be managed using the detected information.

---

# 🏋️ Training

The YOLO models were trained using pretrained YOLO weights and annotations created with Label Studio.

The CNN model was developed and trained from scratch using PyTorch.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Elvenao/Bingo.git
cd Bingo
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python app.py
```

---

# 📦 Main Dependencies

* PyTorch
* OpenCV
* Ultralytics YOLO
* NumPy
* Pillow

---

# 📸 Future Improvements

* Improved CNN accuracy
* Real-time number validation
* Better UI/UX design

---

# 👨‍💻 Author

Developed by Emilio Hernández Sosa

GitHub: https://github.com/Elvenao
