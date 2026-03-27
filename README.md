# Solar Panel Defect Detection System ☀️

An AI-powered application that detects defects in solar panels using deep learning models and generates detailed PDF reports.

---

## 📌 Project Overview

This project is designed to help in **automatic inspection of solar panels** by detecting defects using a trained AI model (YOLO).

The system consists of:

* **Frontend**: JavaFX application for user interaction
* **Backend**: Python Flask server for AI processing
* **AI Model**: YOLO model trained to detect solar panel defects

---

## 🏗️ Project Structure

```
demo/
├── src/main/java/com/example/demo/     # JavaFX Frontend
├── src/main/backend/                   # Python Flask Backend
│   ├── models/                         # AI model files (YOLO)
│   │   └── best.pt
└── pom.xml                            # Maven configuration
```

---

## ⚙️ Prerequisites

* Java 17+
* Maven 3.6+
* Python 3.8+
* pip

---

## 🚀 Setup Instructions

### 1️⃣ Backend Setup

```bash
cd src/main/backend
pip install -r requirements.txt
python run_backend.py
```

Backend runs on: `http://127.0.0.1:5000`

---

### 2️⃣ Frontend Setup

```bash
mvn clean compile
mvn javafx:run
```

---

## 🧠 Features

* 🔍 **Defect Detection**: Detects cracks, contamination, and other defects in solar panels
* 🤖 **AI Model**: Uses YOLO for real-time object detection
* 📄 **PDF Reports**: Generates professional reports with detected defects
* 🖥️ **User-Friendly UI**: Built with JavaFX

---

## 📷 How It Works

1. Upload a solar panel image
2. The backend processes it using the AI model
3. Detected defects are highlighted
4. A PDF report is generated automatically

---

## ⚠️ Troubleshooting

### Backend مشاكل

* تأكدي إن الموديل موجود في `models/best.pt`
* تأكدي إن السيرفر شغال

### Frontend مشاكل

* تأكدي من Java و Maven

### Connection

* لازم backend يشتغل قبل frontend

---

## 📦 Technologies Used

### Frontend

* JavaFX

### Backend

* Flask
* Ultralytics YOLO

### Other

* OpenCV
* ReportLab / PDFBox

---

## 📄 License

This project is for educational purposes.
