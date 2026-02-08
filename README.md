# Fall_Detection_Project-
From Pose to Fall: LSTM and YOLOv11-Pose for Fall Detection in Healthcare

# 🧍 Fall Detection Using YOLOv11-Pose and LSTM

## 📌 Project Summary
This project develops an AI-based fall detection system for healthcare monitoring.  
The system combines pose estimation and deep learning to detect falls from video data and distinguish them from normal daily activities.

The goal is to support faster emergency response and reduce missed fall events, especially for elderly care.

---

## 🧠 Method

### Step 1 — Pose Extraction
- YOLOv11-Pose extracts **17 body keypoints** from video frames.

### Step 2 — Temporal Movement Analysis
- LSTM analyzes body movement over time to detect fall patterns.

This two-stage approach improves accuracy compared to single-frame detection methods.

---

## 📊 Results

- Fall Detection Recall: **99%**  
- Sleeping Detection Recall: **32%**

The model is highly effective at detecting falls. Future work focuses on reducing false alarms between sleeping and falling.

---

## 🗂 Dataset

**GMDCSA24 Human Fall Detection Dataset**

- Total videos: **160**  
- Includes activities such as:
  - Walking
  - Sitting
  - Sleeping
  - Exercising
  - Falling

Each video includes activity timestamps for sequence analysis.
Source: https://www-sciencedirect-com.ezproxy.hamk.fi/science/article/pii/S2352340924008552

## 🛠 Tools and Technologies

- Python  
- YOLOv5  
- YOLOv11-Pose  
- LSTM (Deep Learning)  
- OpenCV  
- NumPy  
- Pandas  
- Scikit-learn  

---

## 👩‍💻 My Contribution

- Designed fall detection pipeline  
- Converted video data into pose-based time-series data  
- Implemented data preprocessing and augmentation  
- Built and evaluated deep learning models  
- Analyzed model performance and limitations  

---

## 🚀 Future Improvements

- Improve sleeping vs fall classification  
- Test with larger real-world datasets  
- Optimize for real-time and edge deployment  

---

## 🏥 Application Areas

- Elderly monitoring  
- Smart healthcare systems  
- Hospital patient monitoring  
- Assisted living environments  

---

