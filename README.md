# Soccer Vision Tracker

A professional computer vision system designed for **tracking players and the ball in soccer games**, with automatic team assignment and court view transformation. This project leverages **YOLOv8** for detection, combined with custom tracking and analytical tools to provide actionable insights from soccer video footage.

---

## Project Description

Soccer Vision Tracker is a robust and modular computer vision solution for sports analytics. It detects players and the ball in each frame, assigns players to their respective teams, and transforms the video to a top-down court view for enhanced visualization and analysis. The system maintains persistent IDs for all players and the ball, enabling accurate tracking and analytics throughout the game.  

This project showcases expertise in **object detection, tracking algorithms, perspective transformation, and video processing**, making it ideal for sports analytics applications, automated highlight generation, and real-time player performance evaluation.

---

## Features

- **Player and Ball Detection:** Uses YOLOv8 for accurate detection of all players and the ball.  
- **Team Assignment:** Automatically differentiates players based on team colors and assigns consistent IDs.  
- **View Transformation:** Converts camera perspective to top-down view for improved spatial analysis.  
- **Tracking Across Frames:** Maintains persistent IDs for smooth tracking and event analysis.  
- **Video Output:** Generates annotated videos showing detected players, teams, and ball movements.  

---

## Technologies

- **Python 3.10+**  
- **[YOLOv8](https://ultralytics.com/)** for detection and tracking  
- **OpenCV** for video processing and visualization  
- **NumPy** for data manipulation  
- **Custom Python Modules** for team assignment, player-ball assignment, and view transformation  

---

## Installation

```bash
git clone https://github.com/<your-username>/soccer-vision-tracker.git
cd soccer-vision-tracker
pip install -r requirements.txt
```
---
# Usage

python main.py --video input_videos/game.mp4

---

# Project Structure

soccer-vision-tracker/

│
├── input_videos/           # Sample input videos
├── output_videos/          # Annotated output videos
├── trackers/               # Tracking classes
├── utils/                  # Utility functions (video reading, saving)
├── team_assigner.py        # Team assignment logic
├── player_ball_assigner.py # Player-ball assignment logic
├── view_transformer.py     # Perspective transformation logic
├── main.py                 # Entry point of the project
├── requirements.txt
└── README.md
---


# Results

- Accurate player and ball detection in diverse lighting conditions

- Correct team assignment and persistent tracking IDs

- Top-down view transformation for enhanced analytics

- Annotated video outputs ready for presentations or further analysis

---



