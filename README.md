# smart-note-calculator

Smart Note Calculator ✍️➕📐
Smart Note Calculator is a handwritten expression recognition tool built with OpenCV and a deep learning model. It allows users to draw mathematical expressions on a virtual pad and get instant answers — mimicking real-time calculation on a smart digital notebook.

🔧 Features
Draw math expressions using a marker tool.

Supports equations and variables like x=5, y=3, and x+y=.

Real-time recognition using a trained deep learning model (.keras) for symbol and digit detection.

Clean interface with controls:

Marker and eraser tools.

Undo and clear all functionality.

Adjustable marker thickness.

Handles multi-line expressions with correct alignment of symbols like - and ..

📦 Tech Stack
Python

OpenCV

NumPy

Keras (for digit classification model)

Custom image processing & contour sorting.

📁 Folder Requirements
nums/ folder containing digit/symbol images for visual rendering of results.

new_num.keras is the trained model file used for symbol/digit recognition.

🚀 Usage
Run the script and draw equations using your mouse. Click on different tool options at the top to switch between marker, eraser, or clear. When you finish writing an expression with =, the result is computed and displayed automatically.
