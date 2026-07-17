# Brain Tumor Detection using MRI Images

This is an end-to-end executable Deep Learning project for detecting brain tumors from MRI images. The project utilizes a MobileNetV2 architecture with transfer learning in TensorFlow/Keras and features a beautiful, interactive web interface built with Streamlit.

This project was built for an MCA Final Year Research Project.

## Project Structure

- `dataset/`: (You must create this) Directory containing your MRI images sorted by class (e.g., glioma, meningioma, notumor, pituitary).
- `train_model.py`: Script to preprocess data, augment it, and train the deep learning model.
- `app.py`: The Streamlit web application for interactive tumor prediction.
- `test_app.py`: Unit tests for the prediction pipeline.
- `requirements.txt`: Python dependencies.

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python installed. Then run:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset
Download a Brain Tumor MRI dataset (such as the Brain Tumor MRI Dataset from Kaggle).
Extract it and place the folders into a directory named `dataset` in the root of this project.

The structure should look like this:
```
brain tumor detection/
│
├── dataset/
│   ├── glioma/       (contains .jpg/.png images)
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── train_model.py
├── app.py
...
```

### 3. Train the Model
Run the training script to train the CNN model on your dataset:
```bash
python train_model.py
```
*Note: This will generate `brain_tumor_model.h5` and `class_indices.json`.*

### 4. Run the Web Application
Start the Streamlit interface to test your model interactively:
```bash
streamlit run app.py
```
This will open a web browser where you can upload an MRI image and see the prediction results!

## Testing
To verify the application logic, you can run the unit tests using `pytest`:
```bash
pytest test_app.py
```
