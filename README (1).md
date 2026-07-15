# 🚗 Smart Repair System

An AI-powered web app that scans a car, detects whether it is damaged or in perfect condition, and — if damaged — instantly estimates the type of damage and approximate repair cost in INR.

**🔗 Live Demo:** [smart-car-repair-estimator.streamlit.app](https://smart-car-repair-estimator.streamlit.app)

---

## ✨ Features

- **Two camera input modes:**
  - 📱 **Device Camera** — take a photo directly from the browser (works on phone or laptop)
  - 📡 **IP Camera Stream** — connect a separate phone running an IP webcam app as the video source
- **On-device damage classification** using a custom-trained TensorFlow Lite model
- **AI-generated repair estimate** — Google Gemini API analyzes the damaged area and returns the likely damage type and estimated repair cost
- Clean, responsive Streamlit interface with instant visual feedback

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Frontend / App Framework | Streamlit |
| Computer Vision | OpenCV, Pillow |
| ML Model | TensorFlow Lite (`ai-edge-litert`) |
| Generative AI | Google Gemini API (multimodal image analysis) |
| Language | Python |

## ⚙️ How It Works

1. User captures/streams an image of the car (via device camera or IP camera)
2. A TensorFlow Lite model classifies the image as **Defective** or **Perfect**
3. If **Defective**, the image is sent to the **Gemini API** with a prompt asking for the damage type and an estimated repair cost
4. The result — status, image, damage description, and cost estimate — is displayed instantly

## 🚀 Running Locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/smart-car-repair-estimator.git
cd smart-car-repair-estimator

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key
# Create a file at .streamlit/secrets.toml with:
# GOOGLE_API_KEY = "your_api_key_here"

# Run the app
streamlit run app.py
```

> ⚠️ Never commit `.streamlit/secrets.toml` — it's already excluded via `.gitignore` to keep your API key private.

## 📁 Project Structure

```
smart-car-repair-estimator/
├── app.py                 # Main Streamlit application
├── model.tflite            # Trained damage-classification model
├── requirements.txt        # Python dependencies
└── .gitignore
```

## 📌 Notes

- The model currently classifies two states: **Defective** and **Perfect**
- Cost estimates are AI-generated approximations, not verified quotes from a mechanic

## 👤 Author

**Raj Kumar Dutta**
