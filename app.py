import streamlit as st
import cv2
import numpy as np
from PIL import Image
import ai_edge_litert.interpreter as litert
from google import genai

# ================= CONFIG =================
MODEL_PATH = "model.tflite"   # keep model.tflite in the same folder as this app.py
LABELS = ['Defective', 'Perfect']

st.set_page_config(page_title="Smart Repair System", page_icon="🚗", layout="centered")

# ================= GEMINI API KEY (safe way) =================
# Locally: create a file .streamlit/secrets.toml with:
#   GOOGLE_API_KEY = "your_key_here"
# On Streamlit Cloud: add the same in "App settings -> Secrets"
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found. Add it in .streamlit/secrets.toml (local) "
             "or in Streamlit Cloud's Secrets settings before scanning.")


# ================= CACHED RESOURCES (load only once) =================
@st.cache_resource
def load_interpreter():
    interpreter = litert.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


@st.cache_resource
def load_gemini_client(api_key):
    return genai.Client(api_key=api_key)


interpreter = load_interpreter()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]


# ================= GEMINI FUNCTION =================
def get_repair_estimate(client, cv2_image):
    try:
        img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        prompt = "Damage (max 5 words) & cost in INR. Format: Damage: X | Cost: ₹XXXX"

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[prompt, pil_image]
        )
        return response.text.strip() if response.text else "No response"

    except Exception:
        return "Info unavailable"


# ================= SHARED: run model + gemini on a captured frame =================
def process_frame(frame):
    """frame must be a BGR numpy array (OpenCV format)."""
    # ---- Model prediction ----
    img = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    label = LABELS[np.argmax(prediction)]

    # ---- Gemini estimate ----
    client = load_gemini_client(GOOGLE_API_KEY)
    if label == "Defective":
        details = get_repair_estimate(client, frame)
    else:
        details = "Car is perfect. No repair needed."

    # ---- Display ----
    img_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(img_rgb_display, use_container_width=True)

    color = "#ff4d4d" if label == "Defective" else "#00ff88"
    st.markdown(
        f"<p style='font-weight:bold; color:{color}; font-size:18px;'>Status: {label}</p>",
        unsafe_allow_html=True
    )
    st.write(details)


# ================= UI =================
st.markdown(
    "<h1 style='text-align:center;'>🚗 Smart Repair System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; opacity:0.7;'>Scan your car and get an instant damage report</p>",
    unsafe_allow_html=True
)

st.divider()

source = st.radio(
    "📷 Choose Camera Source",
    ["Device Camera (this device's own camera)", "IP Camera Stream (separate phone as camera)"],
    help="Device Camera works directly in the browser — great when you open this app "
         "on your phone and want to use its own camera. IP Camera Stream is for when a "
         "*different* phone (running the IP Webcam app) is feeding video to this app."
)

if source.startswith("Device Camera"):
    # Uses the browser's own camera (works on phone browsers too) — no IP address needed
    photo = st.camera_input("Take a photo of the car")

    scan_btn = st.button("🔍 Scan Vehicle", use_container_width=True)

    if scan_btn:
        if photo is None:
            st.warning("Please take a photo first.")
        elif not GOOGLE_API_KEY:
            st.warning("Gemini API key missing — add it in Secrets before scanning.")
        else:
            with st.spinner("Scanning..."):
                pil_img = Image.open(photo).convert("RGB")
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                process_frame(frame)

else:
    ip_address = st.text_input(
        "Enter the IP Camera URL",
        placeholder="e.g. http://192.168.1.5:8080/video",
        help="Open the 'IP Webcam' app on the phone acting as the camera, connect it to the same "
             "WiFi as this device, and paste the video URL it shows (usually ends in /video)."
    )

    scan_btn = st.button("🔍 Scan Vehicle", use_container_width=True)

    if scan_btn:
        if not ip_address:
            st.warning("Please enter the IP camera URL first.")
        elif not GOOGLE_API_KEY:
            st.warning("Gemini API key missing — add it in Secrets before scanning.")
        else:
            with st.spinner("Scanning..."):
                cap = cv2.VideoCapture(ip_address)
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    st.error("❌ Camera not working. Check the IP address (it changes per WiFi network / device) and try again.")
                else:
                    process_frame(frame)