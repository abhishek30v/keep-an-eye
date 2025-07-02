# 🔍 Keep An Eye

> Real-Time AI Surveillance for Wildlife and Farm Safety

Keep An Eye is a real-time animal detection and alert system built using YOLOv8, OpenCV, Cloudinary, and Twilio. It captures live video streams, detects specific animals using a custom-trained model, uploads video clips to the cloud, and sends SMS alerts with GPS-based location information.

---

## ⚙️ Features

* 🎥 Detects animals from live IP/mobile camera feeds
* ✨ Uses a fine-tuned custom YOLOv8 model
* 📩 Sends SMS alerts using Twilio with:
  * Animal detected
  * Confidence level
  * GPS location (Google Maps link)
  * Link to the uploaded video
* 💾 Stores 5-second detection clips to Cloudinary
* ⏰ Runs continuously, with adjustable alert intervals

---

## 🌍 Use Cases

* 🔒 Human-wildlife conflict prevention
* 🤾 Farm safety from wild intrusions
* 🐾 Wildlife monitoring and documentation
* 🏛️ Smart rural & border surveillance

---

## 🔧 Tech Stack

* **YOLOv8 (Ultralytics)** — Object detection
* **Python 3.8+**
* **OpenCV** — Image processing
* **Cloudinary API** — Cloud video storage
* **Twilio API** — SMS alerts
* **Pillow** — Image handling
* **Logging** — Runtime tracking
* **python-dotenv** — Environment variable management

---

## 📚 Dataset

* Downloaded from **OpenImages**
* Total images: **6600+**

  * **\~5,300** for training
  * **\~1,300** for validation
* Classes: Tiger, Leopard, Cheetah, Elephant, Monkey, Deer, Lion, Bear, Pig, Bull
* Added **negative images without labels** to reduce false positives

---

## 🚀 How to Run the Project

### 1. Clone the Repo

```bash
git clone https://github.com/sourav-bisht/keep-an-eye.git
cd keep-an-eye
```

### 2. Set up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare `.env` File (or set environment variables)

```env
-->for twilio
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE_NUMBER=+1xxx
RECIPIENT_PHONE_NUMBER=+91xxx

-->for cloudinary
CLOUDINARY_CLOUD_NAME=your_name
CLOUDINARY_API_KEY=your_key
CLOUDINARY_API_SECRET=your_secret
```

### 5. Add Your Trained Model

Place your `best.pt` YOLOv8 model in the project root directory.

### 6. Run the Main Script

```bash
python main.py
```

> Press `q` to quit. Logs are saved in `/logs`.

---

## 🔌 Physical Deployment (Future Plan)

* Deploy on **Raspberry Pi** or **Banana Pi**
* Use Pi camera or USB webcam
* Connect via Wi-Fi or mobile dongle

---

## 🔫 Challenges Faced

* Cloudinary integration for large video uploads
* Model performance tuning and dataset curation
* Balancing detections using **negative samples**
* Stable frame-rate handling on live streams

---

## ✨ Author

**Sourav Bisht**
`Keep An Eye` is a project driven by a passion for using AI for real-world impact. Let’s connect!

---

## 📚 Resources & Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Twilio Python SDK](https://www.twilio.com/docs/libraries/python)
* [Cloudinary Python SDK](https://cloudinary.com/documentation/python_integration)
* [OpenImages Dataset](https://storage.googleapis.com/openimages/web/index.html)

---

## 🌟 Star this repo if you found it useful!

\#KeepAnEye #YOLOv8 #OpenCV #Twilio #Cloudinary #WildlifeMonitoring #EdgeAI #SmartFarming #PythonProject
