# Integrating Custom STT, LLM, and TTS With Heygen Realtime

This repository integrates **custom Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech (TTS)** capabilities with **Heygen's Interactive Avatar** using real-time communication.

## 🚀 Features
- **Interactive AI Avatar**: Engage in real-time conversations with a Heygen avatar.
- **Custom STT, LLM, and TTS Integration**: Modify and enhance Heygen's pipeline with your own AI models.
- **Pipecat Flow Integration**: Utilize **pipecat** to handle real-time data processing and improve avatar interaction.
- **Voice Cloning Support**: Leverage ElevenLabs' voice cloning feature for a personalized experience.

## 🛠️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Harsh-Garg12/Integrating-Custom-STT-LLM-TTS-With-Heygen-Realtime.git
cd Integrating-Custom-STT-LLM-TTS-With-Heygen-Realtime
```

### 2️⃣ Setting Up `InteractiveAvatarNextJSDemo-realtime-alpha-demo`
This folder contains the interactive avatar UI. 
```bash
cd InteractiveAvatarNextJSDemo-realtime-alpha-demo
npm install --force  # Ensure Node.js version is 20.18.1 or older
npm audit fix --force
npm run build        # Rebuild when making code changes
npm run start        # Start the interactive avatar UI
```

### 3️⃣ Setting Up `pipecat-realtime-demo-main`
This folder manages the real-time STT, LLM, and TTS processing.
```bash
cd pipecat-realtime-demo-main
python -m venv <name of your virtual environment>
.\<name of your virtual environment>\Scripts\activate
pip install -r requirements.txt  # Ensure Python 3.10 or later (avoid 3.13 due to compatibility issues)
python main.py  # Start the real-time backend
```

## 🎭 Enjoy Real-Time AI Conversations
- Modify **TTS settings** inside `main.py` to use **ElevenLabs' voice cloning**.
- Create your **own avatar on Heygen**, then update the avatar **ID in the UI**.
- Experience a real-time AI conversation with your **digital twin!**

## 📝 Credits
This project is based on open-source repositories from **HeyGen**:
- **[InteractiveAvatarNextJSDemo](https://github.com/HeyGen-Official/InteractiveAvatarNextJSDemo)** (realtime-alpha-demo branch)
- **[pipecat-realtime-demo](https://github.com/HeyGen-Official/pipecat-realtime-demo)**

Modified and extended to support **custom STT, LLM, and TTS** integration.

## Resources Used
- Pipecat Documentation: [Pipecat Server LLM Services - Gemini](https://docs.pipecat.ai/server/services/llm/gemini)

---
⭐ If you find this project useful, consider starring the repo!
