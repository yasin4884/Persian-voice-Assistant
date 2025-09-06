import requests
import logging
import sqlite3
import subprocess
import numpy as np
import soundfile as sf
import sounddevice as sd
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import pygame
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# مسیرها
MODEL_PATH = r"C:\Users\yasin\Desktop\fine tune"
WELCOME_AUDIO_PATH = r"C:\Users\yasin\Desktop\openai-fm-sage-old-timey.wav"

# متغیرهای سراسری برای مدل
processor = None
whisper_model = None

def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (prompt TEXT, response TEXT)''')
    conn.commit()
    conn.close()

def load_whisper_model():
    """بارگذاری مدل Whisper فاین‌تیون شده"""
    global processor, whisper_model
    
    if processor and whisper_model:
        return True
        
    try:
        print("🔄 در حال بارگذاری مدل Whisper...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_PATH).to("cpu")
        print("✅ مدل Whisper بارگذاری شد!")
        return True
    except Exception as e:
        print(f"❌ خطا در بارگذاری مدل: {e}")
        return False

def play_welcome_audio():
    """پخش صدای خوشامدگویی"""
    try:
        print("🎵 در حال پخش صدای خوشامدگویی...")
        pygame.mixer.init()
        pygame.mixer.music.load(WELCOME_AUDIO_PATH)
        pygame.mixer.music.play()
        
        # صبر تا پخش تمام شود
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        pygame.mixer.quit()
        print("✅ پخش صدا تمام شد!")
        
    except Exception as e:
        print(f"❌ خطا در پخش صدا: {e}")

def listen_for_wake_word():
    """گوش دادن برای کلمه بیداری 'دستیار'"""
    try:
        duration = 3  # ثانیه
        sample_rate = 16000
        
        print("👂 در حال گوش دادن برای کلمه 'دستیار'...")
        
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        
        # تبدیل به آرایه 1 بعدی
        audio = np.squeeze(audio)
        return audio, sample_rate
        
    except Exception as e:
        print(f"❌ خطا در ضبط صدا: {e}")
        return None, None

def check_for_wake_word(audio_data, sample_rate):
    if not processor or not whisper_model:
        return False
    
    try:
        input_features = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_features
        
        with torch.no_grad():
            predicted_ids = whisper_model.generate(input_features)
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        wake_words = ["دستیار", "assistant", "دستیاری","دست یار","هی","هعی"]
        transcription_lower = transcription.lower()
        
        for wake_word in wake_words:
            if wake_word in transcription_lower:
                print(f"✅ کلمه بیداری تشخیص داده شد: {transcription}")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ خطا در بررسی کلمه بیداری: {e}")
        return False

def check_for_stop_word(audio_data, sample_rate):
    if not processor or not whisper_model:
        return False
    
    try:
        input_features = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_features
        
        with torch.no_grad():
            predicted_ids = whisper_model.generate(input_features)
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        stop_words = ["تمام", "stop", "finish", "خلاص"]
        transcription_lower = transcription.lower()
        
        for stop_word in stop_words:
            if stop_word in transcription_lower:
                print(f"🛑 کلمه توقف تشخیص داده شد: {transcription}")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ خطا در بررسی کلمه توقف: {e}")
        return False

def record_user_command():
    """ضبط دستور کاربر پس از تشخیص کلمه بیداری"""
    try:
        duration = 5  
        sample_rate = 16000
        
        print("🎤 شروع ضبط دستور... (5 ثانیه)")
        print("📢 الان دستور خود را بگویید...")
        
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        
        print("⏹️ ضبط دستور تمام شد!")
        
        audio = np.squeeze(audio)
        return audio, sample_rate
        
    except Exception as e:
        print(f"❌ خطا در ضبط دستور: {e}")
        return None, None

def transcribe_audio(audio_data, sample_rate):
    """تبدیل صوت ضبط شده به متن"""
    if not processor or not whisper_model:
        print("❌ مدل Whisper بارگذاری نشده است!")
        return None
    
    try:
        print("🔄 در حال تشخیص گفتار...")
        
        input_features = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_features
        
        with torch.no_grad():
            predicted_ids = whisper_model.generate(input_features)
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        print(f"📝 متن شناسایی شده: {transcription}")
        return transcription
        
    except Exception as e:
        logger.error(f"خطا در تشخیص گفتار: {e}")
        return None

def gemma3(prompt):
    try:
        response = requests.post('http://localhost:11434/api/generate',
                               json={"model": "gemma3:4b", "prompt": prompt, "stream": False})
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        logger.error(f"Error in Gemma3: {e}")
        return None

def save_to_db(prompt, response):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("INSERT INTO history (prompt, response) VALUES (?, ?)", (prompt, response))
    conn.commit()
    conn.close()

def find_best_keyword_match(user_input, ai_response):
    user_input_lower = user_input.lower()
    ai_response_lower = ai_response.lower() if ai_response else ""
    
    keywords = [
        ("نوت پد", "notepad"),
        ("notepad", "notepad"),
        ("ماشین حساب", "calculator"),
        ("calculator", "calculator"),
        ("calc", "calculator"),
        ("تسک منیجر", "task manager"),
        ("task manager", "task manager"),
        ("taskmgr", "task manager"),
        ("خط فرمان", "command prompt"),
        ("command prompt", "command prompt"),
        ("cmd", "command prompt"),
        ("پاورشل", "powershell"),
        ("powershell", "powershell"),
        ("رجیستری", "registry"),
        ("registry", "registry"),
        ("regedit", "registry"),
        ("کنترل پنل", "control panel"),
        ("control panel", "control panel"),
        ("فایل اکسپلورر", "file explorer"),
        ("file explorer", "file explorer"),
        ("explorer", "file explorer"),
        ("مای کامپیوتر", "my computer"),
        ("my computer", "my computer"),
        ("this pc", "this pc"),
        
        ("مدیریت دیسک", "disk management"),
        ("disk management", "disk management"),
        ("diskmgmt", "disk management"),
        ("مدیریت کامپیوتر", "computer management"),
        ("computer management", "computer management"),
        ("سرویس‌ها", "services"),
        ("services", "services"),
        ("مدیر دستگاه", "device manager"),
        ("دیوایس منیجر", "device manager"),
        ("devmgmt", "device manager"),
        ("device manager", "device manager"),
        ("اج", "edge"),
        ("edge", "edge"),
        ("مرورگر", "edge"),
        
        ("صفحه نمایش", "display"),
        ("display", "display"),
        ("صدا", "sound"),
        ("sound", "sound"),
        ("شبکه", "network"),
        ("network", "network"),
        ("وای فای", "wifi"),
        ("wifi", "wifi"),
        ("بلوتوث", "bluetooth"),
        ("bluetooth", "bluetooth"),
        ("باتری", "battery"),
        ("battery", "battery"),
        
        ("تنظیمات", "settings"),
        ("settings", "settings"),
    ]
    
    for keyword, mapped_value in keywords:
        if keyword in user_input_lower:
            return mapped_value
    
    for keyword, mapped_value in keywords:
        if keyword in ai_response_lower:
            return mapped_value
    
    return None

def map_to_command(keyword):
    if not keyword:
        return None
        
    command_map = {
        "notepad": "notepad",
        "calculator": "calc",
        "task manager": "taskmgr",
        "command prompt": "cmd",
        "powershell": "powershell",
        "registry": "regedit",
        "control panel": "control",
        "file explorer": "explorer",
        "my computer": "explorer",
        "this pc": "explorer",
        "edge":"microsoft-edge",
        
        "disk management": "start diskmgmt.msc",
        "computer management": "start compmgmt.msc",
        "services": "start services.msc",
        "device manager": "start devmgmt.msc",
        
        "settings": "start ms-settings:",
        "display": "start ms-settings:display",
        "sound": "start ms-settings:sound",
        "network": "start ms-settings:network",
        "wifi": "start ms-settings:network-wifi",
        "bluetooth": "start ms-settings:bluetooth",
        "battery": "start ms-settings:batterysaver",
    }
    
    return command_map.get(keyword.lower())

def execute_command(command):
    if command:
        try:
            subprocess.run(command, shell=True, check=True)
            logger.info(f"اجرای موفق: {command}")
            print(f"✅ دستور اجرا شد: {command}")
        except subprocess.CalledProcessError as e:
            logger.error(f"خطا در اجرای دستور: {e}")
            print(f"❌ خطا در اجرای دستور: {e}")
    else:
        print("❌ دستور معتبر پیدا نشد.")

def process_voice_input():
    """پردازش ورودی صوتی با گوش دادن مداوم"""
    if not load_whisper_model():
        print("❌ امکان بارگذاری مدل وجود ندارد!")
        return
    
    print("🎤 حالت دستیار صوتی فعال شد!")
    print("📝 برای بیدار کردن دستیار 'دستیار' بگویید")
    print("📝 برای خروج 'تمام' بگویید")
    print("-" * 50)
    
    while True:
        audio_data, sample_rate = listen_for_wake_word()
        
        if audio_data is None:
            continue
        
        if check_for_stop_word(audio_data, sample_rate):
            print("🛑 دستیار صوتی متوقف شد!")
            break
        
        if check_for_wake_word(audio_data, sample_rate):
            play_welcome_audio()
            
            command_audio, command_sample_rate = record_user_command()
            
            if command_audio is None:
                print("❌ خطا در ضبط دستور!")
                continue
            
            transcription = transcribe_audio(command_audio, command_sample_rate)
            
            if not transcription:
                print("❌ دستور شناسایی نشد!")
                continue
            
            process_command(transcription)
            
            print("\n" + "-"*50)
            print("👂 دوباره گوش می‌دهم برای 'دستیار'...")
            print("-" * 50)

def process_command(user_prompt):
    """پردازش دستور متنی"""
    print("🔄 در حال پردازش...")
    ai_response = gemma3(user_prompt)
    
    if ai_response:
        print(f"🤖 AI Response: {ai_response[:100]}...")
        save_to_db(user_prompt, ai_response)
        
        best_keyword = find_best_keyword_match(user_prompt, ai_response)
        print(f"🔍 کلمه کلیدی یافت شده: {best_keyword}")
        
        command = map_to_command(best_keyword) if best_keyword else None
        print(f"⚡ فرمان: {command}")
        
        execute_command(command)
    else:
        print("❌ خطا در دریافت پاسخ از مدل.")

def process_text_input(user_prompt=None):
    """پردازش ورودی متنی"""
    if not user_prompt:
        user_prompt = input("💬 دستور خود را وارد کنید: ")
    
    process_command(user_prompt)

def show_menu():
    """نمایش منوی انتخاب"""
    print("\n" + "="*50)
    print("🎯 دستیار هوشمند صوتی")
    print("="*50)
    print("1️⃣  ورودی متنی")
    print("2️⃣  ورودی صوتی")
    print("3️⃣  خروج")
    print("="*50)

def main():
    init_db()
    
    while True:
        show_menu()
        
        try:
            choice = input("⚡ انتخاب کنید (1/2/3): ").strip()
            
            if choice == "1":
                print("\n📝 حالت ورودی متنی انتخاب شد")
                process_text_input()
                
            elif choice == "2":
                print("\n🎤 حالت دستیار صوتی انتخاب شد")
                process_voice_input()
                
            elif choice == "3":
                print("👋 خروج از برنامه...")
                break
                
            else:
                print("❌ انتخاب نامعتبر! لطفا 1، 2 یا 3 را انتخاب کنید.")
                
        except KeyboardInterrupt:
            print("\n👋 برنامه متوقف شد.")
            break
        except Exception as e:
            print(f"❌ خطای غیرمنتظره: {e}")
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()
