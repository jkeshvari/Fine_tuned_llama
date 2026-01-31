import streamlit as st
import requests

# تعریف مدل و API Key
MODEL_NAME = "Qwen/QwQ-32B-Preview"  # نام مدل Hugging Face
API_KEY = st.secrets["hf_api_key"]  # دریافت API Key از secrets

# تابع برای ارسال درخواست به مدل
def chat_with_model(message):
    url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    # payload برای ارسال به مدل
    payload = {
        "inputs": message,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 1024,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "top_p": 0.95
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # بررسی خطاهای HTTP
        return response.json()  # بازگشت پاسخ مدل
    except requests.exceptions.RequestException as e:
        return f"Error during chat: {str(e)}"

# رابط کاربری Streamlit
st.title("چت با مدل هوش مصنوعی Hugging Face")

# ورودی کاربر
user_input = st.text_input("پیام خود را وارد کنید:")

# تاریخچه چت
if 'history' not in st.session_state:
    st.session_state.history = []

# دکمه ارسال
if st.button("ارسال"):
    if user_input:
        # ارسال پیام به مدل
        response = chat_with_model(user_input)

        # بررسی نوع پاسخ
        if isinstance(response, list) and len(response) > 0:
            ai_message = response[0].get('generated_text', "پاسخی از مدل دریافت نشد.")
        else:
            ai_message = "پاسخی از مدل دریافت نشد."

        # ذخیره تاریخچه
        st.session_state.history.append((user_input, ai_message))
        
        # نمایش تاریخچه چت
        st.subheader("تاریخچه چت:")
        for user_msg, ai_msg in st.session_state.history:
            st.write(f"**شما:** {user_msg}")
            st.write(f"**مدل:** {ai_msg}")
    else:
        st.warning("لطفاً یک پیام وارد کنید.")