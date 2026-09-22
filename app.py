import streamlit as st
from PIL import Image
import numpy as np
import os
import urllib.request

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.resnet50 import preprocess_input
except ImportError:
    import keras
    from keras.models import load_model
    from keras.applications.resnet50 import preprocess_input

# ===============================
# 1. إعدادات الصفحة والواجهة
# ===============================
st.set_page_config(page_title="CorroVision AI | الفحص الذكي", page_icon="🔎", layout="centered")

custom_css = f"""
<style>
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
    div[data-testid="stDecoration"] {{display: none !important;}}
    div[data-testid="stStatusWidget"] {{display: none !important;}}
    .stAppToolbar {{display: none !important;}}
    
    div[class*="viewerBadge"] {{display: none !important;}}
    div[data-testid="stEmbedFooter"] {{display: none !important;}}
    footer[data-testid="stFooter"] {{display: none !important;}}

.stApp {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: #222222;
}}


html, body, .stApp, p, h2, h3, h4, h5, h6, span, label, div {{
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}

.stApp h1 {{
    direction: ltr !important;
    text-align: left !important;
}}


.stFileUploader label, .stNumberInput label {{
    font-weight: bold;
    color: #ffffff;
}}

.st-emotion-cache-1abbcj6 {{
background-color: #B20600;
}}

.st-emotion-cache-11jd9ak {{
background-color: #B20600;
}}

.st-emotion-cache-10ph27a {{
color: #222222;
}}

.st-emotion-cache-v6r2pr {{
background-color: #222222;
color: #EEEEEE;
padding: 12px;
border-radius: 10px;
}}

.st-emotion-cache-v6r2pr:hover {{
background-color: #B20600;
color: #EEEEEE;
}}

.st-emotion-cache-v6r2pr[data-selected] {{
    background-color: #BE3030;
    color: #EEEEEE;
}}

div[data-baseweb="input"] {{
    direction: ltr !important; 
    text-align: left !important;
}}

.stAlert {{
    direction: rtl !important;
    text-align: right !important;
    border-radius: 10px !important;
}}

.stAlertContainer {{
    color: #EEEEEE;

}}

.stButton > button {{
    position: relative;
    bottom: 20px;
    font-weight: bold !important;
    border-radius: 8px !important;
    background-color: #B20600 !important;
    color: white !important;
    border: none !important;
    padding: 0.6rem 1rem !important;
}}

.stButton > button:hover {{
    background-color: #972828 !important;
    color: white !important;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("CorroVision AI")
st.subheader("نظام دعم القرار الذكي للكشف عن التآكل")
st.markdown("---")

# ===============================
# 2. تحميل النموذج الحقيقي
# ===============================
MODEL_PATH = "best_ResNet50.keras"

MODEL_URL = "https://github.com/Samah-Mahdi/CorroVision-AI/releases/download/v1.0.0/best_ResNet50.keras"

@st.cache_resource
def load_resnet_model(path, url):
    if not os.path.exists(path):
        with st.spinner("جاري تنزيل نموذج الذكاء الاصطناعي لأول مرة (قد يستغرق دقيقة)..."):
            urllib.request.urlretrieve(url, path)
    
    return load_model(path)

model = load_resnet_model(MODEL_PATH, MODEL_URL)

# ===============================
# 3. دالة معالجة الصورة والتنبؤ الحقيقي
# ===============================
def predict_corrosion(image, model_obj):
    img = image.resize((224, 224)).convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    raw_pred = model_obj.predict(img_array)
    pred_arr = np.array(raw_pred)
    
    if pred_arr.ndim == 2 and pred_arr.shape[1] == 1:
        prob = float(pred_arr[0][0])
    elif pred_arr.ndim == 2 and pred_arr.shape[1] == 2:
        prob = float(pred_arr[0][1])
    else:
        prob = float(pred_arr.squeeze())
        
    threshold = 0.35  
    has_corrosion = prob >= threshold
    
    if has_corrosion:
        confidence = round(prob * 100, 2)
    else:
        confidence = round((1 - prob) * 100, 2)
        
    return has_corrosion, confidence

# ===============================
# 4. مدخلات البيانات (صورة + ضغط)
# ===============================
col1, col2 = st.columns(2)

with col1:
    st.markdown("**1. البيانات البصرية (الفحص)**")
    tab1, tab2 = st.tabs(["📂 رفع صورة من الجهاز", "📸 التقاط بالكاميرا"])

with tab1:
    uploaded_file = st.file_uploader("اختر صورة للفحص...", type=["jpg", "jpeg", "png"])
    
with tab2:
    camera_file = st.camera_input("التقاط صورة لأنابيب النفط مباشرة")
    
image_to_process = uploaded_file or camera_file
with col2:
    st.markdown("**2. البيانات التشغيلية (السياق)**")
    pressure = st.number_input("أدخل ضغط الأنبوب الحالي (Bar)", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
    pressure_threshold = 50.0  # الحد الأقصى للضغط الآمن

st.markdown("---")

# ===============================
# 5. تنفيذ التحليل واتخاذ القرار
# ===============================
if st.button("🚀 بدء التحليل الذكي", use_container_width=True):
    if image_to_process is None:
        st.error("الرجاء رفع صورة الأنبوب أولاً لإتمام العملية.")
    elif model is None:
        st.error(f"لم يتم العثور على ملف النموذج في المسار المحدد: {MODEL_PATH}")
        st.info("تأكد من صحة مسار الملف!")
    else:
        with st.spinner('جاري تحليل الصورة بواسطة النموذج ومقاطعة النتائج مع بيانات الضغط...'):
            image = Image.open(image_to_process)
            
            # الحصول على النتائج الحقيقية من النموذج
            has_corrosion, confidence = predict_corrosion(image, model)
            
            st.markdown("### 📊 تقرير الفحص الفوري")
            
            res_col1, res_col2 = st.columns([1, 1.5])
            
            with res_col1:
                st.image(image, caption="الصورة المرفوعة", use_container_width=True)
            
            with res_col2:
                st.metric(label="دقة فحص النموذج (ResNet-50 Confidence)", value=f"{confidence}%")
                
                # تطبيق المنطق المركب (Rule-Based Fusion)
                if not has_corrosion and pressure <= pressure_threshold:
                    st.success("✅ الحالة: آمن - الأنبوب سليم والضغط ضمن المعدل الطبيعي.")
                    st.info("💡 التوصية: الاستمرار في التشغيل الطبيعي. لا حاجة لصيانة فورية.")
                    
                elif has_corrosion and pressure <= pressure_threshold:
                    st.warning("⚠️ الحالة: تنبيه - تم اكتشاف تآكل، لكن الضغط مستقر.")
                    st.info("💡 التوصية: جدولة صيانة وقائية في أقرب فرصة. مراقبة الضغط باستمرار.")
                    
                elif has_corrosion and pressure > pressure_threshold:
                    st.error(f"🚨 الحالة: خطر حرج - تآكل مؤكد مع ضغط مرتفع ({pressure} Bar).")
                    st.error("💡 التوصية التشغيلية: خطر تسرب وشيك! يوصى بخفض الضغط فوراً وتوجيه فريق التدخل السريع.")
                    
                elif not has_corrosion and pressure > pressure_threshold:
                    st.warning("⚠️ الحالة: تنبيه تشغيلي - لا يوجد تآكل مرئي، لكن الضغط مرتفع جداً.")
                    st.info("💡 التوصية: فحص الصمامات وأنظمة التحكم لخفض الضغط للمستوى الآمن.")

