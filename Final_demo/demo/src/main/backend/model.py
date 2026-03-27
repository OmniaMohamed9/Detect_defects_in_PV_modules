from PIL import Image

# لو موديل باي تورش تكتبي تحميل الموديل هنا
# مثال تجريبي فقط:
def load_model():
    # load your real model here
    return None

MODEL = load_model()

def predict_image(path, extra=None):
    # path: مسار الصورة المؤقتة
    # extra: dict من الحقول المرسلة (camera_model, location, notes)
    # هنا نعمل تنبؤ تجريبي — استبدليه بمنطق الموديل الحقيقي
    img = Image.open(path).convert("RGB")
    # تنبؤ وهمي:
    return {
        "label": "ok",
        "confidence": 0.98,
        "text": "No defect detected"
    }
