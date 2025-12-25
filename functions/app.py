import cv2
import os
import uuid
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from serverless_wsgi import handle_request  # 关键导入

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
# 使用临时目录，适用于Serverless环境
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """检查文件扩展名是否合法"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def image_to_base64(img_array):
    """将OpenCV图像数组转换为Base64字符串"""
    _, buffer = cv2.imencode('.jpg', img_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode('utf-8')

def detect_faces(image_path):
    """
    核心人脸检测函数
    返回: (marked_base64, blurred_base64, face_count, faces_list)
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    if img is None:
        return None, None, 0, []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    img_marked = img.copy()
    img_blurred = img.copy()
    faces_list = []

    for i, (x, y, w, h) in enumerate(faces):
        # 在标记图上绘制框
        cv2.rectangle(img_marked, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # 在打码图上模糊人脸
        face_region = img_blurred[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (315, 315), 0)
        img_blurred[y:y+h, x:x+w] = blurred_face

        faces_list.append({
            'id': i+1,
            'x': int(x), 'y': int(y),
            'width': int(w), 'height': int(h)
        })

    # 将结果图像转为Base64字符串，而非保存文件
    marked_base64 = image_to_base64(img_marked)
    blurred_base64 = image_to_base64(img_blurred)

    return marked_base64, blurred_base64, len(faces), faces_list

# ========== 路由定义 ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有选择文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '未选择文件'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': '不支持的文件格式'}), 400

    try:
        # 保存到临时位置进行处理
        temp_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}.jpg")
        file.save(temp_filename)

        # 进行人脸检测（现在返回两个结果的Base64字符串）
        marked_base64, blurred_base64, face_count, faces_info = detect_faces(temp_filename)

        # 清理临时文件
        os.remove(temp_filename)

        if marked_base64 is None:
            return jsonify({'success': False, 'error': '无法处理图片文件'}), 500

        return jsonify({
            'success': True,
            'face_count': face_count,
            'faces': faces_info,
            'marked_image': marked_base64,    # 前端需用 <img src="data:image/jpeg;base64,{marked_image}"> 显示
            'blurred_image': blurred_base64,  # 同上
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        return jsonify({'success': False, 'error': f'服务器处理出错: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Face Detection API'})

# ========== Netlify Functions 专用入口 ==========
def handler(event, context):
    return handle_request(app, event, context)