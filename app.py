# app.py - Flask后端主程序
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import os
import uuid
from werkzeug.utils import secure_filename
from datetime import datetime

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # 可自行修改

# 配置文件上传
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """检查文件扩展名是否合法"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_faces(image_path):
    """
    核心人脸检测函数
    返回: (marked_path, blurred_path, face_count, faces_list)
    """
    # 加载人脸检测器（Haar级联分类器）
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        return None, None, 0, []
    
    # 转换为灰度图（提高检测速度）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 执行人脸检测
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    # 读取原始图片用于绘制两种结果
    img_marked = img.copy()  # 用于画框的图
    img_blurred = img.copy() # 用于打码的图
    
    faces_list = []
    
    for i, (x, y, w, h) in enumerate(faces):
        # ========== 1. 在标记图上绘制绿色框和标签 ==========
        # 绘制矩形框 (BGR颜色格式: 0,255,0 = 绿色)
        color = (0, 255, 0)
        thickness = 3
        cv2.rectangle(img_marked, (x, y), (x+w, y+h), color, thickness)
        
        # 添加人脸编号标签
        label = f"Face {i+1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        label_thickness = 2
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)
        
        # 标签背景
        cv2.rectangle(img_marked, 
                     (x, y - label_height - baseline - 5), 
                     (x + label_width, y - 5), 
                     color, 
                     -1)  # -1表示填充
        
        # 标签文字
        cv2.putText(img_marked, label, 
                   (x, y - baseline - 5), 
                   font, font_scale, (255, 255, 255), label_thickness)
        
        # ========== 2. 在打码图上应用高斯模糊 ==========
        # 提取人脸区域
        face_region = img_blurred[y:y+h, x:x+w]
        # 应用高斯模糊，(15, 15)是模糊核大小，必须是正奇数
        blurred_face = cv2.GaussianBlur(face_region, (315, 315), 0)
        img_blurred[y:y+h, x:x+w] = blurred_face
        
        # 保存人脸坐标信息
        faces_list.append({
            'id': i+1,
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h)
        })
    
    # 生成并保存两张结果图
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    original_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. 标记图（带绿框）
    marked_filename = f"marked_{original_name}_{timestamp}.jpg"
    marked_path = os.path.join(app.config['UPLOAD_FOLDER'], marked_filename)
    cv2.imwrite(marked_path, img_marked, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # 2. 打码图（高斯模糊）
    blurred_filename = f"blurred_{original_name}_{timestamp}.jpg"
    blurred_path = os.path.join(app.config['UPLOAD_FOLDER'], blurred_filename)
    cv2.imwrite(blurred_path, img_blurred, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # 返回两个结果路径
    return marked_filename, blurred_filename, len(faces), faces_list

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理图片上传和人脸检测"""
    # 检查是否有文件部分
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有选择文件'}), 400
    
    file = request.files['file']
    
    # 检查是否选择了文件
    if file.filename == '':
        return jsonify({'success': False, 'error': '未选择文件'}), 400
    
    # 检查文件格式
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': '不支持的文件格式。请上传图片文件（PNG, JPG, JPEG, GIF, BMP）'}), 400
    
    try:
        # 生成安全的文件名
        original_filename = secure_filename(file.filename)
        file_extension = os.path.splitext(original_filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # 保存上传的文件
        file.save(upload_path)
        
        # 进行人脸检测（现在返回两个结果）
        marked_filename, blurred_filename, face_count, faces_info = detect_faces(upload_path)
        
        if marked_filename is None:
            return jsonify({'success': False, 'error': '无法处理图片文件'}), 500
        
        # 返回结果（包含两张结果图的URL）
        return jsonify({
            'success': True,
            'face_count': face_count,
            'faces': faces_info,
            'original_url': f'/static/uploads/{unique_filename}',
            'marked_url': f'/static/uploads/{marked_filename}',    # 带框标记图
            'blurred_url': f'/static/uploads/{blurred_filename}',  # 人脸打码图
            'original_name': original_filename,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        # 记录错误日志
        print(f"处理图片时出错: {str(e)}")
        return jsonify({'success': False, 'error': f'服务器处理出错: {str(e)}'}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """提供上传的图片文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({'status': 'healthy', 'service': 'Face Detection API'})
# ========== 诊断代码（粘贴在 if __name__ == '__main__': 之前）==========
def diagnose_upload():
    """临时诊断路由：打印所有路径信息"""
    @app.route('/diagnose', methods=['POST'])
    def diagnose():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # 1. 打印原始信息
        print(f"[诊断] 收到文件: {file.filename}")
        original_filename = secure_filename(file.filename)
        print(f"[诊断] 安全文件名: {original_filename}")
        
        # 2. 保存上传文件
        unique_filename = f"diagnose_{uuid.uuid4().hex}.jpg"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)
        print(f"[诊断] 文件保存到: {upload_path}")
        print(f"[诊断] 文件存在: {os.path.exists(upload_path)}")
        
        # 3. 尝试处理（但不实际保存结果图，只打印路径）
        try:
            marked_filename = f"diagnose_marked_{uuid.uuid4().hex}.jpg"
            blurred_filename = f"diagnose_blurred_{uuid.uuid4().hex}.jpg"
            marked_path = os.path.join(app.config['UPLOAD_FOLDER'], marked_filename)
            blurred_path = os.path.join(app.config['UPLOAD_FOLDER'], blurred_filename)
            
            print(f"[诊断] 标记图路径: {marked_path}")
            print(f"[诊断] 打码图路径: {blurred_path}")
            
            # 4. 模拟返回的URL
            return jsonify({
                'diagnose': True,
                'original_url': f'/static/uploads/{unique_filename}',
                'marked_url': f'/static/uploads/{marked_filename}',
                'blurred_url': f'/static/uploads/{blurred_filename}',
                'static_folder': app.config['UPLOAD_FOLDER'],
                'absolute_static_path': os.path.abspath(app.config['UPLOAD_FOLDER'])
            })
        except Exception as e:
            print(f"[诊断] 处理出错: {e}")
            return jsonify({'error': str(e)}), 500
    
    return None

# 注册诊断路由（不影响主功能）
diagnose_upload()
# 启动Flask应用
if __name__ == '__main__':
    print("=" * 50)
    print("人脸识别网站启动中...")
    print("功能：人脸检测 + 人脸打码（高斯模糊）")
    print(f"访问地址: http://127.0.0.1:5000")
    print(f"上传目录: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print("=" * 50)
    
    # 启动开发服务器
    app.run(debug=True, host='0.0.0.0', port=5000)