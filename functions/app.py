from flask import Flask
from serverless_wsgi import handle_request # 关键导入

# 创建Flask应用实例
app = Flask(__name__)

# 这是你原来的路由，这里以首页为例
@app.route('/')
def hello():
    return 'Hello World from Netlify Functions!'

# 这是Netlify函数的标准入口，必须命名为 handler
def handler(event, context):
    return handle_request(app, event, context)