#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/3/17 17:51
# @File  : wep_api.py
# @Author: 
# @Desc  :
import logging
import os
from flask import Flask, request, jsonify, abort
from transformers import AutoModel, AutoTokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)
# 日志保存的路径，保存到当前目录下的logs文件夹中
log_path = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(log_path):
    os.makedirs(log_path)
logfile = os.path.join(log_path, "api.log")
# 日志的格式
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(logfile, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

model_dir = "/home/wac/johnson/project/model/THUDM/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
model = model.eval()

@app.route("/api/chat", methods=['POST'])
def chat():
    """
    Args: 基于aspect的情感分析，给定实体，判断实体在句子中的情感
    """
    jsonres = request.get_json()
    # 可以预测多条数据
    data = jsonres.get('data', None)
    if not data:
        return jsonify({"code": 400, "msg": "data不能为空"}), 400
    logging.info(f"数据分别是: {data}")
    input = data.get('text', '')
    history = data.get('history', [])
    response, history = model.chat(tokenizer, input, history=history)
    result = {"response": response}
    logging.info(f"返回的结果是: {result}")
    return jsonify(result)

@app.route("/ping", methods=['GET', 'POST'])
def ping():
    """
    测试
    :return:
    :rtype:
    """
    return jsonify("Pong")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7087, debug=False, threaded=True)