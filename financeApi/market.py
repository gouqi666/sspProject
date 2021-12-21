import flask, json
from flask import request
import pandas as pd
import numpy as np
from app import *
from getnews import *
'''
flask: web框架，通过flask提供的装饰器@server.route()将普通函数转换为服务
登录接口，需要传url、username、passwd
'''
# 创建一个服务，把当前这个python文件当做一个服务
server = flask.Flask(__name__)
# server.config['JSON_AS_ASCII'] = False
# @server.route()可以将普通函数转变为服务 登录接口的路径、请求方式
@server.route('/index',methods=['get','post'])
def index():
    auth('13739188902', 'ZNnb20160801')
    intent=request.values.get('intent','trend')
    if intent=='trend':
        asset=request.values.get('asset','hs300')
        if asset=='market' or asset=='大盘':
            asset='沪深300'
        date=request.values.get('date','today')
        name=pd.read_excel('name.xlsx')
        if asset not in name['name'].tolist():
            messages='对不起，我正在学习当中。'
            resu={'code':111,'data':messages}
            return json.dumps(resu, ensure_ascii=False)
        name.set_index('name',inplace=True)
        type=name.loc[asset,'type']
        code=name.loc[asset,'code']
        if type=='指数':
            resu=indexAnalyse(asset,date,code)
        if type=='股票':
            resu=stockAnalyse(asset,date,code)
        if type=='基金':
            resu=fundAnalyse(asset,date,code)
        return json.dumps(resu, ensure_ascii=False)
    elif intent=='detail':
        asset=request.values.get('asset')
        info=request.values.get('info','收益率')
        date_range=request.values.get('date','最近一周')
        name=pd.read_excel('name.xlsx')
        if asset not in name['name'].tolist():
            messages='对不起，我正在学习当中。'
            resu={'code':111,'data':messages}
            return json.dumps(resu, ensure_ascii=False)
        name.set_index('name',inplace=True)
        type=name.loc[asset,'type']
        code=name.loc[asset,'code']
        if type=='指数':
            resu=indexAnalyse(asset,'recent',code)
        if type=='股票':
            resu=stockDetail(asset,date_range,info,code)
        if type=='基金':
            resu=fundDetail(asset,date_range,info,code)
        return json.dumps(resu, ensure_ascii=False)        
    elif intent=='news':
        tList=getNewsTitle(3)
        messages='为你播报最新财经相关新闻；'
        i=1
        for t in tList:
            messages+=str(i)+'. '+t+';'
            i+=1
        return json.dumps({'code':111,'data':messages}, ensure_ascii=False)      
if __name__ == '__main__':
    
    server.run(port=8888, host='0.0.0.0')# 指定端口、host,0.0.0.0代表不管几个网卡，任何ip都可以访问