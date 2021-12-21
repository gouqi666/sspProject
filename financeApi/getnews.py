import urllib.request
from bs4 import BeautifulSoup
def getUrl (url):
    #定义一个headers,存储刚才复制下来的报头,模拟成浏览器
   headers = ('User-Agent',
               "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36")
   opener = urllib.request.build_opener()
   opener.addheaders = [headers]
   # 将opener安装为全局
   urllib.request.install_opener(opener)
   html = urllib.request.urlopen(url).read().decode('utf-8', 'ignore')
   # print(html)
   bs = BeautifulSoup(html,'lxml')
   # 用beautifulsoup的select,找到所有的<a>标签
   links = bs.select('ul.xh-list.xh-list-f12>li>a')
   return links

import  sys
def getNewsTitle(num):

    url = 'https://finance.sina.com.cn/stock/'
    # 获取对应网页的链接地址
    linklist = getUrl(url)
# 定义一个列表texts存储文章的标题
    texts = []
    # 定义一个列表links存储文章的链接
    links = []
    # 遍历linkllist,存储标题和链接
    for link in linklist:
        texts.append(link.text.strip())
        links.append(link.get('href'))
    #    通过zip,将信息输出到控制台
    i=0
    titleList=[]
    for text, link in zip(texts, links):
        if i>num:
            break
        text = text.strip().replace("原        \n        ", "")
        text = text.strip().replace("转        \n        ", "")
        titleList.append(text)
        i+=1
    return titleList
