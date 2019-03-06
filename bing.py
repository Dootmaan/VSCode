import re #导入正则表达式模块
import requests #python HTTP客户端 编写爬虫和测试服务器经常用到的模块
import random #随机生成一个数，范围[0,1]
import time
from PIL import Image
 
#定义函数方法
def spiderPic(html):
    print('正在查找壁纸,下载中，请稍后......')
    for addr in re.findall('url: "/az/hprichbg/rb/([a-zA-Z0-9_-]*)_1920x1080.jpg"',html,re.S):     #查找URL
        actaddr="https://cn.bing.com/az/hprichbg/rb/"+addr+"_1920x1080.jpg"
        print('正在下载壁纸：'+addr[0:30]+'...')  #爬取的地址长度超过30时，用'...'代替后面的内容
 
        try:
            pics = requests.get(actaddr,timeout=10)  #请求URL时间（最大10秒）
        except requests.exceptions.ConnectionError:
            print('您当前请求的URL地址出现错误')
            continue
 
        fq = open('D:\\Photos\\壁纸\\' + (str(addr)+'.jpg'),'wb')     #下载图片，并保存和命名
        fq.write(pics.content)
        fq.close()

        im = Image.open('D:\\Photos\\壁纸\\' + (str(addr)+'.jpg'))
        im.show()
 
#python的主方法
if __name__ == '__main__':
    print("现在时间："+time.asctime(time.localtime(time.time())) + ",为您下载今日Bing壁纸")
    result = requests.get('https://cn.bing.com/')
    # result = requests.get('https://cn.bing.com/?FORM=BEHPTB&ensearch=1')
 
#调用函数
spiderPic(result.text)
