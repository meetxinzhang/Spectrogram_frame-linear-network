# coding: utf-8
import urllib.request
import urllib.error
import urllib.parse

# 伪装成Chrome浏览器
headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
           # 'Accept_Encoding': 'gzip, deflate, br',
           'Accept-Language': 'zh-CN,zh;q=0.9',
           'Cache-Control': 'max-age=0',
           'Connection': 'keep-alive',
           'Cookie': '_utmz=47666734.1540381830.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utma=47666734.732259021.1540381830.1542852869.1543043215.3; login=bWVldGRldmluLnpoQGdtYWlsLmNvbX4jfjlEaWp3RGd0WkNGNWl1VWU%3D; PHPSESSID=d6o3rccff8ac3u1p7n17cp1n45',
           'DNT': '1',
           'Host': 'www.xeno-canto.org',
           'Upgrade-Insecure-Requests': '1',
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36'}

value = {'source': 'index_nav',
         'form_password': 'Cin1984db',
         'form_email': 'zxin94264@vip.qq.com'
         }
try:
    data = urllib.parse.urlencode(value).encode('utf8')
    response = urllib.request.Request(
        'https://www.douban.com/login', data=data, headers=headers)
    html = urllib.request.urlopen(response)
    result = html.read().decode('utf8')
    print(result)
except urllib.error.URLError as e:
    if hasattr(e, 'reason'):
        print('错误原因是' + str(e.reason))
except urllib.error.HTTPError as e:
    if hasattr(e, 'code'):
        print('错误编码是' + str(e.code))
    else:
        print('请求成功通过。')
