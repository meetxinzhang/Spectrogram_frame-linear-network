# coding: utf-8
# ---
# @File: socket.py
# @description: python http demo. 注意，http协议是客户端请求协议，就是说服务器是不能主动发送信息给客户端的；
#      要实现主动发送，建议采用套接字 Socket 进行通信。
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 1月08, 2020
# ---


from http.server import BaseHTTPRequestHandler, HTTPServer
import json
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8800


class MyRequestHandler(BaseHTTPRequestHandler):
    """
    该类负责处理客户端s的 Request, 继承自 BaseHTTPRequestHandler
    """
    def get_inf(self):
        """
        处理 get 请求
        Add a new todo
        """
        print(self.requestline)

        data = {
            'result_code': '1',
            'result_desc': 'Success',
            'timestamp': '',
            'data': {'message_id': '25d55ad283aa400af464c76d713c07ad'}
        }
        # 设置 response 格式之类的
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write("Hello from Server!" + json.dumps(data).encode())

    def post_inf(self):
        """
        处理 post 请求
        Add a new todo
        """
        print(self.headers)
        print(self.command)

        # HTTP允许传输任意类型的数据对象，正在传输的类型由 Content-Type 加以标记。
        datas = self.rfile.read(int(self.headers['content-length']))
        print(datas.decode())

        pass


class MyCustomHTTPServer(HTTPServer):
    """
    与每个客户端通信的子进程类, 继承自 HTTPServer
    在后期优化的时候，应该放在线程里面去执行；
    """
    def __init__(self, host, port):
        server_address = (host, port)
        HTTPServer.__init__(self, server_address, MyRequestHandler)


def run_server(port):
    """
    监听客户端连接, 该方法应该在服务器开启时便调用，并且放在守护线程里保持运行；
    可以用一个 list.append(server) 记录所有的客户端连接信息，便于后期维护；
    :param port:
    """
    try:
        # 当收到连接请求时，为该客户端开启一个独立的服务器进程，这里应该是三次握手的过程
        server = MyCustomHTTPServer(DEFAULT_HOST, port)
        print("Custom HTTP server started on port: " + port)
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server interrupted and is shutting down...")
        server.socket.close()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    while True:
        run_server(DEFAULT_PORT)

pass
