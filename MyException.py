# coding: utf-8
# ---
# @File: MyException.py
# @description: 自定义异常类
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 3月6, 2019
# ---


class MyException(Exception):
    """
    继承自基类 Exception
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

