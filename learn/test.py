#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/1/3 20:48
# @File ：test.py
# @Author : 胜天半子
# IDE : PyCharm
# @CSDN : https://blog.csdn.net/HG0724?type=blog
# @GitHub : https://github.com/haozhenHG
class MyClass:
    def __init__(self):
        self.name = "Kimi"
        self.age = 30
    def test(self):
        self.name += 'Kimi'
obj = MyClass()
print('obj.__dict__ : ', obj.__dict__)
print('obj.__dir__() : ', obj.__dir__())
print('dir(obj) : ', dir(obj))