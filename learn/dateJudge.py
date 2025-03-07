#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/5 12:21
# @File ：dateJudge.py
# @Author : 胜天半子
# IDE : PyCharm
# @CSDN : https://blog.csdn.net/HG0724?type=blog
# @GitHub : https://github.com/haozhenHG
import hashlib
import time

import requests
from datetime import date
import json

# 判断当天日期
def Judge_Today(url,headers):
    print(url)
    try:
        # 发送请求
        response = requests.get(url,headers=headers)
        response.raise_for_status()  # 如果请求失败，抛出异常

        # 解析JSON数据
        data = response.json()
        print(data)
        holiday_type = data['type']['type']
        # "type": enum(0, 1, 2, 3), // 节假日类型，分别表示 工作日、周末、节日、调休。
        # 判断是否是工作日
        if holiday_type == 0:
            print(f"{today} 是工作日,且是{data['type']['name']}")
        elif holiday_type == 1:
            print(f"{today} 是周末,且是{data['type']['name']}")
        elif holiday_type == 2:
            print(f"{today} 是节假日,且是{data['type']['name']}")
        elif holiday_type == 3:
            print(f"{today} 是调休日。")
        return True
    except requests.RequestException as e:
        print(f"请求接口时发生错误: {e}")
    except KeyError:
        print("解析接口返回数据时发生错误，数据格式可能不正确。")

def send_notice(mobiles,params):
    '''

    Parameters
    ----------
    mobiles   发送业务的手机号
    params    短信内容

    Returns
    -------

    '''
    url = 'https://api.netease.im/sms/sendtemplate.action'
    """
    AppKey	网易云信分配的账号，请替换你在管理后台应用下申请的Appkey
    Nonce	随机数（最大长度128个字符）
    CurTime	当前UTC时间戳，从1970年1月1日0点0 分0 秒开始到现在的秒数(String)
    CheckSum	SHA1(AppSecret + Nonce + CurTime)，三个参数拼接的字符串，进行SHA1哈希计算，转化成16进制字符(String，小写)
    """
    AppKey = "05557c8cdc2f388c487ea1daebcb06c1"
    # 生成128个长度以内的随机字符串
    nonce = hashlib.new('sha512', str(time.time()).encode("utf-8")).hexdigest()
    # 获取当前时间戳
    curtime = str(int(time.time()))
    # 网易云信的 App Secret
    AppSecret = "66b224bd290b"   # 开发文档
    # 根据要求进行SHA1哈希计算
    check_sum = hashlib.sha1((AppSecret + nonce + curtime).encode("utf-8")).hexdigest()

    header = {
        # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "AppKey": AppKey,
        "Nonce": nonce,
        "CurTime": curtime,
        "CheckSum": check_sum
    }

    data = {
        'mobiles': json.dumps(mobiles),  # 手机号
        "templateid": 202503053462448079,  # 模板id
        "params":json.dumps(params),
    }

    resp = requests.post(url, data=data, headers=header)

    print("Response:", resp.content)

if __name__ == '__main__':
    # today = date.today().strftime('%Y-%m-%d')
    # print(today)
    # 获取今天的日期
    today = date.today().strftime('%Y-%#m-%#d')
    print(today)
    # print(today)

    # 构造请求URL   接口默认使用不补零的时间日期
    url = f"https://timor.tech/api/holiday/info/{today}"
    # https://timor.tech/api/holiday/info/2025-4-4
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    }
    if Judge_Today(url,headers):
        mobiles = ["17349868689", "18631906548"] # 要发送的手机号
        params = ["请关注公众号","胜天半月子"] # 模板中的变量
        send_notice(mobiles,params)