#! /usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import re
import os

# os.system('chcp 65001')

if __name__ == "__main__":
    # ,"233009","001707""162411""540006","000849",
    code_list = ["110022","161725","320007","161028","519005","001410",]
    for x in  code_list:
        # http://fund.eastmoney.com/161725.html?spm=search
        url = "http://fund.eastmoney.com/%s.html?spm=search" % (x)
        rsp = requests.get(url)
        if rsp.status_code != requests.codes.ok: continue
        group = re.search(r'id="gz_gsz">([.\d]+).+?class="gz(\w+)".+?id="gz_gszze">([-+.\d]+).+?id="gz_gszzl">(.+?%)', rsp.content)
        title = re.search(r'<title>(.+?)\(', rsp.content).group(1).decode("utf8")
        if not group:
            print url, "is not match"
            continue
        wave = None
        if group.group(2) == "up":
            wave = "↑"
        else:
            wave = "↓"
        wave = wave.decode("utf8")
        print x, title, group.group(1), wave, group.group(3), group.group(4)

