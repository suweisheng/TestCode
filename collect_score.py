# -*- encoding: utf-8 -*-

import xlsxwriter
import re
import os
import json
import sys

# windows平台cmd control默认'gbk'编码, 'utf8'格式的字符串打印出来会报IOError: [Errno 0] Error, 解决方法：1.改window cmd编码格式; 2."xxxx".decode('utf-8').encode('gbk')
reload(sys)
sys.setdefaultencoding('utf-8')

# os.system('chcp 65001')

workbook = xlsxwriter.Workbook('S101_Score.xlsx',) # 新建工作簿excel表
worksheet = workbook.add_worksheet('ScoreInfo') # 新建工作表

 # 设置表头
headings = ['urs','uuid','name','level','role_id','career_id',
            'BaseScore','EquipScore','PetScore','SpellScore',
            'AppearaScore','GemScore','WeaponScore','LegendCard',
            'max_hp','phy_att_min','phy_att_max','mag_att_min',
            'mag_att_max','phy_def','mag_def']

worksheet.write_row('A1', headings)

all_data = []
with open("Z:/zc/RoleScoreInfo/s101_RoleInfo.txt", 'r') as f:
    for data in f.readlines():
        all_data.append(data.replace("\n", "").split(','))

count = 2
# length = len(headings)
for row in all_data:
    # if len(row) != length:
    #     print "error", row
    #     break
    worksheet.write_row('A'+str(count), row)
    count = count + 1

workbook.close() # 关闭工作簿，必须