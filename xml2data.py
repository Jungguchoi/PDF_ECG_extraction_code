###This code is for converting XML to data(alphanumeric)

## Module import (for parsing and multiprocessing)
from multiprocessing import Process
from multiprocessing import Pool, cpu_count
import multiprocessing
import xml.etree.cElementTree as ET
from itertools import chain
from collections import defaultdict
import pandas as pd
from os import listdir
import re
import time
import os
import signal
import numpy as np
import sys

### Define functions for converting

## Parsing and collect element in XML file!

def par_elem(i):
 tree = ET.ElementTree(file = i)
 inf = []
 for elem in tree.iter():
 elem.attrib['text'] = elem.text
 y = [elem.attrib]
 inf = inf + y
 inf_tmp = inf[4:-2]

 return inf_tmp,i

def par_elem_2(i):
 tree = ET.ElementTree(file = i)
 inf = []
 for elem in tree.iter():
 x = [[elem.tag, elem.attrib, elem.text]]
 inf = inf + x
 inf_tmp_2 = inf[4:-2]

 return inf_tmp_2

#Collect element of XML file by each row!

def col_elem(inf_tmp,filename):
 all_dict_2 = []
 all_dict = []
 for n in inf_tmp:
 top_info = n['top']
 text_info = n['text']
 left_info = n['left']
 dic_tmp = {top_info : text_info}
 dic_tmp_2 = {left_info : text_info}
 dic_tmp = [dic_tmp]
 dic_tmp_2 = [dic_tmp_2]

 all_dict = all_dict + dic_tmp
 all_dict_2 = all_dict_2 + dic_tmp_2
 dic_tmp =[]
 dic_tmp_2 =[]
 super_dict = defaultdict(set)

 for d in all_dict:
  for k, v in d.items():
   super_dict[k].add(v)

   r_40 = super_dict['40']
   l_40 = list(r_40)
   r_67 = super_dict['67']
   l_67 = list(r_67)
   r_81 = super_dict['81']
   l_81 = list(r_81)
   r_94 = super_dict['94']
   l_94 = list(r_94)
   r_95 = super_dict['95']
   l_95 = list(r_95)
   r_108 = super_dict['108']
   l_108 = list(r_108)
   r_109 = super_dict['109']
   l_109 = list(r_109)
   r_122 = super_dict['122']
   l_122 = list(r_122)
   r_123 = super_dict['123']
   l_123 = list(r_123)

   return l_40,l_67, l_81, l_94,l_95,l_108,l_109,l_122,l_123

#Find suitable value in elements!

def find_values(l_40, l_67, l_81, l_94, l_95, l_108, l_109, l_122, l_123):
 #Use regex
 pattern1 = r'\d+-\w+-\d+ \d\d:\d\d:\d\d'
 pattern2 = r'\d+'
 pattern3 = r'\d+/\d+'
 pattern4 = r'\w+, \w+'

 ##'40' row!

 # 1)patient ID!
 p_id = [ x for x in l_40 if "ID" in x ]
 if len(p_id) > 1:
  p_id = [ x for x in p_id if "ID:" in x ]
  p_id = ''.join(p_id).replace('ID:','')

 # 2)Patient name!
  p_name = [ x for x in l_40 if re.findall(pattern4, x)]
  p_name = ''.join(p_name)

 # 3)Hospital name!
  hp_name = [ x for x in l_40 if "AJOU" in x ]
  hp_name = ''.join(hp_name)

 # 4)Date!
  date = [ x for x in l_40 if re.findall(pattern1, x)]
  date = ''.join(date)

 ##'67' row!

 # 1)Patient Age!
 p_yr = [ x for x in l_67 if 'yr' in x ]
 p_yr = ''.join(p_yr)
 p_yr = re.sub(r'\d+-\w+-\d+','',p_yr)
 p_yr = p_yr.replace('(','').replace(')','').replace('yr','')

 # 2)Vent.rate!
 vr_u = [ x for x in l_67 if 'BPM' in x ]
 vr_vl = [ x for x in l_67 if re.findall(pattern2, x) ]
 vr_vl = [ x for x in vr_vl if "yr" not in x ]
 vr_vl = [ x for x in vr_vl if "inus" not in x ]
 vr_vl = [ x for x in vr_vl if "bradycardia" not in x ]
 vr_vl = [ x for x in vr_vl if "Atrial " not in x ]
 vr_vl = [ x for x in vr_vl if "atrial" not in x ]
 vr_vl = [ x for x in vr_vl if "Mon" not in x ]
 vr_vl = [ x for x in vr_vl if "mon" not in x ]
 vr_vl = [ x for x in vr_vl if "days" not in x ]
 vr_vl = [ x for x in vr_vl if "complexes" not in x ]
 vr_vl = [ x for x in vr_vl if "ATRIAL" not in x ]
 vr_vl = [ x for x in vr_vl if "Unusual" not in x ]
 vr_vl = [ x for x in vr_vl if "-" not in x ]
 vr_vl = [ x for x in vr_vl if "AV BLOCK" not in x ]
 vr_vl = [ x for x in vr_vl if "AV" not in x ]
 vr_vl = [ x for x in vr_vl if "inversion" not in x ]
 vr_vl = [ x for x in vr_vl if "ECG" not in x ]
 vr_vl = [ x for x in vr_vl if "RATIO" not in x ]
 vr_vl = [ x for x in vr_vl if "hrs" not in x ]
 vr_vl = [ x for x in vr_vl if "wks" not in x ]
 vr_vl = [ x for x in vr_vl if "LEAD" not in x ]
 vr_vl = [ x for x in vr_vl if "*" not in x ]
 vr_vl = [ x for x in vr_vl if "waves" not in x ]
 vr_vl = [ x for x in vr_vl if "CHO" not in x ]
 vr_vl = [ x for x in vr_vl if "GX" not in x ]
 vr_vl = [ x for x in vr_vl if "$" not in x ]
 vr_vl = [ x for x in vr_vl if "SINUS" not in x ]
 vr_vl = [ x for x in vr_vl if "LEADS" not in x ]
 vent_rate = vr_vl
 if len(vent_rate) > 1:
  vent_rate = vent_rate[1]
  vent_rate = ''.join(vent_rate)

 # 3)Diagnosis in first row!
 dia1 = [item for item in l_67 if len(item)>=9]
 dia1 = [ x for x in dia1 if "Vent" not in x ]
 dia1 = [ x for x in dia1 if "yr" not in x ]
 dia1 = [x for x in dia1 if "Confirmed by" not in x]
 dia1 = [x for x in dia1 if "Reconfirmed by" not in x]


 ##'81' row!

 # 1)Sex of patient
 p_sex = [ x for x in l_81 if 'ale' in x ]
 p_sex = ''.join(p_sex)

 # 2)ethnicity of patient
 p_eth = [ x for x in l_81 if 'Oriental' in x ]
 p_eth_2 = [ x for x in l_81 if 'Asian' in x ]
 p_ethn = p_eth + p_eth_2
 p_ethn = ''.join(p_ethn)

 # 3)PR interval
 pr_i_u = [ x for x in l_81 if 'ms' in x ]
 pr_i_vl = [ x for x in l_81 if re.findall(pattern2, x)]
 pr_i_vl = [ x for x in pr_i_vl if "RSR' or QR pattern " not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "Atrial" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "ATRIAL" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "atrial" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "Increased" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "R/S" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "Sinus" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "sinus" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "rhythm" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "bundle" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "branch" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "because" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "and" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "by" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "RSR'" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "-" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "AV" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "mon" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "RATIO" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "QTc" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "QT" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "ms" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "r/s" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "CONSIDER" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "ventricular" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "RATIO" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "EARLY" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "PROBABLE" not in x ]
 pr_i_vl = [ x for x in pr_i_vl if "LEAD" not in x ]
 pr_interval = pr_i_vl
 pr_interval = ''.join(pr_interval)

 # 4)Diagnosis in second row!
 dia2 = [item for item in l_81 if len(item)>=9]
 dia2 = [ x for x in dia2 if "interval" not in x ]
 dia2 = [x for x in dia2 if "Confirmed by" not in x]
 dia2 = [x for x in dia2 if "Reconfirmed by" not in x]

 ##'94' row!

 # 1)Patient Height & Weight

 p_heigt = [ x for x in l_94 if 'cm' in x ]
 p_weight = [ x for x in l_94 if 'kg' in x ]
 p_heigt = ''.join(p_heigt)
 p_weight = ''.join(p_weight)
 p_heigt = str(p_heigt)
 p_weight = str(p_weight)
 p_heigt = p_heigt.replace('cm','')
 p_weight = p_weight.replace('kg','')

 # 2)QRS duration
 qrs_d_u = [ x for x in l_94 if 'ms' in x ]
 qrs_d_vl = [ x for x in l_94 if re.findall(pattern2, x)]
 qrs_d_vl = [ x for x in qrs_d_vl if "cm" not in x ]
 qrs_d_vl = [ x for x in qrs_d_vl if "kg" not in x ]
 qrs_duration = qrs_d_vl
 qrs_duration = ''.join(qrs_duration)

 # 3)Diagnosis in third row!
 dia3 = [item for item in l_94 if len(item)>=9]
 dia3 = [ x for x in dia3 if "duration" not in x ]
 dia3 = [x for x in dia3 if "Confirmed by" not in x]
 dia3 = [x for x in dia3 if "Reconfirmed by" not in x]


 ## top='95' row!(this row is unusual row for diagnosis information)
 dia4 = [item for item in l_95 if len(item)>=10]
 dia4 = [x for x in dia4 if "Confirmed by" not in x]
 dia4 = [x for x in dia4 if "Reconfirmed by" not in x]

 ##'108' row!

 # 1)QT/QTc
 qtqtc_u = [ x for x in l_108 if 'ms' in x ]
 qtqtc_vl = [ x for x in l_108 if re.findall(pattern3, x)]
 qtqtc_vl_tmp = ''.join(qtqtc_vl)
 qt_vl = [i.split('/', 1)[0] for i in qtqtc_vl]
 qtc_vl = [i.split('/', 1)[1] for i in qtqtc_vl]
 qt = qt_vl
 qtc = qtc_vl
 qt = ''.join(qt)
 qtc = ''.join(qtc)

 # 2)Room information
 room_info = [ x for x in l_108 if 'Room' in x ]
 room_info = ''.join(room_info).replace('Room:','')

 # 3)Diagnosis in fourth row!
 dia5 = [item for item in l_108 if len(item)>=9]
 dia5 = [ x for x in dia5 if "Room" not in x ]
 dia5 = [x for x in dia5 if "Confirmed by" not in x]
 dia5 = [x for x in dia5 if "Reconfirmed by" not in x]

 ##'109' row!(this row also unusual row for diagnosis information)
 dia6 = [item for item in l_109 if len(item)>=10]
 dia6 = [x for x in dia6 if "Confirmed by" not in x]
 dia6 = [x for x in dia6 if "Reconfirmed by" not in x]

 ##'122' row!

 # 1)Location information
 loc_info = [ x for x in l_122 if "Loc" in x ]
 loc_info = ''.join(loc_info).replace('Loc:','')

 # 2) Diagnosis information in fifth row!
 dia7 = [item for item in l_122 if len(item)>=9]
 dia7 = [ x for x in dia7 if "P-R-T" not in x ]
 dia7 = [x for x in dia7 if "Confirmed by" not in x]
 dia7 = [x for x in dia7 if "Reconfirmed by" not in x]

 
 ##'123' row!(This row also unusual row for Diagnosis information!)
 dia8 = [item for item in l_123 if len(item)>=9]
 dia8 = [x for x in dia8 if "Confirmed by" not in x]
 dia8 = [x for x in dia8 if "Reconfirmed by" not in x]


 ## Collect Diagnosis information!
 diag = dia1+dia2+dia3+dia4+dia5+dia6+dia7+dia8
 diag = '/'.join(diag)

  return p_id, date,p_ethn, p_yr, p_heigt,p_weight, vent_rate, p_sex, pr_interval, qrs_duration, qt, qtc, room_info, loc_info, diag, qtqtc_vl_tmp

#Find P-R-T axes value!
def fnd_prt(inf_tmp_2,qtqtc_vl_tmp):
 init_index = 0
 initial_index = 0
 tmp = []
 for n in inf_tmp_2:
  n[0]=''
  n[1]=''
  tmp = tmp + [n]
  n=[]

 for i,j in enumerate(tmp):
  if j[2] == 'P-R-T axes':
   final_index = i
  elif j[2] == qtqtc_vl_tmp:
   init_index = init_index + i
   initial_index = initial_index + init_index + 1
 else:
  pass

 prt_tmp = tmp[initial_index : final_index]
 p_axes = list(prt_tmp[2][2])
 p_axes = [ x for x in p_axes if "*" not in x ]
 if len(p_axes) > 3:
 p_axes = []

 p_axes = ''.join(p_axes)
 r_axes = list(prt_tmp[1][2])
 r_axes = [ x for x in r_axes if "*" not in x ]
 if len(r_axes) > 3:
 r_axes = []
 
 r_axes = ''.join(r_axes)
 t_axes = list(prt_tmp[0][2])
 t_axes = [ x for x in t_axes if "*" not in x ]
 t_axes = [ x for x in t_axes if "sinus" not in x ]
 if len(t_axes) > 3:
 t_axes = []


 t_axes = ''.join(t_axes)

  return p_axes, r_axes, t_axes

def execute(filelist):
 os.chdir("/directory/xml")
 try:
 inf_tmp, filename = par_elem(filelist)
 inf_tmp_2 = par_elem_2(filelist)
 l_40,l_67, l_81, l_94,l_95,l_108,l_109,l_122,l_123 = col_elem(inf_tmp,filename)
 p_id, date, p_ethn,p_yr, p_heigt,p_weight, p_sex, vent_rate, pr_interval, qrs_duration, qt, qtc, room_info, loc_info, diag, qtqtc_vl_tmp = find_values(l_40,l_67, l_81, l_94,l_95,l_108,l_109,l_122,l_123)

 p_axes, r_axes, t_axes = fnd_prt(inf_tmp_2, qtqtc_vl_tmp)
 dt_lst = [p_id,date,p_ethn,p_yr,p_heigt,p_weight,vent_rate,p_sex,pr_interval,qrs_duration,qt, qtc, room_info, loc_info, p_axes, r_axes,t_axes,diag]

 dt_lst = pd.DataFrame([dt_lst])
 return dt_lst

 except KeyError:
  print(filename)
  os.system("mv /directory/xml/" + filename + " /directory/unusual_xml/")

  return None

def main(filelist):
 processor = cpu_count()
 proc = os.getpid()

 print("proc_id",proc)
 print(os.fork())
 pool = Pool(processes = cpu_count())
 print("Number of processor:",processor)
 startTime = int(time.time())
 counter = 0

 dt_tmp = pool.map(execute,filelist)
 if dt_tmp is not None :
  df_tmp =pd.DataFrame({"Patient_info":["Patient_ID","Date","Ethnicity","Age(yr)","Height(cm)","Weight(kg)","Sex","Vent.rate(BPM)","PR_interval(ms)","QRS_duration(ms)", "QC(ms)","QTc(ms)","Room","Location number","P_axes","R_axes",'T_axes',"Diagnosis"]})
  df_tmp = df_tmp.T

  df = pd.concat(dt_tmp)

  df_tmp = pd.concat([df_tmp, df], axis=0,ignore_index=True)
  os.chdir("/directory/text/")
  df_tmp.to_csv("Result.txt",header=None, index=None,sep=',',encoding='utf-8')
  df_tmp.to_csv("Result.csv",encoding='utf-8')
  print("Complete")

  endTime = int(time.time())
  print("Total converting time", (endTime - startTime))

if __name__ == "__main__":
 print("Start extract ECG text data!")

 os.chdir("/directory/")
 search_directory = "xml"
 filelist = listdir(search_directory)
 main(filelist)
