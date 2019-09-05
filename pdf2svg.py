###This code is for converting PDF file to SVG file.

##import library for converting ##
from multiprocessing import Process
from multiprocessing import Pool, cpu_count
from os import listdir
import time
import os


## execute function for converting ##
def execute(filelist):
 os.chdir("/directory/pdf ")
 command = "inkscape -z -f /directory/pdf/"+filelist+" -l /directory/svg/"+filelist+".svg"
 os.system(command)
 os.chdir("/directory/pdf ")
 #directory is the physical directory where the file is located





if __name__ == '__main__':
 print("Start convert ECG data!")

 os.chdir("/directory/")
 search_directory = "pdf "
 filelist = listdir(search_directory)

 processor = cpu_count()
 proc = os.getpid()

 print("proc_id",proc)
 print(os.fork())
 print("Number of processor:",processor)

 print("Number_of_pdf_file :", len(filelist))

 pool = Pool(processes = cpu_count())

 startTime = int(time.time())
 print(startTime)
 pool.map(execute, filelist)
 endTime = int(time.time())
 print("Total converting time", (endTime - startTime))
