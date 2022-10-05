# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:17:34 2022

@author: lxie2
"""

import subprocess
from time import sleep
dir_name = 'test'
wake_str = '17:30'
data_process = subprocess.Popen(["python", "dynamic_analysis_debug.py", dir_name, wake_str], \
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# data_process = subprocess.run(["python", "dynamic_analysis_debug.py", dir_name, wake_str])
poll = data_process.poll()
while poll == None:
    sleep(10)
    poll = data_process.poll()
print(f'return code is {data_process.returncode}')
process_output = data_process.stdout.read1().decode('utf-8').strip()
print('process output:' + process_output)
process_error = data_process.stderr.read1().decode('utf-8').strip()
print('process error:' + process_error)
