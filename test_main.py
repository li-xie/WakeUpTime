# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:17:34 2022

@author: lxie2
"""

import subprocess
dir_name = '20220911'
wake_str = '8:30'
data_process = subprocess.Popen(["python", "dynamic_analysis", dir_name, wake_str], \
                                stdout=subprocess.PIPE)
poll = data_process.poll()
if poll != None:
    process_output = data_process.stdout.read1().decode('utf-8').strip()
    print(process_output)