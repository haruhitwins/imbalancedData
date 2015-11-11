# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:20:57 2015

@author: Haolin
"""

atr1 = {"icmp":0, "tcp":1, "udp":2}
atr2 = {"auth":0, "bgp":1, "courier":2, "csnet_ns":3, "ctf":4, \
        "daytime":5, "discard":6, "domain":7, "domain_u":8, \
        "echo":9, "eco_i":10, "ecr_i":11, "efs":12, \
        "exec":13, "finger":14, "ftp":15, "ftp_data":16, \
        "gopher":17, "hostnames":18, "http":19, "http_443":20, \
        "imap4":21, "IRC":22, "iso_tsap":23, "klogin":24, \
        "kshell":25, "ldap":26, "link":27, "login":28, \
        "mtp":29, "name":30, "netbios_dgm":31, "netbios_ns":32, \
        "netbios_ssn":33, "netstat":34, "nnsp":35, "nntp":36, \
        "ntp_u":37, "other":38, "pm_dump":39, "pop_2":40, \
        "pop_3":41, "printer":42, "private":43, "red_i":44, \
        "remote_job":45, "rje":46, "shell":47, "smtp":48, \
        "sql_net":49, "ssh":50, "sunrpc":51, "supdup":52, \
        "systat":53, "telnet":54, "tftp_u":55, "time":56, \
        "tim_i":57, "urh_i":58, "urp_i":59, "uucp":60, \
        "uucp_path":61, "vmnet":62, "whois":63, "X11":64, \
        "Z39_50":65}
atr3 = {"OTH":0, "REJ":1, "RSTO":2, "RSTOS0":3, "RSTR":4, \
        "S0":5, "S1":6, "S2":7, "S3":8, "SF":9, "SH":10}

fileName = 'kddcup-rootkit-imap_vs_back.dat'
lines = []
with open(fileName) as f:
    for l in f:
        if l[0] == '@': 
            continue
        outputline = []
        l = l.strip().split(',')
        label = l.pop()
        assert label == "positive" or label == "negative"
        if label == "positive":
            outputline.append(1)
        else:
            outputline.append(0)
        outputline.append(l[0])
        tmp = [0] * 3
        tmp[atr1[l[1]]] = 1
        outputline.extend(tmp)
        tmp = [0] * 66
        tmp[atr2[l[2]]] = 1
        outputline.extend(tmp)
        tmp = [0] * 11
        tmp[atr3[l[3]]] = 1
        outputline.extend(tmp)
        outputline.extend(l[4:])
        outputline = [str(x) for x in outputline]
        lines.append(','.join(outputline)+'\n')
        
with open('trans1_'+fileName, 'w') as f:
    f.writelines(lines)
    
lines = []
with open(fileName) as f:
    for l in f:
        if l[0] == '@': 
            continue
        outputline = []
        l = l.strip().split(',')
        label = l.pop()
        assert label == "positive" or label == "negative"
        if label == "positive":
            outputline.append(1)
        else:
            outputline.append(0)
        outputline.append(l[0])
        outputline.append(atr1[l[1]])
        outputline.append(atr2[l[2]])
        outputline.append(atr3[l[3]])
        outputline.extend(l[4:])
        outputline = [str(x) for x in outputline]
        lines.append(','.join(outputline)+'\n')
        
with open('trans2_'+fileName, 'w') as f:
    f.writelines(lines)