import datetime
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.request
import gulic
import gurobipy as gp
import ntplib


def RequestTimefromNtp(addr="time.google.com"):
    c = ntplib.NTPClient()
    # Provide the respective ntp server ip in below function
    response = c.request(addr, version=3)
    return time.ctime(response.tx_time)


if not os.environ.get("calibtime"):
    t = RequestTimefromNtp()
    os.environ["calibtime"] = "1"
    os.system(f"faketime '{t}' {sys.executable} {' '.join(sys.argv)}")
    exit(0)
# print(datetime.datetime.now())
# options = {
#     "WLSACCESSID": "abcde",
#     "WLSSECRET": "abcde",
#     "LICENSEID": 000000,
# }
with gp.Env(params=gulic.options) as env, gp.Model(env=env) as model:
    # Formulate problem
    model.optimize()
# os.execv(sys.executable, ["python"] + sys.argv)
# print(sys.argv)

# os.execv(sys.argv[0], sys.argv)
# os.popen(f"faketime{t}")
# print(time.asctime(RequestTimefromNtp()[1]))
# time.tzset()
# time.gmtime(RequestTimefromNtp()[1])
# time.tzset()
# print(time.strftime("%X %x %Z"))
# url = "http://pool.ntp.org/ntp/o/uri.bin"
# response = urllib.request.urlopen(url)
# data = response.read()
# time_data = struct.unpack("!12ll", data[:96])
# timestamp = time_data[0] - 2208988800
# print(timestamp)
# WLSACCESSID=f8a8d24b-bc3b-4822-a718-9196fa3a0788
# WLSSECRET=2997c8d2-37ea-4c20-aa4d-893beecc9615
# LICENSEID=2519992
