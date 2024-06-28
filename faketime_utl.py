import os
import sys
import time

import ntplib


def RequestTimefromNtp(addr="time.google.com"):
    c = ntplib.NTPClient()
    # Provide the respective ntp server ip in below function
    response = c.request(addr, version=3)
    return time.ctime(response.tx_time)


def ensure_time():
    if not os.environ.get("calibtime"):
        print("Calibrating time")
        t = RequestTimefromNtp()
        print("Time calibrated to", t)
        os.environ["calibtime"] = "1"
        # os.system(f"time ./faketime '{t}' {sys.executable} {' '.join(sys.argv)}")
        os.system(f"./faketime '{t}' {sys.executable} {' '.join(sys.argv)}")
        sys.exit()


# print(datetime.datetime.now())
# options = {
#     "WLSACCESSID": "abcde",
#     "WLSSECRET": "abcde",
#     "LICENSEID": 000000,
# }
# with gp.Env(params=gulic.options) as env, gp.Model(env=env) as model:
#     # Formulate problem
#     model.optimize()
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
