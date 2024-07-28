import linecache
import sys
path = sys.argv[1]
if not path.endswith('.rs'):
    exit()