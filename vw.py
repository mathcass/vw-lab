import sys

from vowpalwabbit import pyvw


opts = sys.argv[1:]
print(opts)
model = pyvw.vw(" ".join(opts))
