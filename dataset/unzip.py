# Author: Jerome Zhou
# (Because kao-server doesn't have unzip...)

import zipfile
import sys

if len(sys.argv) != 3:
	print("Number of args incorrect!")
	sys.exit(1)

with zipfile.ZipFile(str(sys.argv[1]), "r") as z:
	z.extractall(str(sys.argv[2]))
