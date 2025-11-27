#!/usr/bin/env python3

import os
print("DEBUG: Setting OMP_NUM_THREADS...")
os.environ['OMP_NUM_THREADS'] = '1'

print("DEBUG: Importing core module...")
from facefusion import core

print("DEBUG: Starting main block...")
if __name__ == '__main__':
	print("DEBUG: Calling core.cli()...")
	core.cli()
	print("DEBUG: core.cli() finished.")