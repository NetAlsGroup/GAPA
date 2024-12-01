import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ""))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))

currentDir = os.path.dirname(os.path.realpath(__file__))
parentDir = os.path.dirname(os.path.dirname(currentDir))
sys.path.append(parentDir)
