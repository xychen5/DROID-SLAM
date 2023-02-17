import os
import sys

def ListFilesToTxt(dir, file, wildcard, recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname = os.path.join(dir, name)
        if(os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname, file, wildcard, recursion)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    file.write(name[:-4] + " " + fullname + "\n")
                    break

tarDir = sys.argv[1]
outputPath = sys.argv[2]

def Test():
    dir = tarDir # 文件路径
    file = open(outputPath, "w")
    wildcard = "png"
    if not file:
        print("cannot open the file %s for writing" % outputPath)
 
    ListFilesToTxt(dir, file, wildcard, 1)
 
    file.close()

Test()

