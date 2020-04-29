import os

samples = ['413008','304014','345874','345875','363356','363357','363358','363359','364285','410081','410156','410157','410218','410219','410220']
#samples = ['410470']     
#option = ""
for i in samples:
    tick = "'"
    command = 'python convertor.py -i ../Files/ -f skimReco_'+i+'_xs -t outTree'
    print(command)
    os.system(command)
