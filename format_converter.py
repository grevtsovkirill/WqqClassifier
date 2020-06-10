import os

#samples = ['413008','304014','345874','345875','363356','363357','363358','363359','364285','410081','410156','410157','410218','410219','410220']
samples = ['410470','700000']     
comp = ['a','d','e']
#option = ""
ver = "v9"

for i in samples:
    for j in comp:
        tick = "'"
        command = 'python convertor.py -i ../Files/'+ver+'/ -f skimReco_'+i+'_'+j+' -t outTree -o ../Files/processed'
        print(command)
        os.system(command)
