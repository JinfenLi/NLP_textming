newlinelist = []
with open("data.txt", 'r') as file:
    line = file.readline()

    while line:
        newline = line[:4] + '\t' + line[5:13]+"\t"+line[14:]
        newlinelist.append(newline)
        line = file.readline()


with open("newdata.txt", 'w') as file:
    file.writelines(newlinelist)
