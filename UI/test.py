import csv

file = open("D:\Maui\Code\MDS final\input\iris-flower-dataset\IRIS.csv")
type(file)
csvreader = csv.reader(file)
header = []
header = next(csvreader)
print(header)