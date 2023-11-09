import csv

with open('./members.csv', newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        print(row)


print("==========================")

# print()

with open('./members.csv', newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        print('{} of {}'.format(row['名字'], row['團體']))