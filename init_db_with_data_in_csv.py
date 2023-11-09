import sqlite3
import csv
with open('./members.csv', newline='') as f:
    csv_reader = csv.DictReader(f)
    members = [(row['名字'], row['團體']) for row in csv_reader]

    print(members)


with open('create_db.sql') as f:
    create_db_sql = f.read()
    db = sqlite3.connect('members.db')

with db:
    db.executescript(create_db_sql)

with db:
    db.executemany('INSERT INTO  members (name, group_name) VALUES (?, ?)', members)

c = db.execute('SELECT * FROM members LIMIT 3')
for row in c:
    print(row)
'''    
(1, '高坂 穂乃果', "μ's")
(2, '絢瀬 絵里', "μ's")
(3, '南 ことり', "μ's")
'''