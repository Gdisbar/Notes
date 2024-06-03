import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Ras25@3355",
  database="pandeyji_eatery"
)

mycursor = mydb.cursor()


# sql = "INSERT INTO pandeyji_eatery.food_items (item_id,name, price) VALUES (%s, %s, %s)"
# val = (11,'Khasta Kochuri',3.00)
# mycursor.execute(sql, val)
# val = [(11,'Khasta Kochuri',3.00),
#   (12,'Cholar Dal',4.00)
# ]
# mycursor.executemany(sql, val)

# mydb.commit()

# sql = "UPDATE pandeyji_eatery.food_items SET name = 'Misti Doi' WHERE item_id = 10"
# mycursor.execute(sql)

# mydb.commit()

print(mycursor.rowcount, "record inserted.")

mycursor.execute("SELECT * FROM pandeyji_eatery.food_items")

myresult = mycursor.fetchall()

for x in myresult:
  print(x)