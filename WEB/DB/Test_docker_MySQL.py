import os
from dotenv import load_dotenv
import mysql.connector
import time

load_dotenv()

try:
    start_time = time.time()
    
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="acro0",
        password=os.getenv("MySQL_passwd"), 
        connection_timeout=10,
        auth_plugin="mysql_native_password"
    )
    
    connection_time = time.time() - start_time
    print(f"✅ Connected via 127.0.0.1 in {connection_time:.2f}s!")
    
    cursor = conn.cursor()
    # sql = "CREATE DATABASE TestDB"
    # sql = """CREATE TABLE TestDB.Employees (
    #     EmployeeID INT PRIMARY KEY,
    #     FirstName VARCHAR(50) NOT NULL,
    #     LastName VARCHAR(50) NOT NULL,
    #     Department VARCHAR(50),
    #     Salary DECIMAL(10, 2)
    # );"""
#     sql = """INSERT INTO TestDB.Employees (EmployeeID, FirstName, LastName, Department, Salary)
# VALUES
#     (1, 'John', 'Doe', 'Sales', 60000.00),
#     (2, 'Jane', 'Smith', 'Marketing', 65000.00),
#     (3, 'Peter', 'Jones', 'Engineering', 80000.00),
#     (4, 'Mary', 'Brown', 'HR', 55000.00),
#     (5, 'Michael', 'Davis', 'Finance', 75000.00),
#     (6, 'Emily', 'Wilson', 'Sales', 62000.00),
#     (7, 'David', 'Garcia', 'Engineering', 82000.00),
#     (8, 'Sarah', 'Miller', 'Marketing', 68000.00),
#     (9, 'Chris', 'Taylor', 'HR', 58000.00),
#     (10, 'Laura', 'Moore', 'Finance', 78000.00);
# """
    sql = """SELECT Department, COUNT(*) FROM TestDB.Employees GROUP BY Department;"""
    # sql = """
    # UPDATE TestDB.Employees 
    # SET Salary = 65000.00
    # WHERE FirstName = 'John' AND LastName = 'Doe';
    # """
    # sql = """
    # DELETE FROM TestDB.Employees 
    # WHERE EmployeeID = 2;
    # """
    cursor.execute(sql)
    # conn.commit()  # only for Create-Update-Delete
    # result = cursor.fetchone()
    result = cursor.fetchall()
    print(f"Query result: {result}")
    print(f"Affected Rows : {cursor.rowcount}")
    
    cursor.close()
    conn.close()
    print(f"✅ 127.0.0.1 connection closed!")
    
except mysql.connector.Error as err:
    print(f"❌ MySQL Error with 127.0.0.1: {err}")
except Exception as e:
    print(f"❌ Other error with 127.0.0.1 : {type(e).__name__}: {e}")
