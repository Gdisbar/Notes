-- WHERE vs HAVING
==================================================================
-- WHERE is used on rows & can't be used with aggregate function
-- HAVING is used either with aggregate function or with group by

-- aggregate function --> function that combines multiple rows
-- types - COUNT(),MIN(),MAX(),AVG(),SUM()

Employee
--------------------------------
| emp_id | salary | experience |
--------------------------------

SELECT MAX(salary) FROM Employee HAVING experience > 3;


==================================================================
-- UNION vs UNION ALL
==================================================================
-- UNION removes duplicate but UNION ALL doesn't duplicate while 


-- UNION combine result of 2 or more select statement where each select
-- statement must follow these conditions
-- > same number of columns
-- > columns must have similar datatypes
-- > column must be in same order i.e id -> address -> contact order of 
-- 	columns should be like this


Customer
------------------------------------------
| cust_id | cust_address | cust_contact |
------------------------------------------

Supplier
------------------------------------------
| supp_id | supp_address | supp_contact |
------------------------------------------

CREATE TEMPORARY TABLE CombinedData AS (
    SELECT cust_id, cust_address, cust_contact
    FROM Customer
    WHERE cust_id BETWEEN 1 AND 5
    UNION ALL
    SELECT supp_id, supp_address, supp_contact
    FROM Supplier
    WHERE supp_id BETWEEN 1 AND 5
);
-- This will have 10 records first 5 from Customer 
-- & next 5 from Supplier , name of columns will be that of
-- Customer table column names
SELECT cust_id,cust_address,cust_contact FROM CombinedData; 

==================================================================
-- IN vs EXISTS
==================================================================
-- IN - multiple OR used when bigger outer query and smaller inner query
-- EXISTS - returns either true or false used when smaller outer query
-- and bigger inner query


SELECT * FROM Customer WHERE city='Mumbai' OR city='Chennai'
OR city='Bangalore';

SELECT * FROM Customer WHERE city IN(Mumbai,Chennai,Bangalore);

Departments
---------------------------------
| DepartmentID | DepartmentName | 
---------------------------------

Employees
---------------------------------------------
| EmployeeID | EmployeeName | DepartmentID  |
---------------------------------------------

-- CREATE TABLE Departments ( DepartmentID int, 
--     DepartmentName varchar(255), PRIMARY KEY (DepartmentID) );

-- CREATE TABLE Employees ( EmployeeID int,
-- EmployeeName varchar(255), DepartmentID int, PRIMARY KEY (EmployeeID), 
-- FOREIGN KEY (DepartmentID),
-- REFERENCES Departments(DepartmentID) );

SELECT DepartmentName FROM Departments 
WHERE EXISTS ( SELECT 1 FROM Employees 
WHERE Employees.DepartmentID = Departments.DepartmentID );

-- The above query gives all department names that have at 
-- least one employee. EXISTS checks for the presence of 
-- any row in the Employees table that matches the condition.

SELECT EmployeeName FROM Employees 
WHERE DepartmentID IN 
( SELECT DepartmentID FROM Departments WHERE 
    DepartmentName = 'HR' OR DepartmentName = 'IT' );

-- The above query gives a list of all the employees which 
-- belong to either the HR or IT departments. IN matches the DepartmentID 
-- against a list of department IDs returned by the subquery.

==================================================================
--JOIN vs SUBQUERY
==================================================================
-- SUBQUERY can only select from 1st table but JOIN can select from
-- both tables

-- JOINs are used when the relationships between the tables 
-- are known and fixed.

-- SUBQUERY is slower than JOIN

Customer
-------------------------------------
| cust_id | cust_name| cust_contact |
--------------------------------------

Order
-----------------------
| order_id | cust_id |
-----------------------

SELECT cust_name,cust_contact FROM Customer 
WHERE cust_id IN (SELECT cust_id FROM Order);

SELECT cust_name,cust_contact,order_id FROM Customer c 
JOIN Order o ON c.cust_id=o.cust_id;

==================================================================
-- JOIN vs UNION
==================================================================
-- UNION combine rows i.e grows vertically, 
-- JOIN merges columns i.e grows horizontally

-- for UNION it's not necessary to have same column name but no. of
-- columns & datatype of columns should be same  

-- JOIN works by combining rows from 2 or more tables based on a 
-- related common column between them

Customers
--------------------------------------
CustomerID  CustomerName    City
1            Alice          New York
2            Bob            Los Angeles
3            Charlie        Chicago

Orders
--------------------------------
OrderID CustomerID  OrderDate
101     1           2023-11-11
102     2           2023-11-12
103     3           2023-11-13

SELECT CustomerID FROM Customers
UNION SELECT CustomerID FROM Orders;

-- CustomerID 
-- --------------------
-- 1
-- 2
-- 3
-- 101
-- 102
-- 103


SELECT Customers.CustomerName, Orders.OrderDate
FROM Customers
INNER JOIN Orders ON Customers.CustomerID = Orders.CustomerID;

-- CustomerName    OrderDate
-- Alice           2023-11-11
-- Bob             2023-11-12
-- Charlie         2023-11-13


