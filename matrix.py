import numpy as np
def get_matrix(rows,cols,name):
print(f"enter elements of{name}(space-separated row-wise):")
matrix=[]
for i in range(rows):
row = list(map(float,input(f"Row{i+1}:").split()))
matrix.append(row)
return np.array(matrix)
#input for matrix1
r1=int(input("Enter number of rows for matrix 1:"))
c1=int(input("Enter number of columns for matrix1:"))
matrix1=get_matrix(r1,c1,"matrix 1")
# input for matrix2
r2 = int(input("Enter the rows of matrix 2:"))
c2 = int(input("Enter the columns of matrix 2:"))
matrix2 = get_matrix(r2, c2, "matrix 2")
#Addition
if matrix1.shape == matrix2.shape:
print("\naddition:\n",matrix1+matrix2)
else:
print("Addition not possible")
#subtraction
if matrix1.shape == matrix2.shape:
print("\nsubtraction:\n",matrix1-matrix2)
else:
print("Subtraction not possible")
#multiplication
if c1==c2:
print("\nmultiplication:\n",np.dot(matrix1,matrix2))
else:
print("\multiplication not possible")
#division(element wise)
if matrix1.shape == matrix2.shape:
try:
print("\nelemnt-wise division:\n",matrix1 /matrix2)
except ZeroDivisionError:
print("\nDivision by zero detected")
else:
print("\nDivision not possible")
#inverse
def matrix_inverse(matrix,name):
if matrix.shape[0]!=matrix.shape[1]:
print(f"\n{name} is not square,Inverse not possible.")
return
try:
inv=np.linalg.inv(matrix)
print(f"\nInverse of {name}:\n",inv)
except np.linalg.LinAlgError:
print(f"\n{name} is not invertible.")
matrix_inverse(matrix1,"matrix1")
matrix_inverse(matrix2,"matrix2")
