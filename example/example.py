import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from pyddlib.mlexpr import MLExpr
from pyddlib.padd import PADD

# Define variables
S0 = PADD.variable(0)
S1 = PADD.variable(1)
S5 = PADD.variable(5)
S6 = PADD.variable(6)
S7 = PADD.variable(7)
S8 = PADD.variable(8)


# Eliminate S5: mu1 = sum_S5 AX*A5
## Assemble A5
A5 = PADD.constant(MLExpr({():0.4}))*(~S5)+PADD.constant(MLExpr({():0.6}))*S5
## Assemble AX
nx = PADD.constant(MLExpr({("a_0",):1}))      
px = PADD.constant(MLExpr({("a_1",):1}))      
T5 = nx*(~S5)+px*S5 # auxiliary ADD
T6 = nx*(~S6)+px*S6 # auxiliary ADD
AX = T5*(~S0) + (T5*(~S1) + T6*S1)*S0
## Combine and eliminate S5
phi1 = AX*A5
mu1 = phi1.restrict({5:False}) + phi1.restrict({5:True})
print("mu1", mu1)
### compute mu1 prime
u1 = PADD.constant(MLExpr({("u1",):1}))   # auxiliary variables
u2 = PADD.constant(MLExpr({("u2",):1}))
u3 = PADD.constant(MLExpr({("u3",):1}))
mu1p = (~S0)*u1 + S0*(u1*(~S1) + S1*(u2*(~S6) + u3*S6))
#print("mu1prime", mu1p)

# Eliminate S6: mu2 = sum_S6 mu1p*A6
## Assemble A6
A6 = PADD.constant(MLExpr({():0.1}))*(~S6)+PADD.constant(MLExpr({():0.9}))*S6
phi2 = mu1p*A6
#print(phi2)
mu2 = phi2.restrict({6:False}) + phi2.restrict({6:True})
print("mu2", mu2)
### Compute mu2 prime
u4 = PADD.constant(MLExpr({("u4",):1}))   # auxiliary variables
u5 = PADD.constant(MLExpr({("u5",):1}))
mu2p = (~S0)*u4 + S0*((~S1)*u4 + S1*u5)
#print("mu2p",mu2p)
# Eliminate S1: mu3 = sum_S1 mu2p*A1
A1 = PADD.constant(MLExpr({():0.625}))*(~S1)+PADD.constant(MLExpr({():0.375}))*S1
#print(A1)
phi3 = mu2p*A1
#print(phi3)
mu3 = phi3.restrict({1:False})+phi3.restrict({1:True})
print("mu3",mu3)
u6 = PADD.constant(MLExpr({("u6",):1}))   # auxiliary variables
u7 = PADD.constant(MLExpr({("u7",):1}))
mu3p = ~S0*u6 + S0*u7

# Eliminate S8: mu4 = sum_S8 mu3p * A8 * AY
A8 = PADD.constant(MLExpr({():0.8}))*(~S8)+PADD.constant(MLExpr({():0.2}))*S8
ny = PADD.constant(MLExpr({("b_0",):1}) )   
py = PADD.constant(MLExpr({("b_1",):1}) ) 
AY = (~S0)*((~S7)*ny+S7*py)+S0*((~S8)*ny+S8*py)
phi4 = mu3p*A8*AY
mu4 = phi4.restrict({8:False})+phi4.restrict({8:True})
print("mu4",mu4)
u8 = PADD.constant(MLExpr({("u8",):1}))   # auxiliary variables
u9 = PADD.constant(MLExpr({("u9",):1}))
u10 = PADD.constant(MLExpr({("u10",):1}))
mu4p = S0*u10 + (~S0)*(S7*u9+(~S7)*u8)

# Eliminate S0: mu5 = sum_S0 mu4p * A0
A0 = PADD.constant(MLExpr({():0.2}))*(~S0)+PADD.constant(MLExpr({():0.8}))*S0
phi5 = mu4p*A0
mu5 = phi5.restrict({0:False}) + phi5.restrict({0:True})
print("mu5",mu5)
u11 = PADD.constant(MLExpr({("u11",):1}))   # auxiliary variables
u12 = PADD.constant(MLExpr({("u12",):1}))
mu5p = ~S7*u11 + S7*u12

# Eliminate S7: mu6 = sum_S7 mu5p * A7
A7 = PADD.constant(MLExpr({():0.3}))*(~S7)+PADD.constant(MLExpr({():0.7}))*S7
phi6 = mu5p*A7
mu6 = phi6.restrict({7:False}) + phi6.restrict({7:True})
print("mu6",mu6)
