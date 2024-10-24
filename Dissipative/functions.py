import numpy as np
from numpy import linalg as la
import qutip as qt
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
"""---------PAULI MATRICES---------It is faster with qutip"""

def X_list(N): # 
    X=[]
    I=qt.qeye(2)
    x=qt.sigmax()
    for i in range(N):
        Op_list=[]
        for j in range(N):
            Op_list.append(I)
        Op_list[i] = x
        X.append(qt.tensor(Op_list))
    return X
def Y_list(N): #QUTIP
    Y=[]
    I=qt.qeye(2)
    y=qt.sigmay()
    for i in range(N):
        Op_list=[]
        for j in range(N):
            Op_list.append(I)
        Op_list[i] = y
        Y.append(qt.tensor(Op_list))
    return Y
def Z_list(N): #QUTIP
    Z=[]
    I=qt.qeye(2)
    z=qt.sigmaz()
    for i in range(N):
        Op_list=[]
        for j in range(N):
            Op_list.append(I)
        Op_list[i] = z
        Z.append(qt.tensor(Op_list))
    return Z

def X_mom_1(N):
    Mom=[]
    X_l=X_list(N)
    for i in range(N) :
        for j in range(i+1,N) :
            Mom.append(qt.tensor(X_l[i])*qt.tensor(X_l[j]))
    return Mom

def Y_mom_1(N):
    Mom=[]
    Y_l=Y_list(N)
    for i in range(N) :
        for j in range(i+1,N) :
            Mom.append(qt.tensor(Y_l[i])*qt.tensor(Y_l[j]))
    return Mom

def Z_mom_1(N):
    Mom=[]
    Z_l=Z_list(N)
    for i in range(N) :
        for j in range(i+1,N) :
            Mom.append(qt.tensor(Z_l[i])*qt.tensor(Z_l[j]))
    return Mom

def Observables_mom_1(N,func):
    O=[]
    I=qt.qeye(2)
    o=func
    for i in range(N):
        Op_list=[]
        for j in range(2*N):
            Op_list.append(I)
        Op_list[i] = o
        O.append(qt.tensor(Op_list))
    Mom=[]
    for i in range(N) :
        for j in range(i+1,N) :
            Mom.append(qt.tensor(O[i])*qt.tensor(O[j]))
    return Mom
    

def X_sing(N): #Qutip

    I = qt.qeye(2)
    x = qt.sigmax()

    Op_list = []
    for j in range(N):
            Op_list.append(I)
    Op_list[0] = x
    X=qt.tensor(Op_list)
    return X

def Y_sing(N): #Qutip

    I = qt.qeye(2)
    y = qt.sigmay()

    Op_list = []
    for j in range(N):
            Op_list.append(I)
    Op_list[0] = y
    Y=qt.tensor(Op_list)
    return X


def Z_sing(N):  # Qutip

    I = qt.qeye(2)
    z = qt.sigmaz()

    Op_list = []
    for j in range(N):
        Op_list.append(I)
    Op_list[0] = z
    Z = qt.tensor(Op_list)
    return Z

def Y_rot_sing(N, phi):
    I = qt.qeye(2)
    Ry=qt.Qobj([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])

    Op_list = []
    for j in range(N):
        Op_list.append(I)
    Op_list[0] = Ry
    Ry_tot = qt.tensor(Op_list)
    return Ry_tot

def Y_rot_sing_dag(N, phi):
    I = qt.qeye(2)
    Ry=qt.Qobj([[np.cos(phi / 2), np.sin(phi / 2)], [-np.sin(phi / 2), np.cos(phi / 2)]])

    Op_list = []
    for j in range(N):
        Op_list.append(I)
    Op_list[0] = Ry
    Ry_tot = qt.tensor(Op_list)
    return Ry_tot

def Proj_to_ex(N, i=0):
    proj=qt.Qobj([[1, 1], [0, 0]])
    I = qt.qeye(2)

    Op_list = []
    for j in range(N):
        Op_list.append(I)
    Op_list[i] = proj

    P=qt.tensor(Op_list)

    return P

def Proj_to_gr(N, i=0):
    proj=qt.Qobj([[0, 0], [1, 1]])
    I = qt.qeye(2)

    Op_list = []
    for j in range(N):
        Op_list.append(I)
    Op_list[i] = proj

    P=qt.tensor(Op_list)

    return P

"""-----------------INITIAL CONDITION------------------------"""
def initial_all_equal_dm(N):
	dims=int(2**N)
	dm = np.full((dims,dims),1.0/float(dims),dtype=np.complex128)
	return dm

def initial_random_dm(N):
    rho=qt.rand_dm(int(pow(2,N)),dims=[[2]*N,[2]*N])
    return rho.full()

def initial_diagonal_dm(N):
	dims=int(2**N)
	dm=np.eye(dims,dtype=np.complex128)*1.0/float(dims)
	return dm

"""--------TOPOLOGY--------------"""
def J_all2all_het(Js,N):     #All to all random network
	J=np.random.uniform(-Js/2.0,Js/2.0,(N,N))
	return J

def J_all2all_hom(Js,N): #All to all homogeneous network
	J=np.full((N,N),Js)
	return J


"""-------EXTERNAL MAGNETIC FIELD-------"""
def h_het(hs,N):
	h=np.random.uniform(-hs,hs,N)
	return h

def h_hom(hs,N):
	h=np.full(N,hs)
	return h

"""---------HAMILTONIAN---------"""
def H_XXZ(J,h,N):#QUTIP
    H=0
    X_=X_list(N)
    Z_=Z_list(N)
    for i in range(N):
        H+=h[i]*Z_[i]
        j=i+1
        while not j==N:
            H+=J[i,j]*X_[i]*X_[j]
            j+=1
    return H
    
    
def H_XX(J,h,N):#QUTIP
    H=0
    X_=X_list(N)

    for i in range(N):
        j=i+1
        while not j==N:
            H+=J[i,j]*X_[i]*X_[j]
            j+=1
    return H    
    

def take_partial_traces(rho_in,N):
  reshaped_rho = rho_in.reshape([2, 2**(N-1), 2, 2**(N-1)])
  rho1 = np.einsum('ijkj->ik', reshaped_rho)
  reduced_rho = np.einsum('jijk->ik', reshaped_rho)
  return np.kron(rho1, reduced_rho)

def evolution_operator(H,dt):
  eigvals, eigvects = la.eigh(H)
  P = eigvects.T
  evol_op = np.dot(P.T, np.dot(np.diag(np.exp(-1j * eigvals * dt)),P))
  return evol_op

def Operator_list(N,func,gamma=1): #QUTIP
    Ens_Op_list=[]
    I=qt.qeye(2)
    gamma=gamma**0.5
    operator=gamma*func()

    for i in range(N):
        Op_list=[]
        for j in range(N):
            Op_list.append(I)
        Op_list[i] = operator
        Ens_Op_list.append(qt.tensor(Op_list))


    return Ens_Op_list

def H_Z(N):#QUTIP
    H=0
    Z_=Z_list(N)
    for i in range(N):
        H+=Z_[i]
    return H

def H_X(N):#QUTIP
    H=0
    X_=X_list(N)
    for i in range(N):
        H+=X_[i]
    return H

def H_Y(N):#QUTIP
    H=0
    Y_=Y_list(N)
    for i in range(N):
        H+=Y_[i]
    return H

def capacity(target,output):
    a=np.corrcoef(output,target)
    return a[0,1]**2


def fit_train_only(x,y,N_training,N_warming=0):
    reg = LinearRegression()
    x_train = x[N_warming:N_training+N_warming]
    y_train = y[N_warming:N_training+N_warming]
    reg.fit(x_train, y_train)
    output=reg.predict(x_train)
    return output,y_train

def capacity_N_grid(x,y,all_N_train,all_N_warm):
    all_capacity = -1*np.ones(shape=(len(all_N_warm),len(all_N_train)))
    for index_N_train,N_train in enumerate(all_N_train):
        for index_N_warm,N_warm in enumerate(all_N_warm):
            all_capacity[index_N_warm,index_N_train]=capacity(*fit_train_only(x,y,N_train,N_warming=N_warm))
    return all_capacity
