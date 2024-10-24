import os
import sys
import time
import numpy as np
import qutip as qt
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
sys.path.insert(0, '..')
from functions import *
import ray



def fit(x, y, N_training, N_warming=0):
    reg = LinearRegression()
    x_train = x[N_warming:N_training + N_warming]
    y_train = y[N_warming:N_training + N_warming]
    reg.fit(x_train, y_train)

    x_test = x[N_training + N_warming:]
    y_test = y[N_training + N_warming:]

    output = reg.predict(x_test)
    return output, y_test


def is_pos_def(x): 
    eig = np.linalg.eigvals(x)	
    for i in range(len(eig)):
    	if eig[i] < 0:
    	   print(eig[i])
    	   return False	  
    return True


def operator_list_full(N, operator, func): #QUTIP
    Ens_Op_list=[]
    I=qt.qeye(2)
 
    if func=="spin":
    	for i in range(N):
            Op_list=[]
            for j in range(2*N):
            	Op_list.append(I)
            Op_list[i] = operator
            Ens_Op_list.append(qt.tensor(Op_list))
    elif func=="reservoir":
    	for i in range(N,2*N):
            Op_list=[]
            for j in range(2*N):
            	Op_list.append(I)
            Op_list[i] = operator
            Ens_Op_list.append(qt.tensor(Op_list))
    
    return Ens_Op_list


def operator_list(N,func): #QUTIP
    Ens_Op_list=[]
    I=qt.qeye(2)
    operator=func
    for i in range(N):
        Op_list=[]
        for j in range(N):
            Op_list.append(I)
        Op_list[i] = operator
        Ens_Op_list.append(qt.tensor(Op_list))
    return Ens_Op_list





def Bx_dyn(seed,hs,dt,gamma,N):
    #global gammaglobal
    #gammaglobal = gamma
    Js = 1.0
    V = 2
    Tm=1
        
    
    Observables = operator_list(N, qt.sigmaz())
    
    L = 4000
    
    np.random.seed(seed)
    J = J_all2all_het(Js, N)
    h = h_hom(hs, N)
    H_ising = H_XXZ(J, h, N)
        
        
    data=np.zeros((Tm*len(Observables),L),dtype=np.complex128)
    options = {"store_states": True, "nsteps": 100000}
    rho1 = qt.Qobj(qt.rand_dm(2**(N)),dims=[[2]*(N),[2]*(N)])
    
    s_k=np.zeros(L)

    for ind in range(L):
        s_k[ind] = np.random.random_sample()
        
    H_x = H_X(N)
    Sminus = Operator_list(N,qt.sigmam,gamma=gamma)
    data = np.zeros((Tm * len(Observables), L), dtype=np.complex128)

    for i in range(L):

        t = np.linspace(0, dt, V)
        H=H_ising+(s_k[i]+1)*hs*H_x
        
        
        result = qt.mesolve(H, rho1, t, c_ops=Sminus, e_ops=Observables, options=options)
        rho1 = qt.Qobj(result.final_state, dims=[[2] * (N), [2] * (N)])
        
        for k in range(len(Observables)):
                data[k][i] = result.expect[k][V-1]
        

    tau_max=25
    perf_l=[]
    for taun in range(tau_max) : 
 
            x_n=np.zeros((L-taun,data.shape[0]))
            for i in range(L-taun):              
              for j in range(data.shape[0]) :  
                x_n[i,j]=np.real(data[j,i+taun])

            out, test = fit(x_n,s_k[:L-taun],1000,1000)
            cap=capacity(out,test)
            perf_l.append(cap)
    
    perf_l=np.asarray(perf_l)
    print("Linear performances :")
    print(perf_l)
    #---------------- Narma---------------------
    tau_max=25
    perf_N=[]
    for n in range(tau_max) :
 
      y=np.zeros(L)
      s=np.zeros(L)
      for i in range(L):
         s[i]=0.02*s_k[i]
      for k in range(n,L):
        somma=0
        for j in range(n) :
            somma=somma+y[k-j-1]
        y[k]=0.3*y[k-1]+0.05*y[k-1]*somma+1.5*s[k-n]*s[k-1]+0.1
    
      x_n=np.zeros((L-n,data.shape[0]))
      for i in range(L-n):
        for j in range(data.shape[0]) :  
                x_n[i,j]=np.real(data[j,i+n])
      out, test = fit(x_n,y[n:],1000,1000)
      cap=capacity(out,test)
      perf_N.append(cap)
       
    #filenameN = "perfNV25/seed_"+str(float(seed))+"_gamma"+str(float(gamma))+"_dT"+str(float(dT))+"_hs"+str(float(hs))+".npy" 
    perf_N=np.asarray(perf_N)
    print("Narma performances :")
    print(perf_N)   
  
        
    filename2 = "perfL_local/seed_" + str(float(seed)) + "_dT" + str(float(dt)) + "_hs" + str(float(hs))+ "_gamma" + str(float(gamma))+ "_N"+str(float(N)) +".npy"
    filename3 = "perfN_local/seed_" + str(float(seed)) + "_dT" + str(float(dt)) + "_hs" + str(float(hs))+ "_gamma" + str(float(gamma))+ "_N"+str(float(N)) +".npy"
    
    #filename4 = "Linear_quality/seed_" + str(float(seed)) + "_gamma" + str(float(gamma)) + "_dT" + str(float(dT)) + "_hs" + str(float(hs)) + ".npy"
    #return filename, data, filename2, perf_l, filename3, perf_N  
    return filename2, perf_l, filename3, perf_N
    

   
 
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("hs", type=float)
    parser.add_argument("dt", type=float)
    parser.add_argument("gamma", type=float)
    parser.add_argument("N", type=int)


    args = parser.parse_args()

    for i in range(args.seed * 10, (args.seed + 1) * 10 + 1):
        # Call main
        filename, a, filename2, b = Bx_dyn(args.seed, args.hs, args.dt, args.gamma, args.N)

        # Save data
        open(os.path.join(os.path.abspath("."), filename), 'a').close()
        np.save(os.path.join(os.path.abspath("."), filename), a)

        open(os.path.join(os.path.abspath("."), filename2), 'a').close()
        np.save(os.path.join(os.path.abspath("."), filename2), b)

