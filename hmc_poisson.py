import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc



def font_stile(a):
	
	rc('font',**{'family':'serif','serif':[a]})
	rc('text', usetex=True)
	
	return True

activated = font_stile('Palatino')
	
	
def poisson(a):
	data = np.random.poisson(lam=a, size=10000)
	return data	

transition_model = lambda x: np.random.normal(x,0.05,(1,))

def prior2(w):
    if(w[0]<=0 or w[1] <=0):
        return 0
    else:
        return 1
def prior(w):
    if(w<=0 ):
        return 0
    else:
        return 1



def manual_log_like_normal(x,data):
    
    n = len(data)
    
    val = np.sum(data)*np.log(x) - n*x- np.sum(data*np.log(data)-data) ## Stirling approximation
    return val



#Defines whether to accept or reject the new sample
def acceptance(x, x_new):
	accept=np.random.uniform(0,1)
	
	return (accept < (np.exp(x_new-x)))


def metropolis_hastings(likelihood_computer,prior, transition_model, param_init,iterations,data,acceptance_rule):
    x = param_init
    accepted = []
    rejected = []   
   
    for i in range(iterations):
       
        x_new =  transition_model(x)     
        
        x_lik = likelihood_computer(x,data)
        
        x_new_lik = likelihood_computer(x_new,data) 
        
        if (acceptance_rule(x_lik + np.log(prior(x)),x_new_lik+np.log(prior(x_new)))):            
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)            
                
    return np.array(accepted), np.array(rejected)


def grad_analitic(data, q):
	n= len(data)
	grad1  = -np.sum(data)/q + n  #It does not matter if I used the stirling appoximation or not
	
	return grad1


def leapfrog(q, p, data, path_len, step_size, massinverse):
	
	for _ in range(path_len):
		
		p =  p  -   0.5*step_size*grad_analitic(data, q)
			
		q =  q  +  step_size*np.matmul(p, massinverse)
		
		#print(np.shape(q), np.shape(np.matmul(p, massinverse)))
			
		p =  p  -   0.5*step_size*grad_analitic(data, q)
		
	#print(step_size*np.sum( - (data-q[0]) / (q[1]**2)))
	return np.array(q), np.array(-p)

def kinematic_computer(p, massinverse):

	
	k = 0.5*(p**2)*massinverse #+  0.5*(p[1]**2)/mass[1] 
	
	return k

def acceptance_h(Up, Uc, Kp, Kc):
	
	val = np.exp( Uc - Up - Kp + Kc )
	
	accept=np.random.uniform(0,1)

	return (accept < val)

def Hamiltonian_sampler(likelihood_computer,integrator, prior, 
						param_init_real,iterations,data,
						acceptance_rule,path_len,step,mass):
   
    x = param_init_real
    
    accepted = []
    rejected = [] 
    
    acceptedp = []
    rejectedp = []  
    
    massinverse = np.array([1./mass])
    
    print('inverse mass',massinverse)
    
    for i in range(iterations):
	  	
	  	
	  	pm = np.random.normal(0,1,(1,)) 
	  	
		x_new, p_new  = integrator(x, pm, data, path_len, step, massinverse)
		
		u_lik = -likelihood_computer(x,data)
		
		u_new_lik = -likelihood_computer(x_new,data) 
		
		k_lik = kinematic_computer(pm, massinverse) 
		
		k_new_lik = kinematic_computer(p_new, massinverse) 
		
		
		
		if (acceptance_h(u_new_lik, u_lik ,k_new_lik , k_lik)):
			x = x_new
			accepted.append(x_new)
			acceptedp.append(p_new)
		else:
			rejected.append(x_new)
			rejectedp.append(p_new)            
                
    return np.array(accepted), np.array(rejected),np.array(acceptedp), np.array(rejectedp)


def orbits (p0,x, data, path_len, step, mass, integrator,likelihood_computer):
	
	q_val = [] 
	p_val = [] 
    
	
	massinverse = np.array([1./mass])
	
	q0 = x
	
	step 
	
	for _ in range(path_len):
		
		
		pi =  p0  -   0.5*step*grad_analitic(data, q0)
			
		q =  q0  +  step*pi/mass
		
		p =  pi  -   0.5*step*grad_analitic(data, q)

		
		#print(kinematic_computer(p, massinverse) - likelihood_computer(q, data) )
		
		#print(q, p )
		
		p_val.append(p)
		q_val.append(q)
		
		q0 = q
		p0 = p
		
	energy = kinematic_computer(p, massinverse) - likelihood_computer(q, data)
	
	return energy, np.array(p_val), np.array(q_val)


def lag_calculator(aray1, aray2, lagmax):
	mean1 = np.mean(aray1)
	mean2 = np.mean(aray2)
	
	sigma1 =0.
	sigma2 =0.
	
	for m in range (0, len(aray1)):	
		
			sigma1 += (aray1[m]  - mean1 )*(aray1[m]  - mean1 )	
			sigma2 += (aray2[m] - mean2 )*(aray2[m] - mean2 )	
			
	lag1 = np.zeros(lagmax)
	lag2 = np.zeros(lagmax)
		
	for k in range(0, lagmax):
				
		for l in range (0, len(aray1) - k):
		
			lag1[k] += (aray1[l]   - mean1 )*(aray1[l+k]   - mean1 )/sigma1
			lag2[k] += (aray2[l]  - mean2 )*(aray2[l+k]  - mean2 )/sigma2
	
	index = np.arange(lagmax)
		
	return index, lag1, lag2



data = poisson(200)

# HMC parameters 
path_len = 6
mass = 1 #[[2,1.5],[1.5,2]]
step = 0.003

# MCMC input 
samps_num  = 50000
start = 199#[4,3]

# HMC orbits shape


E1,p_val, q_val = orbits (2,start, data, 5000, 0.01, mass, leapfrog, manual_log_like_normal)
E2,p_val2, q_val2 = orbits (2*2,start, data, 5000, 0.01, mass, leapfrog, manual_log_like_normal)
E3,p_val3, q_val3 = orbits (4*2,start, data, 5000, 0.01, mass, leapfrog, manual_log_like_normal)

fig, (ax1) = plt.subplots(1, 1,figsize=(10,7))
fig.suptitle('Orbit diagram')
ax1.plot(q_val[:], p_val[:], label=r'Orbit energy'+ str(np.round(E1,2)))
ax1.plot(q_val2[:], p_val2[:], label=r'Orbit energy'+ str(np.round(E2,2)))
ax1.plot(q_val3[:], p_val3[:], label=r'Orbit energy'+ str(np.round(E3,2)))


ax1.set(xlabel=r'$\lambda$', ylabel=r'$P_{\lambda}$')

plt.legend(loc = 'upper right')
plt.savefig('Orbits.png')
plt.close()

print('orbits', np.shape(p_val), np.shape(q_val))

# Metropolis sampling

samples_accepted, samples_rejected = metropolis_hastings(manual_log_like_normal,
									 prior,transition_model,start, samps_num,data,acceptance)

# HMC sampling

samples_accepted_h, samples_rejected_h, samples_accepted_ph, samples_rejected_ph = Hamiltonian_sampler(manual_log_like_normal,
											leapfrog,prior, start, samps_num,data, 
											acceptance,path_len,step , mass)



print(np.shape(samples_accepted_h), np.shape(samples_accepted))



plt.hist(samples_accepted[:20000], bins = 100, density=True, label ='Poisson MH')
plt.hist(samples_accepted_h[:20000], bins = 100, density=True, label = 'Poisson HMC')

plt.axvline(200, color='black', linestyle = 'dashed', label='True value')
plt.axvline(199, color='red', linestyle = 'dashed', label='Start value')
plt.xlabel(r'$\lambda$')
plt.legend(loc = 'upper left')
plt.savefig('HMC.png')
plt.close()


#Phase space plotting

fig, (ax1) = plt.subplots(1, 1,figsize=(10,7))
fig.suptitle('Phase space diagrams')
ax1.scatter(samples_accepted_h[:], samples_accepted_ph[:], label=r'Accepted samples HMC')


ax1.set(xlabel=r'$\lambda$', ylabel=r'$P_{\lambda}$')

plt.legend(loc = 'upper right')
plt.savefig('HMC_phase.png')
plt.close()


# Correlation lengths


a1,b1,c1 = lag_calculator(samples_accepted[:],samples_accepted_h[:], 300)


fig, (ax1) = plt.subplots(1, 1,figsize=(10,7))
fig.suptitle('Correlation')
ax1.plot(a1,b1, label=r'MH' + r'$\lambda$')
ax1.plot(a1,c1, label=r'HMC'+ r'$\lambda$')
ax1.legend(loc = 'upper right')

plt.savefig('correlation_length.png')
plt.close()


## Convergence test: Trace plots

a = 20000
fig, (ax1) = plt.subplots(1, 1,figsize=(10,7))
fig.suptitle('Traceplots')
ax1.plot(np.arange(a),samples_accepted[:a], label=r'MH' + r'$\lambda$')
ax1.plot(np.arange(a),samples_accepted_h[:a], label=r'HMC'+ r'$\lambda$')
ax1.legend(loc = 'upper right')



plt.savefig('TracePlots.png')
plt.close()


