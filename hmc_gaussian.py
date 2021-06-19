import numpy as np
import matplotlib.pyplot as plt


from matplotlib import rc



def font_stile(a):
	
	rc('font',**{'family':'serif','serif':[a]})
	rc('text', usetex=True)
	
	return True

activated = font_stile('Palatino')
	
def gaussian(mu, sigma):
	
	data = np.random.normal(mu, sigma, 10000)
	
	#for i in range(len(data)):
	##	if data[i] < 0:
	#		data[i] = np.random.normal(0,1,1)
	
	return data	

transition_model = lambda x: np.random.normal(x,[0.05,0.01],(2,))

def prior(w):
    if(w[0]<=0 or w[1] <=0):
        return 0
    else:
        return 1


def manual_log_like_normal(x,data):
    #x[0]=mu, x[1]=sigma (new or current)
    #data = the observation
    val = np.sum(-np.log(x[1] * np.sqrt(2* np.pi) )-((data-x[0])**2) / (2*x[1]**2))
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
	
	grad1  = np.sum( - (data-q[0]) / (q[1]**2))
	
	grad2  = np.sum( 1/q[1] - (data-q[0])**2/ (q[1]**3))
		
	return np.array([grad1, grad2])


def leapfrog(q, p, data, path_len, step_size, massinverse):
	
	for _ in range(path_len):
		
		p =  p  -   0.5*step_size*grad_analitic(data, q)
			
		q =  q  +  step_size*np.matmul(p, massinverse)
		
		#print(np.shape(q), np.shape(np.matmul(p, massinverse)))
			
		p =  p  -   0.5*step_size*grad_analitic(data, q)
		
	#print(step_size*np.sum( - (data-q[0]) / (q[1]**2)))
	return np.array(q), np.array(p)

def kinematic_computer(p, massinverse):

	
	k = 0.5*np.matmul(p, np.matmul(massinverse, p))
	
	#print(k)
	#k = 0.5*(p[0]**2)/mass[0] +  0.5*(p[1]**2)/mass[1] 
	
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
    
    massinverse = np.linalg.inv(mass)  
    
    print('inverse mass',massinverse)
    
    for i in range(iterations):
	  	
	  	pm = np.random.normal(0,1,(2,)) 
	  	
		x_new, p_new  = integrator(x, pm, data, path_len, step, massinverse)
		
		u_lik = -likelihood_computer(x,data)
		
		u_new_lik = -likelihood_computer(x_new,data) 
		
		k_lik = kinematic_computer(pm, massinverse) 
		
		k_new_lik = kinematic_computer(p_new, massinverse) 
		
		#print ('hmc',x_new, p_new)
		
		if (acceptance_h(u_new_lik, u_lik ,k_new_lik , k_lik)):
			x = x_new
			accepted.append(x_new)
			acceptedp.append(p_new)
		else:
			rejected.append(x_new)
			rejectedp.append(p_new)            
                
    return np.array(accepted), np.array(rejected),np.array(acceptedp), np.array(rejectedp)


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

# Mock data
data = gaussian(3, 4)

#plt.hist(data, bins = 100)
#plt.show()

# HMC parameters 
path_len = 6
mass = [[2,1.5],[1.5,2]]
step = 0.003

# MCMC input 
samps_num  = 50000
start = [4,3]

# Metropolis sampling
samples_accepted, samples_rejected = metropolis_hastings(manual_log_like_normal,
									 prior,transition_model,start, samps_num,data,acceptance)

# HMC sampling

samples_accepted_h, samples_rejected_h, samples_accepted_ph, samples_rejected_ph = Hamiltonian_sampler(manual_log_like_normal,
											leapfrog,prior, start, samps_num,data, 
											acceptance,path_len,step , mass)



# Variying paramaters sector
#fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,7))
#fig.suptitle('Correlation')
#for i in range (0,6):
#	print('iteration:', i+1)
	
	#a = (i+1)*0.001
#	a = i*0.25
	
#	print(a)
#	samples_accepted_h, samples_rejected_h, samples_accepted_ph, samples_rejected_ph = Hamiltonian_sampler(manual_log_like_normal,
#											leapfrog,prior, start, samps_num,data, 
#											acceptance, path_len, step , [[2,a],[a,2]] )
#
	#print('iteration:', i+1,(i+1)*0.001)
#	plt.plot(samples_accepted_h[:,0], samples_accepted_h[:,1], label = r'off diag mass element = '+str(a))
#	#plt.plot(samples_accepted_h2[:,0], samples_accepted_h2[:,1], color = 'red')

#	a1,b1,c1 = lag_calculator(samples_accepted_h[:,0],samples_accepted_h[:,0], 100)
#	a2,b2,c2 = lag_calculator(samples_accepted_h[:,1],samples_accepted_h[:,1], 200)

	
	
#	ax1.plot(a1,c1, label = r'Integrations = '+str(i+3))
#	ax2.plot(a2,c2,label = r'Integrations = '+str(i+3))
	

#ax1.legend(loc = 'upper right')
#ax2.legend(loc = 'upper right')
#plt.legend(loc='upper right')
#plt.savefig('off_diag_mass_effect.png')
#plt.show()


# Shapes useful for efficiency

print(np.shape(samples_accepted_h), np.shape(samples_accepted))

# Plotting

# Precontour plots (Not 68% and 95%)

plt.scatter(samples_accepted[:,0], samples_accepted[:,1], color = 'red', marker = '+', label = 'MH')
plt.plot(samples_accepted[:,0], samples_accepted[:,1] , color = 'red')

plt.scatter(samples_accepted_h[:,0], samples_accepted_h[:,1], color = 'blue', marker = '+', label = 'HMC')
plt.plot(samples_accepted_h[:,0], samples_accepted_h[:,1], color = 'blue')

# Reality check params

plt.axvline(start[0], color='black', linestyle = 'dashed')
plt.axhline(start[1], color='black', label = 'Starting', linestyle = 'dashed')

plt.axvline(3, color='red', linestyle = 'dashed')
plt.axhline(4, color='red', label = 'True value', linestyle = 'dashed')
#plt.xlim([2.5, 4.5])
#plt.ylim([2.5, 4.5])

plt.ylabel(r'$\sigma$')
plt.xlabel(r'$\mu$')
plt.legend(loc = 'upper right')
plt.savefig('HMC.png')
plt.close()


#Phase space plotting

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,7))
fig.suptitle('Phase space diagrams')
ax1.scatter(samples_accepted_h[:,0], samples_accepted_ph[:,0], label=r'Accepted samples HMC')
ax2.scatter(samples_accepted_h[:,1], samples_accepted_ph[:,1],label=r'Accepted samples HMC')

ax1.set(xlabel=r'$\mu$', ylabel=r'$P_{\mu}$')
ax2.set(xlabel=r'$\sigma$', ylabel=r'$P_{\sigma}$')

plt.legend(loc = 'upper right')
plt.savefig('HMC_phase.png')
plt.close()


# Correlation lengths


a1,b1,c1 = lag_calculator(samples_accepted[:,0],samples_accepted_h[:,0], 100)
a2,b2,c2 = lag_calculator(samples_accepted[:,1],samples_accepted_h[:,1], 200)

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,7))
fig.suptitle('Correlation')
ax1.plot(a1,b1, label=r'MH' + r'$\mu$')
ax1.plot(a1,c1, label=r'HMC'+ r'$\mu$')
ax1.legend(loc = 'upper right')
ax2.plot(a2,b2 ,label=r'MH'+ r'$\sigma$')
ax2.plot(a2,c2 ,label=r'HMC'+ r'$\sigma$')
ax2.legend(loc = 'upper right')


plt.savefig('correlation_length.png')
plt.close()


## Convergence test: Trace plots

a = 5000
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,7))
fig.suptitle('Traceplots')
ax1.plot(np.arange(a),samples_accepted[:a,0], label=r'MH' + r'$\mu$')
ax1.plot(np.arange(a),samples_accepted_h[:a,0], label=r'HMC'+ r'$\mu$')
ax1.legend(loc = 'upper right')
ax2.plot(np.arange(a),samples_accepted[:a,1] ,label=r'MH'+ r'$\sigma$')
ax2.plot(np.arange(a),samples_accepted_h[:a,1] ,label=r'HMC'+ r'$\sigma$')
ax2.legend(loc = 'upper right')



plt.savefig('TracePlots.png')
plt.close()


