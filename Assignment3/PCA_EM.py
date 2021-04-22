# -*- coding: utf-8 -*-
"""
Created on Tue Nov 2 18:10:40 2020

@author: Nabil
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import multivariate_normal

def load_data(file_name):
    file = open(file_name)
    lines = file.readlines()
    
    dataset = []
    
    for line in lines:
        var = line.split()
        data = []
        for i in range(len(var)):
            data.append(float(var[i]))
        dataset.append(data)
        
    return np.array(dataset)

data = load_data("data_new.txt")
#print(data)

def column_mean(dataset):
    d = np.transpose(dataset)
    #print(d.shape)
    D = d.shape[0]
    N = d.shape[1]
    
    mean = np.zeros(D)
    
    
    for i in range(D):
        sum = 0
        for j in range(N):
            sum = sum + d[i][j]
        mean[i] = float(sum/N)
        
    return mean

a = column_mean(data)
#print(a.shape)
#print(np.dot((data[0] - a[0]), (data[0] - a[0]).T)/100)

def construct_cov_matrix(data, mean):
    N = data.shape[0]
    D = data.shape[1]
    cov = np.zeros((D,D))
    
    for i in range(D):
        for j in range(D):
            s = 0
            for d in data:
                s = s + ((d[i] - mean[i])*(d[j] - mean[j]))
            cov[i][j] = (float(s/N))
            
        
    return cov

#cov = construct_cov_matrix(data, a)
print("cov done")
#print(np.cov(data - a, rowvar=False))


def PCA(data, mean, dim):
    cov = construct_cov_matrix(data, mean)
    h = np.linalg.eigh(cov, UPLO='L')
    eigen_val = h[0]
    eigen_vec = h[1]
    e_val_idx_sorted = np.argsort(eigen_val)
    k = eigen_vec[:,e_val_idx_sorted[len(e_val_idx_sorted) - dim: len(e_val_idx_sorted)]]
    reduced_data = np.dot((data - mean), k)
    return reduced_data
print("PCA done")
dataset = PCA(data, a, 2)
#print("e_val", e_val)
#print("e_vec", e_vec)
#print(k)
x = dataset[:,0]
y = dataset[:,1]
plt.plot(x,y,'.')

def initialize_weights(num_cluster, seed):
    np.random.seed(seed)
    weight = np.random.uniform(0,1, num_cluster)
    w = weight/np.sum(weight)
    #assert (w.shape == (num_cluster))
    return w
print("initial weights: ")
#w = initialize_weights(3, 50)
#print(w)

def initialize_means(num_cluster, dim, seed):
    means = []
    for i in range(num_cluster):
        np.random.seed(seed)
        m = np.random.uniform(-4, 4, dim)
        means.append(m)
        seed = seed + 10
        
    return np.array(means)
print("initial means: ")
#m = initialize_means(3,2,50)
#print(m)

def initialize_cov_mat(num_cluster, dim, seed):
    cov_mat = []
    for i in range(num_cluster):
        np.random.seed(seed)
        m = np.random.uniform(0,6,(dim, dim))
        cov_mat.append(m)
        seed = seed + 10
        
    return np.array(cov_mat)

print("initial covariance matrices: ")
#cov = initialize_cov_mat(3,2,50)
#print(cov)

def compute_probability(x, dim, means, co_var):
    det = np.linalg.det(co_var)
    inv_co_var = np.linalg.inv(co_var)
    x_m = x - means
    inv_x = np.dot(inv_co_var, x_m)
    exp_term = np.exp(-0.5*np.dot(x_m.T, inv_x))
    mul = (1./(((2*np.pi)**float(dim/2))*np.sqrt(abs(det))))*exp_term
    return mul

#print(compute_probability(dataset[0], 2, m[0], cov[0]))
#s = multivariate_normal.pdf(dataset[0], mean=m[0], cov=cov[0])
#print(s)

def E_step(dataset, w, mu, co_var):
    N = len(dataset)
    K = len(w)
    dim = len(dataset[0])
    
    p = np.zeros((N,K))
    
    for i in range(N):
        for k in range(K):
            p[i][k] = w[k]*compute_probability(dataset[i], dim, mu[k], co_var[k])
        p[i] = p[i] / np.sum(p[i])
        
    return p

#pik = E_step(dataset, w, m, cov)
#print(pik)
#print(np.sum(pik, axis = 0))

def update_means(dataset, p):
    N = len(dataset)
    K = len(p[0])
    d = len(dataset[0])
    
    mu = np.zeros((K, d))
    p_sum = np.sum(p, axis = 0)
    
    for k in range(K):
        w_sum = 0
        for i in range(N):
            w_sum = w_sum + p[i][k]*dataset[i]
        mu[k] = w_sum / p_sum[k]
        
    return mu

#mu = update_means(dataset, pik)  
#print("updated mu : ")        
#print(mu) 

def update_weights(dataset, p):
    N = dataset.shape[0]
    p_sum = np.sum(p, axis = 0)
    
    return p_sum/N

#w = update_weights(dataset, pik)
#print("weights updated : ")
#print(w)
#print(pik)
def update_cov(m,p,data,dim):
    cov_mat_k = []
    K = p.shape[1]
    for k in range(K):
        #cvk = []
        for i in range(dim):
            sum_k = 0
            s = 0
            for j in range(len(data)):
                s = s + p[j][k]
                xx = np.matrix(data[j])-np.matrix(m[k])
                x = xx.transpose()
                mul = np.dot(x, x.T)
                sum_k = sum_k + p[j][k]*mul
            
            cov = sum_k/s
        #cvk.append(cov)
        #print(cvk)
        #cv = np.matrix(cvk)
        cov_mat_k.append(cov)
    cv = np.array(cov_mat_k)
    return cv

#cov_up = update_cov(mu,pik,dataset,3,2)
#print('Cov updated')
#print(cov_up)

def calculate_log_likelihood(wt,mean,data,dim,covar):
    log_likelihood = 0
    for i in range(len(data)):
        sum = 0
        for j in range(len(wt)):
            sum = sum + wt[j]*compute_probability(data[i], dim, mean[j], covar[j])
        #print(np.log2(sum))compute_probabilty(dataset[i], dim, mean[j], covar[j])
        log_likelihood = log_likelihood + np.log2(sum)
    return log_likelihood

#print(calculate_log_likelihood(w,mu,dataset,2,cov_up))

def EM_algorithm(data, num_cluster, dim, seed):
    weights = initialize_weights(num_cluster, seed)
    mu = initialize_means(num_cluster, dim, seed)
    cov = initialize_cov_mat(num_cluster, dim, seed)
    log_likelihood = calculate_log_likelihood(weights,mu,data,2,cov)
    cnt = 0
    while True:
        pik = E_step(data, weights, mu, cov)
        mu = update_means(dataset, pik)
        weights = update_weights(dataset, pik)
        cov = update_cov(mu ,pik ,data,2)
        log_ = calculate_log_likelihood(weights,mu,data,2,cov)
        cnt = cnt + 1
        if np.abs(log_likelihood - log_) < 0.005:
            print(cnt)
            break
        print(cnt, " ",log_)
        log_likelihood = log_
    return mu, weights, cov

mu, w, cv = EM_algorithm(dataset, 3, 2, 40)
print("mean : ")
print(mu)
print("weights: ")
print(w)
print("covariance : ")
print(cv)


def clustering_data_points(dataset, mu, cov, K):
    dim = dataset.shape[1]
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for i in range(len(dataset)):
        prob = []
        for j in range(K):
            p = compute_probability(dataset[i], dim, mu[j], cv[j])
            prob.append(p)
        prob = np.array(prob)
        p_mx = np.argmax(prob)
        if p_mx == 0:
            cluster1.append(dataset[i])
        elif p_mx == 1:
            cluster2.append(dataset[i])
        else:
            cluster3.append(dataset[i])
            
    a = np.array(cluster1)
    b = np.array(cluster2)
    c = np.array(cluster3)
    
    x1 = a[:,0]
    y1 = a[:,1]
    plt.plot(x1,y1,'co')
    x2 = b[:,0]
    y2 = b[:,1]
    plt.plot(x2,y2,'go')
    x3 = c[:,0]
    y3 = c[:,1]
    plt.plot(x3,y3,'ro')
    
clustering_data_points(dataset, mu, cv, 3)
    
    
    

    
    
    


    
         