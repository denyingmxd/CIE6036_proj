import numpy as np



def orginal_utility(x,theta,b,d):
    if theta>=1./b:
        print('warning: theta too large compared to b')
        exit()
    if x<b:
        return 0
    if x>d:
        return 0
    return theta*(x-b)+1

def get_nd(N,A,b,d,h,H_0):
    upper = 1/6*N*(N+1)*(2*N+1)*d**2+h
    lower = A*b**2+h
    xx=3*(H_0-h)/(2*A*d**2)+1/2*np.sqrt((9*(H_0-h)**2)/(A**2*d**4) - 1/432)
    yy=3*(H_0-h)/(2*A*d**2)-1/2*np.sqrt((9*(H_0-h)**2)/(A**2*d**4) - 1/432)
    y_d = xx**(1/3)+yy**(1/3)-1/2
    # print(y_d)
    # print(upper)
    if H_0<lower:
        return 0
    else:
        return np.floor(y_d)

def get_n_b(N,A,b,h,H_0):
    upper = 1 / 6 * N * (N + 1) * (2 * N + 1) * b ** 2 + h
    lower = A * b ** 2 + h
    xx = 3 * (H_0 - h) / (2 * A * b ** 2) + 1 / 2 * np.sqrt((9 * (H_0 - h) ** 2) / (A ** 2 * b ** 4) - 1 / 432)
    yy = 3 * (H_0 - h) / (2 * A * b ** 2) - 1 / 2 * np.sqrt((9 * (H_0 - h) ** 2) / (A ** 2 * b ** 4) - 1 / 432)
    y_b = xx ** (1 / 3) + yy ** (1 / 3) - 1 / 2
    print(upper)
    if lower<=H_0:
        print(y_b)
        return np.floor(y_b)
    if H_0<lower:
        return 0
    else:
        return N


def allocate_1():
    pass


def max_allocatin(N,H_0,h,A,theta,b,d):
    n_d = get_nd(N,A,b,d,h,H_0)
    # print('max allocation with d:',n_d)
    q_d=-1/2*d*n_d\
        +np.sqrt(
        -1/3*d**2*n_d*(n_d+1)**2*(n_d+2)+4*(n_d+1)*(H_0-h)/A
    )/(2*(n_d+1))

    if q_d<b:
        SW_d = n_d*(1+theta*(d-b))
    else:
        SW_d = n_d*(1+theta*(d-b))+(1+theta*(q_d-b))
    return SW_d


def val_water(j,k,b,d,A,H_0,h):
    fs=[d]*j
    fs.extend([b]*(k-j))
    fs=np.array(fs)
    left=H_0
    for x in range(len(fs)):
        H_ = left-A*(np.sum(fs[x:])**2)
        left = H_
    if left<0:
        return False
    else:
        return True



def optimal_allocation(N, H_0, h, A, theta, b,d,alpha):
    mm = -1
    H_0 = H_0*(1-alpha)
    for j in range(1,N+1):
        for k in range(j,N+1):
            if not val_water(j,k,b,d,A,H_0,h):
                continue
            welfare = j*orginal_utility(d,theta,b,d) + (k-j)*orginal_utility(b,theta,b,d)
            if welfare>mm:
                mm = welfare
                # print(j,k,mm)
    return mm



    pass


def min_allocation(N, H_0, h, A, theta, b):
    bound = 1/6*N*(N+1)*(2*N+1)*A*b**2+h
    # print(bound)
    n_b=get_n_b(N,A,b,h,H_0)
    return n_b

