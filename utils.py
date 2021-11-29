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

def square_root_utility(x):
    return np.sqrt(x)

def diff_utility(case,x):
    if case=='sqrt':
        ut = np.sqrt(x)
    if case=='log':
        ut = np.log2(x+1)
    if case=='sqrt3':
        ut = x**(1/3)
    if case=='sqrt4':
        ut = x**(1/4)
    if case=='sqrt5':
        ut = x**(1/5)

    return ut


def get_nd(N,A,b,d,h,H_0):
    upper = 1/6*N*(N+1)*(2*N+1)*d**2+h
    lower = A*b**2+h
    xx=3*(H_0-h)/(2*A*d**2)+1/2*np.sqrt((9*(H_0-h)**2)/(A**2*d**4) - 1/432)
    yy=3*(H_0-h)/(2*A*d**2)-1/2*np.sqrt((9*(H_0-h)**2)/(A**2*d**4) - 1/432)
    # print((9*(H_0-h)**2)/(A**2*d**4) - 1/432)
    # print(yy)
    if yy<0.0000001:
        yy=0
        y_d = xx ** (1 / 3) + yy ** (1 / 3) - 1 / 2
    else:
        y_d = xx**(1/3)+yy**(1/3)-1/2
    # print(d,y_d,xx,yy)
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
    # print(upper)
    if lower<=H_0 and H_0<=upper:
        # print(y_b)
        return np.floor(y_b)
    if H_0<lower:
        return 0
    if H_0>upper:
        return N


def algorithm_one(H_0, A, h, x_list):
    num = len(x_list)
    result = []
    for i in range(num):
        claim = x_list[i]
        # for j in range(claim, 0, 0.0001):
        # print(claim)
        while claim >= 0:
            current_cost = 0
            for k in range(i+1):
                total_sum = claim
                for m in range(k,i):
                    total_sum += result[m]
                # print(total_sum)
                current_cost += A * total_sum**2
            # print(current_cost)

            if current_cost <= H_0 - h:
                result.append(claim)
                break
            claim -= 0.001
        if len(result) != i+1:
            break
            # result.append(0)
        # print(result)
    current_len = len(result)
    for i in range(current_len,num):
        result.append(0)
    # print(len(result))
    return result



def I_2(q_d,b):
    if q_d<b:
        return 0
    else:
        return 1


def get_square_root_usage(demmand,H_0,N,h,A):
    n_d = get_nd(N, A, 0, demmand, h, H_0)

    q_d = -1 / 2 * demmand * n_d \
          + np.sqrt(
        -1 / 3 * demmand ** 2 * n_d * (n_d + 1) ** 2 * (n_d + 2) + 4 * (n_d + 1) * (H_0 - h) / A
    ) / (2 * (n_d + 1))
    usages=[demmand]*int(n_d)
    print(usages)
    usages.extend([q_d])
    while len(usages)<=N:
        usages.append(0)
    return usages

def polynomial_pricing_allocation(N, H_0, h, A, theta, b, d,heterogeneou=False):
    SW=[]
    pi_f = lambda f: 1+theta*(f-b)-m*f**2 if f>=b and f<=d else 0
    ms = range(100,10000,1)
    for m in ms:
        eq = theta/2./m
        if eq<=b:
            pi_b = pi_f(eq)
            if pi_b>0:
                demmand=b
            else:
                demmand=0
        elif eq>b and eq<d:
            pi_eq = pi_f(eq)
            if pi_eq>0:
                demmand=eq
            else:
                demmand=0
        else:
            pi_d = pi_f(d)
            if pi_d > 0:
                demmand = d
            else:
                demmand = 0
        if demmand==0:
            SW.append(0)
            continue
        aaa = flow_restrict_allocation(N, H_0, h, A, theta, b,d,demmand)
        SW.append(aaa)
        # print(m,aaa,demmand)
    optimal = max(SW)
    return optimal

def poly_demmands(N, H_0, h, A, theta1,theta2, b, d):
    SW=[]
    pi_f_1 = lambda f: 1+theta1*(f-b)-m*f**2 if f>=b and f<=d else 0
    pi_f_2 = lambda f: 1+theta2*(f-b)-m*f**2 if f>=b and f<=d else 0
    ms = range(1,10000,10)
    demands=[]
    for m in ms:
        eq1 = theta1/2./m
        eq2 = theta2/2./m
        if eq1<=b:
            pi_b = pi_f_1(eq1)
            if pi_b>0:
                demmand1=b
            else:
                demmand1=0
        elif eq1>b and eq1<d:
            pi_eq = pi_f_1(eq1)
            if pi_eq>0:
                demmand1=eq1
            else:
                demmand1=0
        else:
            pi_d = pi_f_1(d)
            if pi_d > 0:
                demmand1 = d
            else:
                demmand1 = 0
       ##---------------------------
        if eq2<=b:
            pi_b = pi_f_2(eq2)
            if pi_b>0:
                demmand2=b
            else:
                demmand2=0
        elif eq2>b and eq2<d:
            pi_eq = pi_f_2(eq2)
            if pi_eq>0:
                demmand2=eq2
            else:
                demmand2=0
        else:
            pi_d = pi_f_2(d)
            if pi_d > 0:
                demmand2 = d
            else:
                demmand2 = 0
        # print(m,aaa,demmand)
        demands.append([demmand1,demmand2]*5)
    return demands


def polynomial_mn_pricing_allocation(N, H_0, h, A, theta, b, d, return_pos=False):
    SW=[]
    comb=[]
    pi_f = lambda f: 1+theta*(f-b)-(m*f**2+n*f) if f>=b and f<=d else 0
    for m in range(100,10000,10):
        for n in range(0,10):
            eq = (theta-n)/(2*m)
            if eq<=b:
                pi_b = pi_f(eq)
                if pi_b>0:
                    demmand=b
                else:
                    demmand=0
            elif eq>b and eq<d:
                pi_eq = pi_f(eq)
                if pi_eq>0:
                    demmand=eq
                else:
                    demmand=0
            else:
                pi_d = pi_f(d)
                if pi_d > 0:
                    demmand = d
                else:
                    demmand = 0
            if demmand==0:
                SW.append(0)
                continue
            aaa = flow_restrict_allocation(N, H_0, h, A, theta, b,d,demmand)
            SW.append(aaa)
            comb.append([m,n])
        # print(m,aaa,demmand)
    optimal = max(SW)
    ind = np.argmax(SW)
    optimal_base = comb[ind]
    # print(optimal_base)
    if return_pos:
        return optimal,optimal_base
    return optimal

def polynomial_diff_mn_pricing_allocation(N, H_0, h, A, theta, b, d,ratio):
    SW=[]
    pi_f = lambda f: 1+theta*(f-b)-(m*f**2+n*f) if f>=b and f<=d else 0
    for m in range(100,10000,10):
        n = m*ratio
        eq = (theta-n)/(2*m)
        if eq<=b:
            pi_b = pi_f(eq)
            if pi_b>0:
                demmand=b
            else:
                demmand=0
        elif eq>b and eq<d:
            pi_eq = pi_f(eq)
            if pi_eq>0:
                demmand=eq
            else:
                demmand=0
        else:
            pi_d = pi_f(d)
            if pi_d > 0:
                demmand = d
            else:
                demmand = 0
        if demmand==0:
            SW.append(0)
            continue
        aaa = flow_restrict_allocation(N, H_0, h, A, theta, b,d,demmand)
        SW.append(aaa)
        # print(m,aaa,demmand)
    optimal = max(SW)
    return optimal






def pricing_allocation(N,H_0,h,A,theta,b,d):
    n_b=get_n_b(N,A,b,h,H_0)
    n_d =get_nd(N,A,b,d,h,H_0)
    print(n_d,n_b)
    q_d=-1/2*d*n_d\
        +np.sqrt(
        -1/3*d**2*n_d*(n_d+1)**2*(n_d+2)+4*(n_d+1)*(H_0-h)/A
    )/(2*(n_d+1))
    if theta<(n_b-n_d-I_2(q_d,b))/((n_d*(d-b)+max(q_d-b,0))):
        print('pricing using max allocation and p in range [{},{}]'.format(theta,1/b))
        SW_p=n_b
    else:
        print('pricing using min allocation and p in range [{},inf]'.format(theta))
        SW_p=n_d*(1+theta*(d-b))+I_2(q_d,b)+theta*max(q_d-b,0)
    return SW_p



def flow_restrict_allocation(N, H_0, h, A, theta, b,d,c):
    n_c = get_nd(N, A, b, c, h, H_0)
    # print('max allocation with c:',n_c)
    q_c = -1 / 2 * c * n_c \
          + np.sqrt(
        -1 / 3 * c ** 2 * n_c * (n_c + 1) ** 2 * (n_c + 2) + 4 * (n_c + 1) * (H_0 - h) / A
    ) / (2 * (n_c + 1))

    if n_c>=N:
        SW_c=N*(1+theta*(c-b))
    if n_c<N and q_c<b:
        SW_c = n_c * (1 + theta * (c - b))
    if n_c<N and q_c>=b:
        SW_c=n_c*(1+theta*(c-b))+(1+theta*(q_c-b))
    return SW_c




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



def optimal_allocation(N, H_0, h, A, theta, b,d):
    mm = -1
    H_0 = H_0
    for j in range(1,N+1):
        for k in range(j,N+1):
            if not val_water(j,k,b,d,A,H_0,h):
                continue
            welfare = j*orginal_utility(d,theta,b,d) + (k-j)*orginal_utility(b,theta,b,d)
            if welfare>mm:
                mm = welfare
                # print(j,k,mm)
    return mm





def min_allocation(N, H_0, h, A, theta, b):
    bound = 1/6*N*(N+1)*(2*N+1)*A*b**2+h
    # print(bound)
    n_b=get_n_b(N,A,b,h,H_0)
    return n_b

def get_demmand(case,p):
    if case=='sqrt':
        demmand = 1/(4*p**2)
    if case=='log':
        demmand = 1./p-1
    if case=='sqrt3':
        demmand = (3*p)**(-3/2)
    if case=='sqrt4':
        demmand = (4*p)**(-4/3)
    if case=='sqrt5':
        demmand = (5*p)**(-5/4)

    return demmand
