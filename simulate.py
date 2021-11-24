import numpy as np
from utils import orginal_utility,algorithm_one,\
    max_allocatin,min_allocation,optimal_allocation,pricing_allocation,\
    flow_restrict_allocation,square_root_utility,get_square_root_usage,\
    get_demmand,diff_utility,get_nd,polynomial_pricing_allocation,\
    polynomial_mn_pricing_allocation, polynomial_diff_mn_pricing_allocation
import matplotlib.pyplot as plt


def test_min_allocation():
    N = 3
    H_0 = 118
    h = 30
    A = 100
    theta = 4
    base = np.arange(0.2,3,0.01)
    rr = []
    for b in base:
        res = min_allocation(N, H_0, h, A, theta, b)
        rr.append(res)
    plt.plot(base,rr)
    plt.show()
    return 0

def test_max_allocation():
    N = 3
    H_0 = 118
    h = 30
    A = 100
    theta = 4
    b = 0.2
    rr = []
    base=np.arange(b,1,0.01)
    for d in base:
        res = max_allocatin(N, H_0, h, A, theta, b,d)
        rr.append(res)
    plt.plot(base,rr)
    plt.title('max allocation')
    plt.show()

    return 0

def test_optimal_allocation():
    N = 40
    H_0 = 251.4
    h = 30
    A = 100
    theta = 4
    b = 0.001
    d = 0.01
    alphas = np.arange(0.0,0.8,0.01)
    alpha=0.1
    res = optimal_allocation(N, H_0, h, A, theta, b,d,alpha)
    print(res)
    return 0


def compare_SWs_and_SWp():
    N = 40
    H_0 = 251.5
    h = 30
    A = 100
    theta = 10
    b = 0.001
    d = 0.01
    alphas = np.arange(0.0,0.85,0.01)
    max_ress=[]
    optimal_ress=[]
    pricing_ress=[]
    for alpha in alphas:
        max_res = max_allocatin(N, H_0*(1-alpha), h, A, theta, b,d)
        optimal_res = optimal_allocation(N, H_0*(1-alpha), h, A, theta, b,d)
        pricing_res = pricing_allocation(N, H_0*(1-alpha), h, A, theta, b,d)
        print(max_res)
        print(optimal_res)
        print(pricing_res)
        # exit()
        pricing_ress.append(pricing_res)
        max_ress.append(max_res)
        optimal_ress.append(optimal_res)
    max_ress=np.array(max_ress)
    optimal_ress=np.array(optimal_ress)
    plt.plot(alphas,max_ress/optimal_ress)
    plt.plot(alphas,pricing_ress/optimal_ress)
    plt.show()
    return 0


def test_flow_restrict_allocation():
    N=3
    H_0=118
    h=30
    A=100
    theta=4
    b=0.2
    d=1
    cs=np.arange(b,d,0.01)
    rr=[]
    for c in cs:
        res = flow_restrict_allocation(N, H_0, h, A, theta, b,d,c)
        rr.append(res)
    plt.plot(cs,rr)
    plt.show()
    return



def test_supply_restrict_allocation():
    N=10
    A=100
    H_0=131
    h=30
    b=0.01
    d=0.1
    theta1=1
    theta2s=np.arange(1,100,1)
    for theta2 in theta2s:
        pass


def test_square_root_utility():
    N=3
    H_0=200
    h=30
    A=100
    prices=np.arange(0.1,1.5,0.01)
    uts=[]
    for p in prices:
        demmand = 1/(4*p**2)
        # usages=get_square_root_usage(demmand,H_0,N,h,A)
        usages=algorithm_one(H_0,A,h,[demmand]*N)
        utility = np.sum([square_root_utility(x) for x in usages])
        uts.append(utility)
    plt.plot(prices,uts)
    plt.show()


def test_fig18():
    H_0=130
    h=30
    A=100
    ns=np.arange(10,110,10)
    prices=np.arange(0.1,20,0.1)
    aa=[]
    bb=[]
    for n in ns:
        uts = []
        for p in prices:
            demmand = 1 / (4 * p ** 2)
            # usages=get_square_root_usage(demmand,H_0,N,h,A)
            usages = algorithm_one(H_0, A, h, [demmand] * n)
            utility = np.sum([square_root_utility(x) for x in usages])
            uts.append(utility)
        SW_p=max(uts)
        ##SW_s
        SW_s=((H_0-h)/A)**(1/4)
        aa.append(SW_p)
        bb.append(SW_s)
        print(aa)
        print(bb)
    plt.plot(ns,np.array(aa)/np.array(bb))
    plt.show()





def test_diff_utility():
    N = 5
    H_0 = 200
    h = 30
    A = 100
    prices = np.arange(0.1, .8, 0.01)
    cases=['sqrt','sqrt3','sqrt4','sqrt5','log']
    uts = [[] for i in range(len((cases)))]
    for p in prices:
        # demmand = 1 / (4 * p ** 2)
        for i,case in enumerate(cases):
            demmand = get_demmand(case,p)
            # usages=get_square_root_usage(demmand,H_0,N,h,A)
            usages = algorithm_one(H_0, A, h, [demmand] * N)
            utility = np.sum([diff_utility(case,x) for x in usages])
            uts[i].append(utility)
    for i in range(len(uts)):
        plt.plot(prices, uts[i],label=cases[i])
    plt.legend()
    plt.show()


def test_Fig_10(extend=False):
    A=100
    H_0=131
    h=30
    b=0.01
    ds=[0.07,0.1]
    # d=0.07
    qq=[]
    N=10
    theta1=1
    theta2s=np.arange(1,100,1)
    if extend:
        theta2s=np.arange(0,1,0.01)

    for d in ds:
        cs = np.arange(b, d, 0.002)
        pricing_uts = []
        resctrict_uts = []
        for j in range(len(theta2s)):
            theta2=theta2s[j]
            customer_demands = [b, d] * (N // 2)
            thetas=[theta1,theta2]*(N//2)
            allocated = algorithm_one(H_0,A,h,customer_demands)
            utility = np.sum([orginal_utility(allocated[i],thetas[i],b,d)for i in range(len(allocated))])
            pricing_uts.append(utility)

            rr = []
            for c in cs:
                n_c = int(get_nd(N, A, b, c, h, H_0))
                # print('max allocation with c:',n_c)
                q_c = -1 / 2 * c * n_c \
                      + np.sqrt(
                    -1 / 3 * c ** 2 * n_c * (n_c + 1) ** 2 * (n_c + 2) + 4 * (n_c + 1) * (H_0 - h) / A
                ) / (2 * (n_c + 1))
                res=0
                print(n_c)
                if n_c>=N:
                    for i in range(N):
                        if i%2==0:
                            res+=1+theta1*(c-b)
                        else:
                            res += 1 + theta2 * (c - b)
                else:
                    for i in range(n_c):
                        if i%2==0:
                            res+=1+theta1*(c-b)
                        else:
                            res += 1 + theta2 * (c - b)
                    if n_c%2==0:
                        if q_c<b:
                            res+=0
                        else:
                            res += 1 + theta1 * (q_c - b)
                    else:
                        if q_c<b:
                            res+=0
                        else:
                            res += 1 + theta2 * (q_c - b)
                rr.append(res)
            optimal_restrict=max(rr)
            resctrict_uts.append(optimal_restrict)
        resctrict_uts=np.array(resctrict_uts)
        pricing_uts=np.array(pricing_uts)
        quotient = resctrict_uts/pricing_uts
        qq.append(quotient)
    plt.plot(theta2s,qq[0],label='d=0.07')
    plt.plot(theta2s,qq[1],label='d=0.1')
    plt.legend()
    plt.show()



def test_Fig_19():
    N=2
    H_0=130
    h=30
    A=100
    v1=1
    v2s=np.arange(1,20,1)
    restriction_uts=[]

    pricing_uts=[]
    cs=np.arange(0.1,7,0.1)
    for v2 in v2s:
        ##flow restriction
        rr = []
        optimal_restrict=[]
        for c in cs:
            n_c = int(get_nd(N, A, 0, c, h, H_0))
            # print('max allocation with c:',n_c)
            q_c = -1 / 2 * c * n_c \
                  + np.sqrt(
                -1 / 3 * c ** 2 * n_c * (n_c + 1) ** 2 * (n_c + 2) + 4 * (n_c + 1) * (H_0 - h) / A
            ) / (2 * (n_c + 1))
            res = 0
            if n_c >= N:
                for i in range(N):
                    if i % 2 == 0:
                        res += v1*np.sqrt(c)
                    else:
                        res += v2*np.sqrt(c)
            else:
                for i in range(n_c):
                    if i % 2 == 0:
                        res += v1*np.sqrt(c)
                    else:
                        res += v2*np.sqrt(c)
                if n_c%2==0:
                    res += v1 * np.sqrt(c)
                else:
                    res += v2 * np.sqrt(c)
            rr.append(res)

        ##pricing

        prices = np.arange(0.1, 5, 0.1)
        uts = []
        for p in prices:
            utility=0
            demmands = [v1 / (4 * p ** 2),v2/(4*p**2)]*(N//2)
            # usages=get_square_root_usage(demmand,H_0,N,h,A)
            usages = algorithm_one(H_0, A, h, demmands)
            for i in range(len(usages)):
                if i % 2 == 0:
                    utility += v1 * np.sqrt(usages[i])
                else:
                    utility += v2 * np.sqrt(usages[i])
            uts.append(utility)
        optimal_pricing_res = max(uts)
        optimal_restrict = max(rr)
        restriction_uts.append(optimal_restrict)
        pricing_uts.append(optimal_pricing_res)
    restriction_uts=np.array(restriction_uts)
    pricing_uts=np.array(pricing_uts)
    plt.plot(v2s,restriction_uts/pricing_uts)
    plt.show()



def test_polynomial_pricing():
    N = 40
    H_0 = 251.5
    h = 30
    A = 100
    theta = 10
    b = 0.001
    d = 0.01
    alphas = np.arange(0.0, 0.85, 0.01)
    max_ress = []
    optimal_ress = []
    pricing_ress = []
    polynomial_ress = []
    polynomial_mn_ress = []
    for alpha in alphas:
        max_res = max_allocatin(N, H_0 * (1 - alpha), h, A, theta, b, d)
        optimal_res = optimal_allocation(N, H_0 * (1 - alpha), h, A, theta, b, d)
        pricing_res = pricing_allocation(N, H_0 * (1 - alpha), h, A, theta, b, d)
        polynomial_res = polynomial_pricing_allocation(N, H_0 * (1 - alpha), h, A, theta, b, d)
        polynomial_mn_res = polynomial_mn_pricing_allocation(N, H_0 * (1 - alpha), h, A, theta, b, d)
        pricing_ress.append(pricing_res)
        max_ress.append(max_res)
        optimal_ress.append(optimal_res)
        polynomial_ress.append(polynomial_res)
        polynomial_mn_ress.append(polynomial_mn_res)
    max_ress = np.array(max_ress)
    optimal_ress = np.array(optimal_ress)
    polynomial_ress = np.array(polynomial_ress)
    polynomial_mn_ress = np.array(polynomial_mn_ress)
    plt.plot(alphas, max_ress / optimal_ress)
    plt.plot(alphas, pricing_ress / optimal_ress)
    plt.plot(alphas, polynomial_ress / optimal_ress)
    plt.plot(alphas, polynomial_mn_ress / optimal_ress)
    plt.show()
    return 0

def test_polynomial_pricing_diff_mn():
    N = 40
    H_0 = 251.5
    h = 30
    A = 100
    theta = 10
    b = 0.001
    d = 0.01
    alphas = np.arange(0.0, 0.85, 0.01)
    optimal_ress = []
    n_m_ratio=np.arange(0,1./100,0.0001)
    results=[[] for i in range(len(n_m_ratio))]
    for alpha in alphas:
        optimal_res = optimal_allocation(N, H_0 * (1 - alpha), h, A, theta, b, d)
        print(alpha)
        for i,ratio in enumerate(n_m_ratio):
            sub_res=polynomial_diff_mn_pricing_allocation(N, H_0 * (1 - alpha), h, A, theta, b, d,ratio)
            results[i].append(sub_res)
        optimal_ress.append(optimal_res)
        # print(optimal_res)
    optimal_ress=np.array(optimal_ress)
    results=np.array(results)
    print(results.shape)
    for i in range(len(results)):
        plt.plot(results[i]/optimal_ress,label=str(n_m_ratio[i]))
    plt.legend()
    plt.show()
    return 0

def test_polynomial_pricing_find_mn():
    N = 40
    H_0 = 251.5
    h = 30
    A = 100
    theta = 10
    b = 0.001
    d = 0.01
    alphas = np.arange(0.0, 0.85, 0.01)
    poss=[]
    for alpha in alphas:
        print(alpha)
        polynomial_res,pos = polynomial_mn_pricing_allocation(N, H_0 * (1 - alpha), h, A, theta, b, d,True)
        poss.append(pos)
    poss=np.array(poss)
    # plt.plot(alphas,poss[:,0]/(poss[:,1]+0.01),label='ratio')
    plt.plot(alphas,poss[:,0],label='m')
    plt.plot(alphas,poss[:,1],label='n')
    plt.legend()
    plt.show()
    return 0

def test_polynomial_pricing_and_supply_restriction():
    N = 40
    H_0 = 251.5
    h = 30
    A = 100
    theta = 10
    b = 0.001
    d = 0.01
    alphas = np.arange(0.0, 0.85, 0.01)
    poly_ress=[]
    supply_restriction_ress=[]
    optimal_ress=[]
    cs=np.arange(b,d,0.0001)
    for alpha in alphas:
        poly_res = polynomial_pricing_allocation(N, H_0 * (1 - alpha), h, A, theta, b, d)
        sub_ress=[]
        print(alpha)
        optimal_res = optimal_allocation(N, H_0 * (1 - alpha), h, A, theta, b, d)
        for c in cs:
            sub_res = flow_restrict_allocation(N, H_0 * (1 - alpha), h, A, theta, b, d,c)
            sub_ress.append(sub_res)
        supply_restriction_ress.append(max(sub_ress))
        poly_ress.append(poly_res)
        optimal_ress.append(optimal_res)
    poly_ress=np.array(poly_ress)
    supply_restriction_ress=np.array(supply_restriction_ress)
    optimal_ress=np.array(optimal_ress)
    plt.plot(alphas,poly_ress/optimal_ress,label='poly/optimal')
    plt.plot(alphas,supply_restriction_ress/optimal_ress,label='supply/optimal')
    plt.legend()
    plt.show()


def test_Fig_10_polynomial():
    A = 100
    H_0 = 131
    h = 30
    b = 0.01
    d=0.07
    qq = []
    N = 10
    theta1 = 1
    theta2s = np.arange(1, 100, 1)
    cs = np.arange(b, d, 0.002)
    pricing_uts = []
    resctrict_uts = []
    for j in range(len(theta2s)):
        theta2 = theta2s[j]
        customer_demands = [b, d] * (N // 2)
        thetas = [theta1, theta2] * (N // 2)
        allocated = algorithm_one(H_0, A, h, customer_demands)
        utility = np.sum([orginal_utility(allocated[i], thetas[i], b, d) for i in range(len(allocated))])
        pricing_uts.append(utility)

        rr = []
        for c in cs:
            n_c = int(get_nd(N, A, b, c, h, H_0))
            # print('max allocation with c:',n_c)
            q_c = -1 / 2 * c * n_c \
                  + np.sqrt(
                -1 / 3 * c ** 2 * n_c * (n_c + 1) ** 2 * (n_c + 2) + 4 * (n_c + 1) * (H_0 - h) / A
            ) / (2 * (n_c + 1))
            res = 0
            print(n_c)
            if n_c >= N:
                for i in range(N):
                    if i % 2 == 0:
                        res += 1 + theta1 * (c - b)
                    else:
                        res += 1 + theta2 * (c - b)
            else:
                for i in range(n_c):
                    if i % 2 == 0:
                        res += 1 + theta1 * (c - b)
                    else:
                        res += 1 + theta2 * (c - b)
                if n_c % 2 == 0:
                    if q_c < b:
                        res += 0
                    else:
                        res += 1 + theta1 * (q_c - b)
                else:
                    if q_c < b:
                        res += 0
                    else:
                        res += 1 + theta2 * (q_c - b)
            rr.append(res)
        optimal_restrict = max(rr)
        resctrict_uts.append(optimal_restrict)



    resctrict_uts = np.array(resctrict_uts)
    pricing_uts = np.array(pricing_uts)
    quotient = resctrict_uts / pricing_uts
    qq.append(quotient)
    plt.plot(theta2s, qq[0], label='d=0.07')
    plt.plot(theta2s, qq[1], label='d=0.1')
    plt.legend()
    plt.show()


def main():

    # test_max_allocation()
    # test_min_allocation()
    # test_optimal_allocation()

    # compare_SWs_and_SWp()

    # test_flow_restrict_allocation()
    # test_supply_restrict_allocation()
    # test_square_root_utility()
    # test_diff_utility()
    # test_fig18()

    # test_Fig_10()
    # test_Fig_10(extend=True)
    # test_Fig_19()

    # test_polynomial_pricing()
    # test_polynomial_pricing_diff_mn()
    # test_polynomial_pricing_find_mn()
    # test_polynomial_pricing_and_supply_restriction()

    test_Fig_10_polynomial()
    pass

main()