import numpy as np
from utils import orginal_utility,algorithm_one,\
    max_allocatin,min_allocation,optimal_allocation,pricing_allocation,\
    flow_restrict_allocation,square_root_utility,get_square_root_usage,\
    get_demmand,diff_utility
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
    theta = 0.001
    b = 0.001
    d = 0.01
    alphas = np.arange(0.0,0.85,0.01)
    max_ress=[]
    optimal_ress=[]
    pricing_ress=[]
    for alpha in alphas:
        max_res = max_allocatin(N, H_0*(1-alpha), h, A, theta, b,d)
        optimal_res = optimal_allocation(N, H_0*(1-alpha), h, A, theta, b,d)
        pricing_res = pricing_allocation(N, H_0*(1-alpha), h, A, theta, b,d,p=0.1)
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
    cs=np.arange(0.2,1,0.01)
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






def main():

    # test_max_allocation()
    # test_min_allocation()
    # test_optimal_allocation()

    # compare_SWs_and_SWp()

    # test_flow_restrict_allocation()
    # test_supply_restrict_allocation()
    # test_square_root_utility()
    test_diff_utility()
    # test_fig18()
    pass

main()