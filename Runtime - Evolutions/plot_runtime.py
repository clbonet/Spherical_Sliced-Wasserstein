import numpy as np
import matplotlib.pyplot as plt


L_s = np.loadtxt("./Comparison_SW_Sinkhorn").reshape(1,5,20)
L_w = np.loadtxt("./Comparison_SW_W").reshape(1,5,20)
L = np.zeros((1,2,5,20))
for k, proj in enumerate([50,200]):
    L[0, k] = np.loadtxt("./Comparison_SSW_projs_"+str(proj))

L_unif = np.zeros((1,2,5,20))
for k, proj in enumerate([50,200]):
    L_unif[0, k] = np.loadtxt("./Comparison_SSW2_unif_projs_"+str(proj))

L_sw1 = np.zeros((1,2,5,20))
for k, n_projs in enumerate([50,200]):
    L_sw1[0, k] = np.loadtxt("./Comparison_SSW1_projs_"+str(n_projs))


fig = plt.figure(figsize=(6,3))

ds = [3]
samples = [int(1e2),int(1e3),int(1e4),int(1e5/2),int(1e5)]

for i, d in enumerate(ds):
    for l, n_projs in enumerate([200]):
        m = np.mean(L_sw1[i, l], axis=-1)
        s = np.std(L_sw1[i, l], axis=-1)

        plt.plot(samples, m, label=r"$SSW_1$," + r" $L=$"+str(n_projs)) # + r" $\mu=$["+str(mu[0])+","+str(mu[1])+","+str(mu[2])+"]")
        plt.fill_between(samples, m-s, m+s,alpha=0.5)

        m = np.mean(L[i, l], axis=-1)
        s = np.std(L[i, l], axis=-1)

        plt.plot(samples, m, label=r"$SSW_2$, BS," + r" $L=$"+str(n_projs)) # + r" $\mu=$["+str(mu[0])+","+str(mu[1])+","+str(mu[2])+"]")
        plt.fill_between(samples, m-s, m+s,alpha=0.5)

        m = np.mean(L_unif[i, l], axis=-1)
        s = np.std(L_unif[i, l], axis=-1)

        plt.plot(samples, m, label=r"$SSW_2$, Unif," + r" $L=$"+str(n_projs)) # + r" $\mu=$["+str(mu[0])+","+str(mu[1])+","+str(mu[2])+"]")
        plt.fill_between(samples, m-s, m+s,alpha=0.5)

    m_w = np.mean(L_w[i], axis=-1)
    s_w = np.std(L_w[i], axis=-1)

    plt.loglog(samples, m_w, label=r"W")
    plt.fill_between(samples, m_w-s_w, m_w+s_w, alpha=0.5)


    m_s = np.mean(L_s[i], axis=-1)
    s_s = np.std(L_s[i], axis=-1)

    plt.loglog(samples, m_s, label=r"Sinkhorn")
    plt.fill_between(samples, m_s-s_s, m_s+s_s, alpha=0.5)

plt.xlabel("Number of samples in each distribution", fontsize=13)
plt.ylabel("Seconds", fontsize=13)
#     plt.yscale("log")
    # plt.xscale("log")
    
plt.legend(fontsize=13, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", ncol=2)
# plt.title("Computational Time", fontsize=13)
plt.grid(True)
plt.savefig("./Comparison_SW_W2.pdf", format="pdf", bbox_inches="tight")
plt.show()
