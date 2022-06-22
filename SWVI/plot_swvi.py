import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
#parser.add_argument("--prior", type=str, default="unif_sphere", help="Specify prior")
#parser.add_argument("--d_latent", type=int, default=3, help="Dimension of the latent space")
args = parser.parse_args()


L_ess_sw = np.loadtxt("./ess_sw", delimiter=",")
L_ess_sws = np.loadtxt("./ess_sw_sphere", delimiter=",")
L_kl_sw = np.loadtxt("./kl_sw", delimiter=",")
L_kl_sws = np.loadtxt("./kl_sw_sphere", delimiter=",")


absc = np.array(range(L_ess_sw.shape[1]))*100
ntry = L_ess_sw.shape[0]

# m_ess_sw = np.mean(np.log10(L_ess_sw), axis=0)
# s_ess_sw = np.std(np.log10(L_ess_sw), axis=0)

# m_ess_sws = np.mean(np.log10(L_ess_sws), axis=0)
# s_ess_sws= np.std(np.log10(L_ess_sws), axis=0)

m_ess_sw = np.mean(L_ess_sw, axis=0)
s_ess_sw = np.std(L_ess_sw, axis=0)

m_ess_sws = np.mean(L_ess_sws, axis=0)
s_ess_sws= np.std(L_ess_sws, axis=0)


fig = plt.figure(figsize=(6,3))
plt.plot(absc, m_ess_sws, label="SSWVI")
plt.fill_between(absc,m_ess_sws-2*s_ess_sws/np.sqrt(ntry),m_ess_sws+2*s_ess_sws/np.sqrt(ntry),alpha=0.5)
plt.plot(absc, m_ess_sw, label="SWVI")
plt.fill_between(absc,m_ess_sw-2*s_ess_sw/np.sqrt(ntry),m_ess_sw+2*s_ess_sw/np.sqrt(ntry),alpha=0.5)
plt.grid(True)
plt.xlabel("Iterations",fontsize=13)
plt.title(r"ESS")
plt.legend(fontsize=13)
# plt.savefig("./ESS.png", format="png")
plt.savefig("./ESS.pdf", format="pdf", bbox_inches="tight")
plt.close("all")



absc = np.array(range(L_kl_sw.shape[1]))*100
ntry = L_kl_sw.shape[0]

m_kl_sw = np.mean(L_kl_sw, axis=0)
s_kl_sw = np.std(L_kl_sw, axis=0)

m_kl_sws = np.mean(L_kl_sws, axis=0)
s_kl_sws= np.std(L_kl_sws, axis=0)


fig = plt.figure(figsize=(6,3))
plt.plot(absc, m_kl_sws, label="SSWVI")
plt.fill_between(absc,m_kl_sws-2*s_kl_sws/np.sqrt(ntry),m_kl_sws+2*s_kl_sws/np.sqrt(ntry),alpha=0.5)
plt.plot(absc, m_kl_sw, label="SWVI")
plt.fill_between(absc,m_kl_sw-2*s_kl_sw/np.sqrt(ntry),m_kl_sw+2*s_kl_sw/np.sqrt(ntry),alpha=0.5)
plt.grid(True)
plt.xlabel("Iterations",fontsize=13)
plt.title(r"KL")
plt.legend(fontsize=13)
# plt.savefig("./KL.png", format="png")
plt.savefig("./KL.pdf", format="pdf", bbox_inches="tight")
plt.close("all")
