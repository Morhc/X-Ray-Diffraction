from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os

plt.rcParams['text.usetex'] = True

def identify_peaks(x, y, expt):
    """ returns the indices of any detected peaks """

    def func(x,a,b,c):
        return c**(-x-b) + a

    window = 20

    xs, mns, stds = [], [], []
    cx, cy = [], []

    for i in range(window, y.size, window):
        curr = y[i-window:i]
        mn, std = np.mean(curr), np.std(curr)
        mns.append(mn)
        stds.append(std)
        xs.append(i-window/2)
        if std < 25:
            cx.append(i-window/2)
            cy.append(mn)

    cx, cy = np.asarray(cx), np.asarray(cy)
    xs, mns, stds = np.asarray(xs), np.asarray(mns), np.asarray(stds)

    popt, _ = curve_fit(func, cx, cy, p0=(20, -800, 1.1))
    fy = func(np.asarray(xs), popt[0], popt[1], popt[2])

    if expt == 'Metal Wire': filter = np.abs(fy-mns) > mns-stds
    elif expt == 'Powder Metal': filter = np.abs(fy-mns) > 0.4*(mns-stds)
    else:
        filter = fy == 'p'
        raise(f'Your choice of experiment "{expt}" is invalid. Choose a valid experiment.')

    find = xs[filter]

    if expt == 'Metal Wire': factor = 2.5
    elif expt == 'Powder Metal': factor = 1
    else: factor = 0

    peakx, peaky = [], []
    for ind in find:
        curr = y[int(ind-factor*window):int(ind+factor*window)]
        val, arg = np.max(curr), int(ind-factor*window) + np.argmax(curr)
        th2 = x[arg]

        if th2 not in peakx:
            peakx.append(th2)
            peaky.append(val)

    return np.asarray(peakx), np.asarray(peaky)

def identify_plane(theta, K, cubeplot, latticeplot, latticetable):
    #from appendix 10 (p. 516)
    simple = [100, 110, 111, 200, 210, 211, 220, 300] #300 also has 221
    face =   [111, 200, 220, 311, 222, 400, 331, 420]
    body =   [110, 200, 211, 220, 310, 222, 321, 400]
    diamond= [111, 220, 311, 400, 331, 422, 511, 440] #511 also has 333

    cubic_types = ['Simple', 'Face', 'Body', 'Diamond']

    sin2theta, cos2theta = np.sin(theta)**2, np.cos(theta)**2

    def xxx(num): return sum([int(i)**2 for i in str(num)])

    s_s, s_f, s_b, s_d = [], [], [], []
    sx, fx, bx, dx = [], [], [], []
    for sin2, si, fa, bo, di in zip(sin2theta, simple, face, body, diamond):
        s_s.append(sin2/xxx(si))
        s_f.append(sin2/xxx(fa))
        s_b.append(sin2/xxx(bo))
        s_d.append(sin2/xxx(di))

        sx.append(xxx(si))
        fx.append(xxx(fa))
        bx.append(xxx(bo))
        dx.append(xxx(di))

    s_s, s_f, s_b, s_d = np.asarray(s_s), np.asarray(s_f), np.asarray(s_b), np.asarray(s_d)
    sx, fx, bx, dx = np.asarray(sx), np.asarray(fx), np.asarray(bx), np.asarray(dx)

    xs, ys = [sx,fx,bx,dx], [s_s,s_f,s_b,s_d]

    s_st, f_st, b_st, d_st = np.std(s_s), np.std(s_f), np.std(s_b), np.std(s_d)

    plt.scatter(sx,s_s,edgecolor='black',facecolor='grey',label=rf'Simple: $\pm${np.round(s_st, 4)}')
    plt.scatter(fx,s_f,edgecolor='black',facecolor='tan',label=rf'Face: $\pm${np.round(f_st, 4)}')
    plt.scatter(bx,s_b,edgecolor='black',facecolor='green',label=rf'Body: $\pm${np.round(b_st, 4)}')
    plt.scatter(dx,s_d,edgecolor='black',facecolor='lightblue',label=rf'Diamond: $\pm${np.round(d_st, 4)}')
    plt.title('Comparing the Different Cubic Types')
    plt.ylabel(r'$\frac{sin^2(\theta)}{s}$', fontsize=15)
    plt.xlabel(r'$s=h^2 + k^2 + l^2$', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cubeplot)
    plt.close('all')

    i = np.argmin([s_st, f_st, b_st, d_st])

    cubic_type, ddd, data, params = cubic_types[i], [simple, face, body, diamond][i], ys[i], xs[i]

    la, lb, lw = 0.154051, 0.154433, 0.154178
    f = 1/np.sqrt(data)
    aa, ab, aw = 10*la*f/2, 10*lb*f/2, 10*lw*f/2

    def line(x,m,b): return m*x + b

    po_a, pc_a = curve_fit(line, sin2theta, aa)
    po_b, pc_b = curve_fit(line, sin2theta, ab)
    po_w, pc_w = curve_fit(line, sin2theta, aw)

    xtest = np.linspace(0,1,100)
    ya = line(xtest, *po_a)
    yb = line(xtest, *po_b)
    yw = line(xtest, *po_w)

    plt.plot(xtest,ya,'black',zorder=1)
    plt.plot(xtest,yb,'black',zorder=1)
    plt.plot(xtest,yw,'black',zorder=1)

    plt.scatter(sin2theta, aa, edgecolor='black',facecolor='blue',label=r'$\lambda$ = ' + '0.154051nm')
    plt.scatter(sin2theta, ab, edgecolor='black',facecolor='yellow',label=r'$\lambda$ = ' + '0.154433nm')
    plt.scatter(sin2theta, aw, edgecolor='black',facecolor='green',label=r'$\lambda$ = ' + '0.154178nm')

    plt.ylabel(r'$a_0$ $(\AA)$', fontsize=12)
    plt.xlabel(r'$sin^2(\theta)$', fontsize=12)
    plt.title(r'Lattice constant $a_0$ vs $sin^2(\theta)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(latticeplot)


    with open(latticetable, mode='w', encoding = 'utf-8') as f:
        f.writelines(f'#The data was found to be of cubic type {cubic_type}.\n')
        f.writelines(f'#The best a0 for la was {round(ya[-1],4)}+/-{round(np.sqrt(np.diag(pc_a))[0],4)}\n')
        f.writelines(f'#The best a0 for lb was {round(yb[-1],4)}+/-{round(np.sqrt(np.diag(pc_b))[0],4)}\n')
        f.writelines(f'#The best a0 for lw was {round(yw[-1],4)}+/-{round(np.sqrt(np.diag(pc_w))[0],4)}\n')

        f.writelines('#a0_a (Ang)\ta0_b (Ang)\ta0_w (Ang)\tsin2theta\t s\t hkl\n')
        for a0a, a0b, a0w, sin2, s, hkl in zip(aa, ab, aw, sin2theta, params, ddd):
            f.writelines(f'{a0a}\t{a0b}\t{a0w}\t{sin2}\t{s}\t{hkl}\n')




expts = ['Metal Wire', 'Powder Metal']

for expt in expts:
    folder = rf'Expt - {expt}'
    dataset = os.path.join(folder, f'{expt.replace(" ", "")}_RawData')

    peakplot = os.path.join(folder, f'{expt.replace(" ", "")}_Peaks.png')
    peaktable = os.path.join(folder, f'{expt.replace(" ", "")}_Peaks')
    kplot = os.path.join(folder, f'{expt.replace(" ", "")}_KPlot.png')
    cubeplot = os.path.join(folder, f'{expt.replace(" ","")}_IdentifyPlane.png')
    latticeplot = os.path.join(folder, f'{expt.replace(" ", "")}_Lattice.png')
    latticetable = os.path.join(folder, f'{expt.replace(" ", "")}_Lattice')

    raw = pd.read_csv(dataset, comment='#', header=None, sep=' ')
    raw = raw.rename(columns={0: 'Two Theta', 1: 'H', 2: 'K', 3: 'Epoch', 4: 'Seconds', 5: 'Monitor', 6:  'Detector'})

    peakx, peaky = identify_peaks(raw['Two Theta'], raw['Detector'], expt)

    plt.scatter(peakx, peaky, marker='x', facecolor='black',zorder=2, label='Peaks')
    plt.plot(raw['Two Theta'], raw['Detector'], color="#9b19f5", zorder=1, label='Raw Data')
    plt.title(f'Counts vs Angular Position: {expt}', loc='center')
    plt.ylabel('Counts')
    plt.xlabel('Angular Position '+ r'($2\theta$, degrees)')
    plt.xlim(0,130)
    plt.ylim(0,850)
    plt.xticks(range(0,130+1,10), labels=[f'{i}°' for i in range(0,130+1,10)])
    plt.legend()
    plt.savefig(peakplot)
    plt.close('all')

    with open(peaktable, mode='w', encoding = 'utf-8') as f:
        f.writelines('#2Theta (deg)\tCounts\n')
        for th2, count in zip(peakx, peaky): f.writelines(f'{th2}\t{count}\n')


    def get_d(th, lam):
        return lam/(2*np.sin(th))

    th = np.radians(peakx/2)

    alpha, beta, weighted = get_d(th, 0.154051), get_d(th, 0.154433), get_d(th, 0.154178)
    dth = np.radians(0.05)
    d_alp, d_bet, d_wei = -alpha*dth/np.tan(th), -beta*dth/np.tan(th), -weighted*dth/np.tan(th) #page 350
    #a = d*sqrt(h^2+k^2+l^2)
    #d_a / a = -cot(th) dth #page 351

    rel = np.cos(th)**2 / np.sin(th) + np.cos(th)**2 / th

    def line(x, m, b): return m*x + b

    popt, pcov = curve_fit(line, rel, d_wei/weighted, p0=[-0.005, 0])
    perr = np.sqrt(np.diag(pcov))

    kfit = line(rel, *popt)

    """
    plt.scatter(rel, d_alp/alpha, edgecolor='black', facecolor='blue', label=r'$\lambda$ = ' + '0.154051nm', zorder=2)
    plt.scatter(rel, d_bet/beta, edgecolor='black', facecolor='yellow', label=r'$\lambda$ = ' + '0.154433nm', zorder=3)
    plt.scatter(rel, d_wei/weighted, edgecolor='black', facecolor='lightgreen', label=r'$\lambda$ = ' + '0.154178nm', zorder=4)
    plt.plot(rel, kfit, color='black', label=rf'$K = {popt[0]:.2e} \pm {perr[0]:.2e}', zorder=1)

    plt.title(r'Determining $K$')
    plt.xlabel(r'$\frac{cos^2(\theta)}{sin(\theta)}$' + r' $+$ ' + r'$\frac{cos^2(\theta)}{\theta}$', fontsize=15)
    plt.ylabel(r'$\frac{\Delta d}{d}$', fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(kplot)
    plt.close('all')
    """

    plt.scatter(np.sin(th)**2, weighted, edgecolor='black', facecolor='blue', label=r'$\lambda$ = ' + '0.154178nm', zorder=2)

    plt.title(r'Determining $K$')
    plt.xlabel(r'$sin^2(\theta)$', fontsize=15)
    plt.ylabel(r'$d$', fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(kplot)
    plt.close('all')


    identify_plane(th, popt[0], cubeplot, latticeplot, latticetable)



    #page 355

    #from page 310 in Cullity, Table 10-1
