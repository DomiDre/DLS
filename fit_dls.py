import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import lmfit
import scipy.constants as consts

      
if __name__=='__main__':
        k = consts.k
        temperature = 298.15

        numargs = len(sys.argv) -1
        if numargs == 0:
                files_in_cwd = [f for f in os.listdir('.') if os.path.isfile(f)]
                hfile = None
                for hfile in files_in_cwd:
                        if hfile.endswith(".params"):
                                parampath = hfile
                if hfile is None:
                        print("Usage: python fitdls.py parameterfile [..]")
                        print("Possible Parameters:")
                        sys.exit("None")
        else:
                parampath =  sys.argv[1]
        if not os.path.exists(parampath):
                sys.exit("File does not exist: " + filepath)
        
        paramfile = open(parampath, "r")
        for line in paramfile:
                if line.startswith("#"):
                        continue
                
                split_line = line.strip().split()
                if split_line[0] == "datafile":
                        filepath = split_line[1]
                elif split_line[0] == "wavelength":
                        wavelength = float(split_line[1])
                elif split_line[0] == "n":
                        refractive_index = float(split_line[1])
                elif split_line[0] == "viscosity":
                        viscosity = float(split_line[1])
                elif split_line[0] == "temperature":
                        temperature = float(split_line[1])
                elif split_line[0] == "save_report":
                        if split_line[1] == "True":
                                save_report = True
                        else:
                                save_report = False
                elif split_line[0] == "plot_result":

                        if split_line[1] == "True":
                                plot_result = True
                        else:
                                plot_result = False
                elif split_line[0] == "force_zero":

                        if split_line[1] == "True":
                                force_zero = True
                        else:
                                force_zero = False
                

                
                
        paramfile.close()
        
        print("Loaded " + parampath)
        print("Set following parameters:")
        print("Viscosity: " + str(viscosity))
        print("n: " + str(refractive_index))
        print("wavelength: " + str(wavelength))
        
        
        if not os.path.exists(filepath):
                sys.exit("File does not exist: " + filepath)
        dataname = os.path.basename(filepath).split(".")[0]
                                           
        data = np.genfromtxt(filepath)
        
        q = data[:, 0]
        gamma = data[:, 1]
        sgamma = data[:, 2]
        q2 = q**2

        def linear(p, x):
                return p["m"].value*x + p["c"].value

        def residual(p, x, y, sy):
                return (linear(p, x) - y)#/sy
        
        m_estimate = (gamma[-1] - gamma[0]) / (q2[-1] - q2[0])
        para = lmfit.Parameters()
        para.add("m", m_estimate)
        para.add("c", 0, vary=(not force_zero))

        fit_result = lmfit.minimize(residual, para, args=(q2, gamma, sgamma))
        report = lmfit.fit_report(fit_result)
        if save_report:
                report_file = open(dataname+"_fit_report.dat", "w")
                report_file.write(report)
                report_file.close()
        print(report)
        para = fit_result.params
        
        print_format = "{:.1f}"
        D = para["m"].value*1e-20
        
        
        R = lambda D: k*temperature / (6*np.pi*viscosity*D) # Stokes-Einstein Equation

        part_radius = R(D)*1e9
        print("r = ", print_format.format(part_radius), " nm")
        print("d = ", print_format.format(2*part_radius), " nm")
        fig, ax = plt.subplots(figsize=(10/2.54, 10/2.54), dpi=254)
        ax.errorbar(q2, gamma, sgamma, marker='.', linestyle='None', color='red')
        ax.plot(q2, linear(para, q2), linestyle='-', marker='None', color='black', label="$\Gamma\,=\,Dq^2+c$\n"+
                                                                          "$D\,=\, "+"{:.2e}".format(D)+" \,\mathrm{m^2 s^{-1}}$\n"+
                                                                          "$c\,=\, "+"{:.2e}".format(para["c"].value)+" \,\mathrm{s^{-1}}$")
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        ax.set_xlabel(r"$q^2\,/\,\mathrm{\AA^{-2}}$")
        ax.set_ylabel(r"$\Gamma\,/\,\mathrm{s^{-1}}$")
        plt.legend(numpoints=1, fontsize=10, loc="upper left").draw_frame(False)
        fig.tight_layout()
        
        if plot_result:
                plt.savefig(dataname+".png", dpi=254)
        plt.show()
