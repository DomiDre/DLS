import sys
import numpy as np
import os
import glob

from PyQt4 import QtCore, QtGui
from matplotlib.backends.backend_qt4agg \
    import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg \
    import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
from matplotlib import gridspec


pi = np.pi    

def load_dat_file(file_path):
        load_file = open(file_path, 'r', errors="ignore")
        tau = []
        g2 = []
        t_frequency = []
        t0 = 0.
        timestep = 2E-7

        frequency_data = []
        t1 = 0.

        running_number = 0    
        load_correlation_data = False
        load_count_frequency_data = False
        
        for line in load_file:
                if line.startswith('TEMP '):
                    T = float(line.split()[2])
                elif line.startswith('ANG '):
                    angle = float(line.split()[2])
                elif line.startswith('WAVE '):
                    wavelength = float(line.split()[2])
                elif line.startswith('INDX '):
                    refraction_index = float(line.split()[2])
                elif line.startswith(' DUR '):
                    duration = float(line.split()[2])
                
                elif line.startswith('COR'):
                    load_correlation_data = True
                    continue
                elif line.startswith('TRST'):
                    load_correlation_data = False
                    time_step_between_samples = float(line.split()[2])
                elif line.startswith('TRA'):
                    load_count_frequency_data = True
                    continue
                elif line.startswith('HIS'):
                    load_count_frequency_data = False
            
                if load_correlation_data:
                    data = line.split()
                    for val in data:
                        if running_number >= 15:
                            if np.allclose(running_number % 8,0):
                                timestep *= 2
                        g2.append(float(val))
                        tau.append(t0)
                        t0 += timestep
                        running_number += 1
                

                if load_count_frequency_data:
                    data = line.split()
                    for val in data:
                        frequency_data.append(float(val))
                        t_frequency.append(t1)
                        t1 += time_step_between_samples
                
        load_file.close()
        
        tau = tau[1:]
        g2 = g2[1:]
        dat_file_data = {
        "tau": np.asarray(tau),
        "g2": np.asarray(g2),
        "t_freq": np.asarray(t_frequency),
        "freq": np.asarray(frequency_data),
        "T": T,
        "ang": angle,
        "q": 4.*pi*refraction_index/(wavelength*10) * np.sin(angle/2. * pi/180),
        "n": refraction_index,
        "wavelength": wavelength,
        "dur": duration}
        return dat_file_data


def calc_g2(tau, Gamma):
        g1 = np.exp(-Gamma*tau)
        g2 = 1 + 0.33*g1**2
        return g2
        
        
class DLSViewer(QtGui.QMainWindow):
        def __init__(self, data, datlist, parent = None):
                super(DLSViewer, self).__init__(parent)
                self.data = data
                self.datlist = datlist
                
                self.index = 0
                self.N = len(self.datlist)
                
                self.current_data = None #self.data[self.datlist[self.index]]
                
                
                #self.fig = Figure(figsize=(15,10))
                self.fig = plt.figure(figsize=(15, 10))
                self.ax11 = plt.subplot2grid((2, 3), (0, 0))
                self.ax12 = plt.subplot2grid((2, 3), (0, 1))
                self.ax13 = plt.subplot2grid((2, 3), (0, 2))
                
                self.plot11, = self.ax11.plot([], [], color='black', marker='.')
                self.ax11.set_xlabel(r"$t\,/\,\mathrm{s}$")
                self.ax11.set_ylabel(r"$f\,/\,\mathrm{Hz}$")
                
                self.ax11.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
                
                self.plot13, = self.ax13.plot([], [], color='black', marker='.', linestyle='--')
                self.ax13.set_xlabel(r"$\Gamma\,/\,\mathrm{s^{-1}}$")
                self.ax13.set_ylabel(r"$A(\Gamma)$")
                self.ax13.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
                self.ax13.set_xscale('log')
                
                self.ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
                self.plot2, = self.ax2.plot([], [], color='black', marker='.', linestyle='None')
                self.plot2theo, = self.ax2.plot([], [], color='red', marker='None', linestyle='-')
                self.ax2.set_xlabel(r"$\tau\,/\,\mathrm{s}$")
                self.ax2.set_ylabel(r"$g_2$")
                self.ax2.set_xlim([2e-7, 100])
                self.ax2.set_ylim([0., 3])
                self.ax2.set_xscale('log')
                
                
                self.fig.tight_layout()
                
                
                self.canvas = FigureCanvas(self.fig)

                # use addToolbar to add toolbars to the main window directly!
                self.toolbar = NavigationToolbar(self.canvas, self)
                self.addToolBar(self.toolbar)

                self.bnext = QtGui.QPushButton('Next')
                self.bprev = QtGui.QPushButton('Previous')
                
                self.bnext.clicked.connect(self.next)
                self.bprev.clicked.connect(self.prev)
                
                self.infolabel = QtGui.QLabel("")
                #self.lineEditMomentum1 = QtGui.QLineEdit()
                #self.lineEditMomentum1.setMaximumSize(200, 30)

                self.main_widget = QtGui.QWidget(self)
                self.setCentralWidget(self.main_widget)

                layout = QtGui.QGridLayout()
                layout.addWidget(self.canvas, 0, 0, 1, 2)
                layout.addWidget(self.infolabel, 1, 0, 1, 2)
                
                layout.addWidget(self.bprev, 2, 0)
                layout.addWidget(self.bnext, 2, 1)
                
                #layout.addWidget(self.lineEditMomentum1)

                self.main_widget.setLayout(layout)
                self.show_result_tab_plot()
                self.update_plots()
                
                
        def show_result_tab_plot(self):
                self.ax12.cla()
                self.ax12.set_xlabel(r"$q^2\,/\,\mathrm{\AA^{-2}}$")
                self.ax12.set_ylabel(r"$\Gamma\,/\,\mathrm{s^{-1}}$")
                
                for dataname in self.data:
                        datafile = self.data[dataname]
                        self.ax12.errorbar(datafile["q"]**2, datafile["Gamma"], datafile["sGamma"], color='black', marker='.')
                
                self.ax12.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
                self.current_point12, = self.ax12.plot([], [], marker=r'$\circ$', color='red', markersize=10)
                
                self.canvas.draw()
                
                
        def update_plots(self):
                self.current_data = self.data[self.datlist[self.index]]
                self.plot11.set_data(self.current_data["t_freq"], self.current_data["freq"])
                self.ax11.set_xlim([self.current_data["t_freq"][0], self.current_data["t_freq"][-1]])
                self.ax11.set_ylim([0.8*min(self.current_data["freq"]), 1.2*max(self.current_data["freq"])])
                
                self.current_point12.set_data(self.current_data["q"]**2, self.current_data["Gamma"])
                
                self.plot13.set_data(self.current_data["SON_Gamma"],self.current_data["SON_A"])
                self.ax13.set_xlim([self.current_data["SON_Gamma"][0], self.current_data["SON_Gamma"][-1]])
                self.ax13.set_ylim([0, 1.2])
                
                self.plot2.set_data(self.current_data["tau"], self.current_data["g2"])
                self.plot2theo.set_data(self.current_data["tau"], self.current_data["g2_contim"])
                self.ax2.set_xlim([self.current_data["tau"][0], self.current_data["tau"][-1]])
                self.infolabel.setText("Opened file: " + self.datlist[self.index] + "\n" +
                                        "2 th = " + str(self.current_data["ang"]) + " deg\n"+
                                        "n = " + str(self.current_data["n"]) + "\n"+
                                        "wavelength = " + str(self.current_data["wavelength"]) + " nm\n"+
                                        "T  = " + str(self.current_data["T"]) + " K\n"+
                                        "Measurement Time = " + str(self.current_data["dur"]) + " s\n"+
                                        "q = " + "{:.3e}".format(self.current_data["q"]) + " A-1")
                
                self.canvas.draw()

        def next(self, event):
                self.index = (self.index + 1) % self.N
                self.update_plots()
        

        def prev(self, event):
                self.index = (self.index - 1) % self.N
                self.update_plots()


if __name__=='__main__':
        numargs = len(sys.argv) -1
        if numargs == 0:
                print("Usage: python contim_viewer.py dls_folder_path [..]")
                print("Possible Parameters:")
                sys.exit("None")
                
        else:
                data_folder_path = sys.argv[1]
                if not os.path.isdir(data_folder_path):
                        sys.exit("Folder path: " + str(data_folder_path) + " does not exist.")
                
                list_of_dat_files = sorted(glob.glob(data_folder_path + "/*.DAT"))
                if len(list_of_dat_files) == 0:
                        sys.exit("No *.DAT files in " + str(data_folder_path))
                
                if not os.path.exists(data_folder_path+"/RESULT.TAB"):
                        sys.exit("RESULT.TAB missing in " + str(data_folder_path))
                
                
                list_of_son_files = []
                for datfile in list_of_dat_files:
                        son_path = datfile.split(".DAT")[0] + ".SON"
                        if os.path.exists(son_path):
                                list_of_son_files.append(son_path)
                        else:
                                datname = os.path.basename(datfile).split(".DAT")[0]
                                n_dat = len(datname)
                                if n_dat > 8:
                                        print("Normally data names may only have 8 characters. Somethings fishy...")
                                spacers = "" +" "*(8-n_dat)
                                son_path = datfile.split(".DAT")[0] + spacers + ".SON"
                                if os.path.exists(son_path):
                                        list_of_son_files.append(son_path)
                                else:
                                        print("Can't find " + son_path)
                        
                if not len(list_of_son_files) == len(list_of_dat_files):
                        sys.exit()
                
                
                dat_files_list = []
                dat_files_data = {}
                for dat_file in list_of_dat_files:
                        datname = os.path.basename(dat_file).split(".DAT")[0]
                        dat_files_list.append(datname)
                        dat_files_data[datname] = load_dat_file(dat_file)
                
                
                result_tab_file = open(data_folder_path+"/RESULT.TAB", "r")
                for line in result_tab_file:
                        split_line = line.strip().split()
                        info_list = []
                        for i, entry in enumerate(split_line):
                                if i == 0:
                                        continue
                                info_list.append(float(entry))
                        dat_files_data[split_line[0]]["result_tab"] = info_list
                        dat_files_data[split_line[0]]["Gamma"] = dat_files_data[split_line[0]]["result_tab"][1]
                        dat_files_data[split_line[0]]["sGamma"] = dat_files_data[split_line[0]]["result_tab"][2]
                
                for i, son_file in enumerate(list_of_son_files):
                        son_data = np.genfromtxt(son_file)
                        dat_files_data[dat_files_list[i]]["SON_Gamma"] = son_data[:, 0] 
                        dat_files_data[dat_files_list[i]]["SON_A"] = son_data[:, 1]
                        dat_files_data[dat_files_list[i]]["SON_sA"] = son_data[:, 2]
                        
                        dat_files_data[dat_files_list[i]]["g2_contim"] = calc_g2(dat_files_data[dat_files_list[i]]["tau"], 
                                                                                 dat_files_data[dat_files_list[i]]["Gamma"]) 
                        
                
                
                qApp = QtGui.QApplication(sys.argv)
                main_window = DLSViewer(dat_files_data, dat_files_list)
                main_window.show()
                sys.exit(qApp.exec_())   
                




