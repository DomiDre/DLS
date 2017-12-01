import matplotlib as mpl
import time
mpl.use("Qt5Agg")

from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas,\
        NavigationToolbar2QT as NavigationToolbar
import PyQt5.QtWidgets as pyqt5widget

import warnings
# remove some annoying deprecation warnings
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import numpy as np
import sys, os, glob, lmfit, pickle
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')


mpl.rcParams['font.size'] = 8
pi = np.pi    
version = 1.2

class DLSViewer(pyqt5widget.QMainWindow):
    def __init__(self, parent = None):
        super().__init__()
        
        self.n_args = len(sys.argv) -1
        self.data_folder_path = os.getcwd()
        self.list_of_dat_files = sorted(glob.glob(self.data_folder_path + "/*.DAT"))
        if len(self.list_of_dat_files) == 0:
            sys.exit("No *.DAT files in " + str(self.data_folder_path))

        self.index = 0
        self.n_skip = 5
        
        self.qtimer = QtCore.QTimer()
        
        if os.path.exists(self.data_folder_path+"/DLS_fitting.p"):
            print("Loading previous fitting results from " +\
                        self.data_folder_path+"/DLS_fitting.p")
            self.dat_files_list, self.dat_files_data =\
                pickle.load( open( self.data_folder_path+"/DLS_fitting.p", "rb" ) )
        else:
            self.load_dat_files()

        wavelengths = []
        ref_indxs = []
        durs = []
        for datfile in self.dat_files_list:
            data_from_file = self.dat_files_data[datfile]
            wavelengths.append(data_from_file["wavelength"])
            durs.append(data_from_file["dur"])
            ref_indxs.append(data_from_file["n"])
            
        self.mean_wavelength = np.mean(wavelengths)
        self.mean_dur = np.mean(durs)
        self.mean_n = np.mean(ref_indxs)
        self.N = len(self.dat_files_list)
        self.current_data = None
        
        self.fig = plt.figure(figsize=(15/2.54, 10/2.54))
        self.ax11 = plt.subplot2grid((2, 3), (0, 0))
        self.ax12 = plt.subplot2grid((2, 3), (0, 1))
        
        self.plot11, = self.ax11.plot([], [], color='black', marker='.')
        self.ax11.set_xlabel(r"$\mathit{t} \, / \, s$")
        self.ax11.set_ylabel(r"$\mathit{f} \, / \, kHz$")
        
        self.ax11.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        
        
        self.ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        self.plot2, = self.ax2.plot([], [], color='black', linestyle='None')
        self.plot2theo, = self.ax2.plot([], [], color='red', marker='None')
        self.plot2cross1, = self.ax2.plot([], [], color='red', marker='None')
        self.plot2cross2, = self.ax2.plot([], [], color='red', marker='None')
        
        self.ax2.set_xlabel(r"$\tau \, / \, s$")
        self.ax2.set_ylabel(r"$g_2$")
        self.ax2.set_xlim([2e-7, 100])
        self.ax2.set_ylim([0.9, 2.1])
        self.ax2.set_xscale('log')
        
        
        self.fig.tight_layout()
        
        self.canvas = FigureCanvas(self.fig)

        # use addToolbar to add toolbars to the main window directly!
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.addToolBar(self.toolbar)

        self.bnext = pyqt5widget.QPushButton('Next')
        self.bprev = pyqt5widget.QPushButton('Previous')
        
        self.bvalid = pyqt5widget.QPushButton('Toggle Valid')
        self.bsave = pyqt5widget.QPushButton('Save Data')
        self.bswitchGamma = pyqt5widget.QPushButton('Switch Gamma')
        
        self.lgamma1 = pyqt5widget.QLabel("Gamma1: ")
        self.tgamma1 = pyqt5widget.QLineEdit()
        self.cgamma1 = pyqt5widget.QCheckBox(self)
        self.cgamma1.setChecked(True)
            
        self.lgamma2 = pyqt5widget.QLabel("Gamma2: ")
        self.tgamma2 = pyqt5widget.QLineEdit()
        self.cgamma2 = pyqt5widget.QCheckBox(self)
        self.cgamma2.setChecked(True)
        
        self.lA1 = pyqt5widget.QLabel("A1: ")
        self.tA1 = pyqt5widget.QLineEdit()
        self.cA1 = pyqt5widget.QCheckBox(self)
        self.cA1.setChecked(True)

        self.lf = pyqt5widget.QLabel("f: ")
        self.tf = pyqt5widget.QLineEdit()
        self.cf = pyqt5widget.QCheckBox(self)
        self.cf.setChecked(True)

        self.lnskip = pyqt5widget.QLabel("Skip: ")
        self.tnskip = pyqt5widget.QLineEdit()
        self.tnskip.setText(str(self.n_skip))
        
        self.bplot = pyqt5widget.QPushButton('Replot')
        self.brefit = pyqt5widget.QPushButton('Refit')
        self.bsinglemodefit = pyqt5widget.QPushButton('Single Mode Fit')
        self.breloaddata = pyqt5widget.QPushButton('Reload .dat files')
        self.bsinglemodefitall = pyqt5widget.QPushButton('Fit all Single Mode')
        self.bplot2d = pyqt5widget.QPushButton('Plot Data 2D')
        self.bsavethGamma = pyqt5widget.QPushButton('Save data th, Gamma')
        self.bsave_counts = pyqt5widget.QPushButton('Save all counts')
        
        self.bsave_corfreq = pyqt5widget.QPushButton('Save Autocorrelation+Frequency')
        
        self.bnext.clicked.connect(self.next)
        self.bprev.clicked.connect(self.prev)
        self.bvalid.clicked.connect(self.toggle_valid)
        self.bsave.clicked.connect(self.save_data)
        self.bswitchGamma.clicked.connect(self.switch_gamma)
        self.bplot.clicked.connect(self.replot)
        self.brefit.clicked.connect(self.refit)
        self.bsinglemodefit.clicked.connect(self.singlemodefit)
        self.bsinglemodefitall.clicked.connect(self.singlemodefitall)
        self.bplot2d.clicked.connect(self.plot2d)
        self.breloaddata.clicked.connect(self.reload_datfiles)
        self.bsavethGamma.clicked.connect(self.save_data_theta_Gamma)
        self.bsave_counts.clicked.connect(self.save_counts)
        self.bsave_corfreq.clicked.connect(self.save_corr_freq)

        self.infolabel = pyqt5widget.QLabel("")
        self.main_widget = pyqt5widget.QWidget(self)
        self.setCentralWidget(self.main_widget)

        width = 600
#        self.tgamma1.setFixedWidth(width)
#        self.tgamma2.setFixedWidth(width)
#        self.tA1.setFixedWidth(width)
        self.bnext.setFixedWidth(width)
        self.bprev.setFixedWidth(width)
        self.bvalid.setFixedWidth(width)
        self.bsave.setFixedWidth(width)
        self.bswitchGamma.setFixedWidth(width)
        self.bplot.setFixedWidth(width)
        self.brefit.setFixedWidth(width)
        self.breloaddata.setFixedWidth(width)
        self.bsinglemodefit.setFixedWidth(width)
        self.bsinglemodefitall.setFixedWidth(width)
        self.bsavethGamma.setFixedWidth(width)
        self.bsave_counts.setFixedWidth(width)
        self.bsave_corfreq.setFixedWidth(width)
        
        self.lgamma1.setAlignment(QtCore.Qt.AlignRight)
        self.lgamma2.setAlignment(QtCore.Qt.AlignRight)
        self.lA1.setAlignment(QtCore.Qt.AlignRight)
        self.lf.setAlignment(QtCore.Qt.AlignRight)
        self.lnskip.setAlignment(QtCore.Qt.AlignRight)
        
        
        button_widget = pyqt5widget.QWidget(self) 
        button_layout = pyqt5widget.QGridLayout()
        button_layout.addWidget(self.bprev, 0, 0)
        button_layout.addWidget(self.bnext, 0, 1)
        button_layout.addWidget(self.bvalid, 1, 0)
        button_layout.addWidget(self.bsave, 1, 1)
        button_layout.addWidget(self.bswitchGamma, 2, 0)
        button_layout.addWidget(self.bsinglemodefit, 2, 1)
        button_layout.addWidget(self.bplot, 3, 0)
        button_layout.addWidget(self.brefit, 3, 1)
        button_layout.addWidget(self.bsinglemodefitall, 4, 0)
        button_layout.addWidget(self.breloaddata, 4, 1)
        button_layout.addWidget(self.bplot2d, 5, 0)
        button_layout.addWidget(self.bsavethGamma, 5, 1)
        button_layout.addWidget(self.bsave_counts, 6, 0)
        button_layout.addWidget(self.bsave_corfreq, 6, 1)
        button_widget.setLayout(button_layout)
        
        parameter_widget = pyqt5widget.QWidget(self) 
        parameter_layout = pyqt5widget.QGridLayout()
        parameter_layout.addWidget(self.lgamma1, 0, 0)
        parameter_layout.addWidget(self.tgamma1, 0, 1)
        parameter_layout.addWidget(self.cgamma1, 0, 2)
        parameter_layout.addWidget(self.lgamma2, 1, 0)
        parameter_layout.addWidget(self.tgamma2, 1, 1)
        parameter_layout.addWidget(self.cgamma2, 1, 2)
        parameter_layout.addWidget(self.lA1, 2, 0)
        parameter_layout.addWidget(self.tA1, 2, 1)
        parameter_layout.addWidget(self.cA1, 2, 2)
        parameter_layout.addWidget(self.lf, 3, 0)
        parameter_layout.addWidget(self.tf, 3, 1)
        parameter_layout.addWidget(self.cf, 3, 2)
        parameter_layout.addWidget(self.lnskip, 5, 0)
        parameter_layout.addWidget(self.tnskip, 5, 1)
        parameter_widget.setLayout(parameter_layout)

        layout = pyqt5widget.QGridLayout()
        layout.addWidget(self.canvas, 0, 0, 1, 2)
        layout.addWidget(self.infolabel, 0, 2)
        layout.addWidget(button_widget, 1, 0, 1, 2)
        layout.addWidget(parameter_widget, 1,2)

        self.main_widget.setLayout(layout)
        self.show_result_tab_plot()
        self.update_plots()
        
        self.save_data(None)

    def load_dat_files(self):
        self.dat_files_list = []
        self.dat_files_data = {}
        for dat_file in self.list_of_dat_files:
            datname = os.path.basename(dat_file).split(".DAT")[0]
            self.dat_files_list.append(datname)
            self.dat_files_data[datname] = self.load_dat_file(dat_file)
            self.dat_files_data[datname] = self.fit_gamma( self.dat_files_data[datname] )
    
    def help(self):
        print("Usage: python fit_dls_two_mode.py [dls_folder_path] [..]")
        print("Possible Parameters:")
        print("-saveth\tSave with two theta in first column.")

    def show_result_tab_plot(self):
        self.ax12.cla()
        self.ax12.set_xlabel(r"$\mathit{q^2} \, / \, \AA^{-2}$")
        self.ax12.set_ylabel(r"$\Gamma \, / \, s^{-1}$")
        
        minq = np.inf
        maxq = -np.inf
        minG = np.inf
        maxG = -np.inf
        for dataname in self.dat_files_data:
            datafile = self.dat_files_data[dataname]
            if datafile["valid"]:
                plot_color = 'black'
            else:
                plot_color = 'red'
            q2_val = datafile["q"]**2
            Gval = datafile["Gamma1"]
            if q2_val < minq:
                minq = q2_val
            if q2_val > maxq:
                maxq = q2_val
            if Gval < minG:
                minG = Gval
            if Gval > maxG:
                maxG = Gval
            self.ax12.errorbar(q2_val, Gval,\
                             datafile["sGamma1"], color=plot_color, marker='.')
        
        self.ax12.set_xlim([minq*0.9, maxq*1.1])
        self.ax12.set_ylim([minG*0.9, maxG*1.1])
        self.ax12.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        self.current_point12, = self.ax12.plot([], [], marker=r'$\circ$', color='red', markersize=10)
        self.fig.tight_layout()
        self.canvas.draw()

    def reload_datfiles(self):
        try:
            self.n_skip = int(self.tnskip.text())
        except ValueError:
            print("Error. Did not recognize value for skipping data points.")
        self.load_dat_files()
        self.replot()
        
    def update_info_label(self):
        self.current_data = self.dat_files_data[self.dat_files_list[self.index]]
        self.infolabel.setText("Opened file: " + self.dat_files_list[self.index] + "\n" +
            "2 th = " + str(self.current_data["ang"]) + " deg\n"+
            "n = " + str(self.current_data["n"]) + "\n"+
            "wavelength = " + str(self.current_data["wavelength"]) + " nm\n"+
            "T  = " + str(self.current_data["T"]) + " K\n"+
            "Measurement Time = " + str(self.current_data["dur"]) + " s\n"+
            "q = " + "{:.3e}".format(self.current_data["q"]) + " A-1\n"+
            "A1 = " + "{:.3e}".format(self.current_data["A1"]) + " +/- " +\
                     "{:.3e}".format(self.current_data["sA1"]) + "s-1\n"+
            "Gamma1 = " + "{:.3e}".format(self.current_data["Gamma1"]) + " +/- " +\
                    "{:.3e}".format(self.current_data["sGamma1"]) + "s-1\n"+
            "Gamma2 = " + "{:.3e}".format(self.current_data["Gamma2"]) + " +/- " +\
                    "{:.3e}".format(self.current_data["sGamma2"]) + "s-1\n"+
            "f = " + "{:.3e}".format(self.current_data["f"]) + " +/- " +\
                    "{:.3e}".format(self.current_data["sf"]) + "\n"+
            "Valid: " + str(self.current_data["valid"]))
        self.tgamma1.setText("{:.3e}".format(self.current_data["Gamma1"]))
        self.tgamma2.setText("{:.3e}".format(self.current_data["Gamma2"]))
        self.tA1.setText("{:.3e}".format(self.current_data["A1"]))
        self.tf.setText("{:.3e}".format(self.current_data["f"]))
            
            
    def update_plots(self):
        self.current_data = self.dat_files_data[self.dat_files_list[self.index]]
        self.plot11.set_data(self.current_data["t_freq"], self.current_data["freq"])
        self.ax11.set_xlim([self.current_data["t_freq"][0], self.current_data["t_freq"][-1]])
        self.ax11.set_ylim([0.8*min(self.current_data["freq"]), 1.2*max(self.current_data["freq"])])
        
        self.current_point12.set_data(self.current_data["q"]**2, self.current_data["Gamma1"])
        
        if self.current_data["valid"]:
            self.plot2cross1.set_data([], [])
            self.plot2cross2.set_data([], [])
                
        else:
            self.plot2cross1.set_data(self.ax2.get_xlim(), self.ax2.get_ylim())
            self.plot2cross2.set_data(self.ax2.get_xlim(), self.ax2.get_ylim()[::-1])
        self.plot2.set_data(self.current_data["tau"], self.current_data["g2"])
        self.plot2theo.set_data(self.current_data["tau"], self.current_data["g2_fit"])
        self.ax2.set_xlim([self.current_data["tau"][0], self.current_data["tau"][-1]])
        self.update_info_label()
        self.canvas.draw()

    def next(self, event):
        self.index = (self.index + 1) % self.N
        self.update_plots()

    def prev(self, event):
        self.index = (self.index - 1) % self.N
        self.update_plots()
    
    def toggle_valid(self, event):
        self.dat_files_data[self.dat_files_list[self.index]]["valid"] =\
              not self.dat_files_data[self.dat_files_list[self.index]]["valid"]
        self.update_plots()
        self.show_result_tab_plot()

    def switch_gamma(self, event):
        self.dat_files_data[self.dat_files_list[self.index]]["Gamma1"],\
            self.dat_files_data[self.dat_files_list[self.index]]["Gamma2"] =\
                self.dat_files_data[self.dat_files_list[self.index]]["Gamma2"],\
                self.dat_files_data[self.dat_files_list[self.index]]["Gamma1"]
        self.dat_files_data[self.dat_files_list[self.index]]["sGamma1"],\
            self.dat_files_data[self.dat_files_list[self.index]]["sGamma2"] =\
                self.dat_files_data[self.dat_files_list[self.index]]["sGamma2"],\
                self.dat_files_data[self.dat_files_list[self.index]]["sGamma1"]
        self.dat_files_data[self.dat_files_list[self.index]]["A1"] =\
                1-self.dat_files_data[self.dat_files_list[self.index]]["A1"]
        self.update_plots()
        self.show_result_tab_plot()
    
    
    def save_data(self, event, save_thG=False):
        save_file = open(self.data_folder_path+"/DLS_results.xye", "w")
        precision = "{:.4e}"
        save_file.write("#DLS data evaluated using Dominiques two mode fitting "+\
                        "script v" + str(version) + "\n")
        save_file.write("#For questions contact: Dominique.Dresen@uni-koeln.de\n")
        save_file.write("#Read .dat files from folder: " + self.data_folder_path + "\n")
        save_file.write("#Skipped " + str(self.n_skip) + " datapoints "+\
                        "at beginning while reading tau, g2 data from each .dat file.\n")
        save_file.write("#Mean duration of measurements: " + str(self.mean_dur) + " s\n")
        save_file.write("#Refractive index: " + str(self.mean_n) + "\n")
        save_file.write("#Wavelength: " + str(self.mean_wavelength) + " nm\n")
        
        #are all datasets fitted in single mode?
        mode_single = False
        for dataname in self.dat_files_list:
            dataset = self.dat_files_data[dataname]
            mode_single = mode_single or np.allclose(dataset["Gamma2"], 0.)

        if save_thG:
            if mode_single:
                first_header = "#2th/° \tGamma/s-1 \tsGamma/s-1\t f\t sf\n"
            else:
                first_header = "#2th/° \tGamma1/s-1 \tsGamma1/s-1\tGamma2/s-1 "+\
                                        "\tsGamma2/s-1\tA1 \tsA1\t f\t sf\n"
            x_arg = "ang"
        else:
            if mode_single:
                first_header = "#q/A-1 \t\tGamma/s-1 \tsGamma/s-1\tf\t sf\n"
            else:
                first_header = "#q/A-1 \t\tGamma1/s-1 \tsGamma1/s-1 \tGamma2/s-1 "+\
                                        "\tsGamma2/s-1 \tA1 \t\tsA1\t f\t sf\n"
            x_arg = "q"
        

        save_file.write(first_header)
        for dataname in self.dat_files_list:
            dataset = self.dat_files_data[dataname]
            if mode_single:
                savetxt = precision.format(dataset[x_arg]) +"\t"+\
                        precision.format(dataset["Gamma1"]) + "\t" +\
                        precision.format(dataset["sGamma1"])+ "\t"+\
                        precision.format(dataset["f"]) + "\t" +\
                        precision.format(dataset["sf"])+ "\n"
            else:
                savetxt = precision.format(dataset[x_arg]) +"\t"+\
                        precision.format(dataset["Gamma1"]) + "\t" +\
                        precision.format(dataset["sGamma1"])+ "\t"+\
                        precision.format(dataset["Gamma2"]) + "\t" +\
                        precision.format(dataset["sGamma2"])+ "\t"+\
                        precision.format(dataset["A1"]) + " \t" +\
                        precision.format(dataset["sA1"])+"\t"+\
                        precision.format(dataset["f"]) + "\t" +\
                        precision.format(dataset["sf"])+ "\n"
            if not dataset["valid"]:
                savetxt = "#"+savetxt
            save_file.write(savetxt)
        save_file.close()
        
        pickle.dump ( (self.dat_files_list, self.dat_files_data),\
                    open(self.data_folder_path+"/DLS_fitting.p", "wb"))
        print("Saved data to " + self.data_folder_path+"/DLS_results.xye")
        print("All data used for fitting is binary stored in " +\
                                self.data_folder_path+"/DLS_fitting.p")
    
    def save_counts(self, event):
        save_file = open(self.data_folder_path+"/DLS_counts.xye", "w")
        precision = "{:.4e}"
        save_file.write("#DLS counts extracted using Dominiques two mode fitting "+\
                        "script v" + str(version) + "\n")
        save_file.write("#For questions contact: Dominique.Dresen@uni-koeln.de\n")
        save_file.write("#Read .dat files from folder: " + self.data_folder_path + "\n")
        save_file.write("#Skipped " + str(self.n_skip) + " datapoints "+\
                        "at beginning while reading tau, g2 data from each .dat file.\n")
        save_file.write("#Mean duration of measurements: " + str(self.mean_dur) + " s\n")
        save_file.write("#Refractive index: " + str(self.mean_n) + "\n")
        save_file.write("#Wavelength: " + str(self.mean_wavelength) + " nm\n")
        
        save_file.write("#2th/° \tf_mean/kHz\tf_std/kHz\n")
        for dataname in self.dat_files_list:
            dataset = self.dat_files_data[dataname]
            mean_frequency = np.mean(dataset['freq'])
            std_frequency = np.std(dataset['freq'], ddof=1)
            savetxt = precision.format(dataset['ang']) +"\t"+\
                        precision.format(mean_frequency) + "\t" +\
                        precision.format(std_frequency)+ "\n"
            if not dataset["valid"]:
                savetxt = "#"+savetxt
            save_file.write(savetxt)
        save_file.close()
    
    def save_data_theta_Gamma(self, event):
        self.save_data(event, save_thG=True)

    def replot(self):
        try:
            gamma1 = float(self.tgamma1.text())
            gamma2 = float(self.tgamma2.text())
            A1 = float(self.tA1.text())
            f = float(self.tf.text())

            init_p = lmfit.Parameters()
            init_p.add("f", f, min=0)
            init_p.add("Gamma1", gamma1)
            init_p.add("Gamma2", gamma2)
            init_p.add("A1", A1, min=0, max=1)

            self.dat_files_data[self.dat_files_list[self.index]]["A1"] = A1
            self.dat_files_data[self.dat_files_list[self.index]]["Gamma1"] = gamma1
            self.dat_files_data[self.dat_files_list[self.index]]["Gamma2"] = gamma2
            self.dat_files_data[self.dat_files_list[self.index]]["f"] = f
            self.dat_files_data[self.dat_files_list[self.index]]["g2_fit"] =\
                self.calc_g2(init_p, self.dat_files_data[self.dat_files_list[self.index]]["tau"])
            self.show_result_tab_plot()
            self.update_plots()
        except TypeError:
            print("Non float number entered in box")
    
    def plot2d(self):
        tau = self.dat_files_data[self.dat_files_list[self.index]]["tau"]
        q2_vals = []
        g2_vals = []
#        Nq2 = len(self.dat_files_list)
#        g2_array = np.zeros((Nq2, len(tau)))
        for i in range(len(self.dat_files_list)):
            current_file = self.dat_files_data[self.dat_files_list[i]]
            if not current_file["valid"]:
                continue
            q2 = current_file["q"]**2
            g2 = current_file["g2"]
            
            already_loaded = False
            for idx, other_q2val in enumerate(q2_vals):
                if np.allclose(other_q2val, q2):
                    already_loaded = True
                    loaded_idx = idx
                    break
                    
            if already_loaded:
                g2_vals[loaded_idx].append(g2)
            else:
                q2_vals.append(q2)
                g2_vals.append([g2])
        
        g2_array = np.zeros((len(q2_vals), len(tau)))
        for ix, loaded_g2 in enumerate(g2_vals):
            g2_array[ix, :] = np.sum(loaded_g2, axis=0)/len(loaded_g2)
            
        self.plotdialog = PlotDialog(q2_vals, tau, g2_array, self)
        
        self.plotdialog.resize(800,800)
        self.plotdialog.show()
        self.plotdialog.exec_()

    def refit(self, event):
        try:
            gamma1 = float(self.tgamma1.text())
            gamma2 = float(self.tgamma2.text())
            A1 = float(self.tA1.text())
            f = float(self.tf.text())
            
            init_p = lmfit.Parameters()
            init_p.add("f", f, min=0, vary=self.cf.isChecked())
            init_p.add("Gamma1", gamma1, min=0, vary=self.cgamma1.isChecked())
            init_p.add("Gamma2", gamma2, min=0, vary=self.cgamma2.isChecked())
            init_p.add("A1", A1, min=0, max=1, vary=self.cA1.isChecked())
            tau = self.dat_files_data[self.dat_files_list[self.index]]["tau"]
            fit_result = lmfit.minimize(self.residuum, init_p,\
                         args=(tau, self.dat_files_data[self.dat_files_list[self.index]]["g2"]),\
                         method="leastsq")

            self.dat_files_data[self.dat_files_list[self.index]]["A1"] =\
                                                fit_result.params["A1"].value
            self.dat_files_data[self.dat_files_list[self.index]]["sA1"] =\
                                                fit_result.params["A1"].stderr
            self.dat_files_data[self.dat_files_list[self.index]]["Gamma1"] =\
                                                fit_result.params["Gamma1"].value
            self.dat_files_data[self.dat_files_list[self.index]]["sGamma1"] =\
                                                fit_result.params["Gamma1"].stderr
            self.dat_files_data[self.dat_files_list[self.index]]["Gamma2"] =\
                                                fit_result.params["Gamma2"].value
            self.dat_files_data[self.dat_files_list[self.index]]["sGamma2"] =\
                                                fit_result.params["Gamma2"].stderr
            self.dat_files_data[self.dat_files_list[self.index]]["f"] =\
                                                fit_result.params["f"].value
            self.dat_files_data[self.dat_files_list[self.index]]["sf"] =\
                                                fit_result.params["f"].stderr
            self.dat_files_data[self.dat_files_list[self.index]]["g2_fit"] =\
                                                self.calc_g2(fit_result.params, tau)
            self.show_result_tab_plot()
            self.update_plots()
        except TypeError:
            print("Non float number entered in box")
    
    def singlemodefit(self):
        try:
            gamma1 = float(self.tgamma1.text())
            f = float(self.tf.text())
            gamma2 = 0
            A1 = 1
            init_p = lmfit.Parameters()
            init_p.add("f", f, min=0, vary=self.cf.isChecked())
            init_p.add("Gamma", gamma1, min=0, vary=self.cgamma1.isChecked())
            
            tau = self.dat_files_data[self.dat_files_list[self.index]]["tau"]
            fit_result = lmfit.minimize(self.residuum_single, init_p,\
                         args=(tau, self.dat_files_data[self.dat_files_list[self.index]]["g2"]),\
                         method="leastsq")

            self.dat_files_data[self.dat_files_list[self.index]]["A1"] = 1
            self.dat_files_data[self.dat_files_list[self.index]]["sA1"] = 0
            self.dat_files_data[self.dat_files_list[self.index]]["Gamma1"] =\
                    fit_result.params["Gamma"].value
            self.dat_files_data[self.dat_files_list[self.index]]["sGamma1"] =\
                    fit_result.params["Gamma"].stderr
            self.dat_files_data[self.dat_files_list[self.index]]["Gamma2"] = 0
            self.dat_files_data[self.dat_files_list[self.index]]["sGamma2"] = 0
            self.dat_files_data[self.dat_files_list[self.index]]["f"] =\
                    fit_result.params["f"].value
            self.dat_files_data[self.dat_files_list[self.index]]["sf"] =\
                    fit_result.params["f"].stderr
            self.dat_files_data[self.dat_files_list[self.index]]["g2_fit"] =\
                    self.calc_g2_single(fit_result.params, tau)
            self.show_result_tab_plot()
            self.update_plots()
        except TypeError:
            print("Non float number entered in box")
            
    def singlemodefitall(self):
        for ifile, dat_file in enumerate(self.list_of_dat_files):
            self.index = ifile
            self.update_plots()
            QtCore.QCoreApplication.processEvents()
            self.singlemodefit()
            QtCore.QCoreApplication.processEvents()
        
        
    def load_dat_file(self, file_path):
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
        
        tau = tau[self.n_skip:]
        g2 = g2[self.n_skip:]
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

    def save_corr_freq(self):
        t_freq = self.current_data['t_freq']
        freq = self.current_data['freq']
        tau = self.current_data['tau']
        g2 = self.current_data['g2']
        
        with open('0AutocorrelationFunction.dat', 'w') as f:
            f.write('#tau / s \t g2\n')
            for i, tau_val in enumerate(tau):
                f.write(str(tau_val) + '\t' + str(g2[i]) + '\n' )
        
        with open('0CountFrequency.dat', 'w') as f:
            f.write('#t_freq / s \t freq/kHz\n')
            for i, t_val in enumerate(t_freq):
                f.write(str(t_val) + '\t' + str(freq[i]) + '\n' )
        

    def calc_g2(self, p, tau):
        f = p["f"].value
        Gamma1 = p["Gamma1"].value
        Gamma2 = p["Gamma2"].value
        A1 = p["A1"].value
        g1 = A1*np.exp(-tau*Gamma1) + (1.-A1)*np.exp(-tau*Gamma2)
        return 1 + f*g1**2
        # return fortranDLS.dls.calc_g2_twog(tau, A1, Gamma1, Gamma2, 1., f)

    def residuum(self, p, tau, g2):
        return g2 - self.calc_g2(p, tau)

    def calc_g2_single(self, p, tau):
        f = p["f"].value
        Gamma = p["Gamma"].value
        g1 = np.exp(-tau*Gamma)
        return 1 + f*g1**2
        # return fortranDLS.dls.calc_g2(tau, Gamma, 1., f)

    def residuum_single(self, p, tau, g2):
        return g2 - self.calc_g2_single(p, tau)

    def fit_gamma(self, dataset):
        init_p = lmfit.Parameters()
        init_p.add("f", 1, min=0)
        init_p.add("Gamma1", 100, min=0, max=1e6)
        init_p.add("Gamma2", 10000, min=0, max=1e6)
        init_p.add("A1", 0.5, min=0, max=1)
        
        fit_result = lmfit.minimize(self.residuum, init_p, args=(dataset["tau"], dataset["g2"]), method="leastsq")
        
        dataset["A1"] = 1-fit_result.params["A1"].value
        dataset["sA1"] = fit_result.params["A1"].stderr
        dataset["Gamma1"] = fit_result.params["Gamma2"].value
        dataset["sGamma1"] = fit_result.params["Gamma2"].stderr
        dataset["Gamma2"] = fit_result.params["Gamma1"].value
        dataset["sGamma2"] = fit_result.params["Gamma1"].stderr
        dataset["f"] = fit_result.params["f"].value
        dataset["sf"] = fit_result.params["f"].stderr
        dataset["g2_fit"] = self.calc_g2(fit_result.params, dataset["tau"])
        dataset["valid"] = True
        
        return dataset

class PlotDialog(pyqt5widget.QDialog):
    def __init__(self, q2_vals, tau, g2_array, parent=None):
        super().__init__(parent)
        
        #Nice Coloring:
        c = mpl.colors.ColorConverter().to_rgb
        custom_colors = [(0, 0, 0, 0),\
                 (0.18, 0.05, 0.05, 0.2),\
                 (0.28, 0, 0, 1),\
                 (0.4, 0.7, 0.85, 0.9),\
                 (0.45, 0, 0.75, 0),\
                 (0.6, 1, 1, 0),\
                 (0.75, 1, 0, 0),\
                 (0.92 , 0.6, 0.6, 0.6),\
                 (1  , 0.95, 0.95, 0.95)]
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, item in enumerate(custom_colors):
            pos, r, g, b = item
            cdict['red'].append([pos, r, r])
            cdict['green'].append([pos, g, g])
            cdict['blue'].append([pos, b, b])
        self.cmap = mpl.colors.LinearSegmentedColormap('CustomMap', cdict)
        self.cmap.set_bad(color='black')


        self.fig = plt.figure(figsize=(8/2.54, 8/2.54))
        self.ax = self.fig.add_axes([0.2, 0.15, 0.6, 0.8])
        im = self.ax.pcolormesh(q2_vals, tau, g2_array.T, cmap=self.cmap)
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label("$g_2$")
        self.ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        self.ax.set_yscale('log')
        self.ax.set_xlim([min(q2_vals), max(q2_vals)])
        self.ax.set_ylim([min(tau), max(tau)])
        self.ax.set_xlabel("$q^2 \, / \, m^{-2}$")
        self.ax.set_ylabel(r"$\tau \, / \, s$")
        self.canvas = FigureCanvas(self.fig)

        # use addToolbar to add toolbars to the main window directly!
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout = pyqt5widget.QVBoxLayout()
        layout.addWidget(self.canvas, 0)
        layout.addWidget(self.toolbar, 1)
        

        self.main_widget = pyqt5widget.QWidget(self)
        self.main_widget.setLayout(layout)

        
if __name__=='__main__':
    qApp = pyqt5widget.QApplication(sys.argv)
    main_window = DLSViewer()
    main_window.resize(1248, 900)
    main_window.show()
    sys.exit(qApp.exec_())   
