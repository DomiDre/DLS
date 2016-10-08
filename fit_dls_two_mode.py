import matplotlib as mpl
mpl.use("Qt5Agg")


from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas,\
        NavigationToolbar2QT as NavigationToolbar
import PyQt5.QtWidgets as pyqt5widget

#from matplotlib.backends.backend_qt4agg \
#    import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg \
#    import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import sys, os, glob, lmfit, pickle, fortranDLS, warnings

# remove some annoying deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

mpl.rcParams['font.size'] = 8
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
    
    tau = tau[5:]
    g2 = g2[5:]
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


def calc_g2(p, tau):
    f = p["f"].value
    Gamma1 = p["Gamma1"].value
    Gamma2 = p["Gamma2"].value
    A1 = p["A1"].value
    return fortranDLS.dls.calc_g2_twog(tau, A1, Gamma1, Gamma2, 1., f)

def residuum(p, tau, g2):
    return g2 - calc_g2(p, tau)

def calc_g2_single(p, tau):
    f = p["f"].value
    Gamma = p["Gamma"].value
    return fortranDLS.dls.calc_g2(tau, Gamma, 1., f)

def residuum_single(p, tau, g2):
    return g2 - calc_g2_single(p, tau)

def fit_gamma(dataset):
    init_p = lmfit.Parameters()
    init_p.add("f", 1, min=0)
    init_p.add("Gamma1", 100, min=0, max=1e6)
    init_p.add("Gamma2", 10000, min=0, max=1e6)
    init_p.add("A1", 0.5, min=0, max=1)
    
    fit_result = lmfit.minimize(residuum, init_p, args=(dataset["tau"], dataset["g2"]), method="leastsq")
    
    dataset["A1"] = 1-fit_result.params["A1"].value
    dataset["sA1"] = fit_result.params["A1"].stderr
    dataset["Gamma1"] = fit_result.params["Gamma2"].value
    dataset["sGamma1"] = fit_result.params["Gamma2"].stderr
    dataset["Gamma2"] = fit_result.params["Gamma1"].value
    dataset["sGamma2"] = fit_result.params["Gamma1"].stderr
    dataset["f"] = fit_result.params["f"].value
    dataset["sf"] = fit_result.params["f"].stderr
    dataset["g2_fit"] = calc_g2(fit_result.params, dataset["tau"])
    dataset["valid"] = True
    
    return dataset

class DLSViewer(pyqt5widget.QMainWindow):
    def __init__(self, data, datlist, data_folder_path, parent = None):
        super(DLSViewer, self).__init__(parent)
        self.data = data
        self.datlist = datlist
        
        self.data_folder = data_folder_path
        
        self.index = 0
        self.N = len(self.datlist)
        
        self.current_data = None
        
        
        self.fig = plt.figure(figsize=(15/2.54, 10/2.54))
        self.ax11 = plt.subplot2grid((2, 3), (0, 0))
        self.ax12 = plt.subplot2grid((2, 3), (0, 1))
        
        self.plot11, = self.ax11.plot([], [], color='black', marker='.')
        self.ax11.set_xlabel(r"$\mathit{t} \, / \, s$")
        self.ax11.set_ylabel(r"$\mathit{f} \, / \, Hz$")
        
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
        
        self.lgamma2 = pyqt5widget.QLabel("Gamma2: ")
        self.tgamma2 = pyqt5widget.QLineEdit()
        
        self.lA1 = pyqt5widget.QLabel("A1: ")
        self.tA1 = pyqt5widget.QLineEdit()
        
        self.bplot = pyqt5widget.QPushButton('Replot')
        self.brefit = pyqt5widget.QPushButton('Refit')
        self.bsinglemodefit = pyqt5widget.QPushButton('Single Mode Fit')
        
        self.bnext.clicked.connect(self.next)
        self.bprev.clicked.connect(self.prev)
        self.bvalid.clicked.connect(self.toggle_valid)
        self.bsave.clicked.connect(self.save_data)
        self.bswitchGamma.clicked.connect(self.switch_gamma)
        self.bplot.clicked.connect(self.replot)
        self.brefit.clicked.connect(self.refit)
        self.bsinglemodefit.clicked.connect(self.singlemodefit)
        self.infolabel = pyqt5widget.QLabel("")
        self.main_widget = pyqt5widget.QWidget(self)
        self.setCentralWidget(self.main_widget)

        width = 600
        self.tgamma1.setFixedWidth(width)
        self.tgamma2.setFixedWidth(width)
        self.tA1.setFixedWidth(width)
        self.bnext.setFixedWidth(width)
        self.bprev.setFixedWidth(width)
        self.bvalid.setFixedWidth(width)
        self.bsave.setFixedWidth(width)
        self.bswitchGamma.setFixedWidth(width)
        self.bplot.setFixedWidth(width)
        self.brefit.setFixedWidth(width)
        self.lgamma1.setAlignment(QtCore.Qt.AlignRight)
        self.lgamma2.setAlignment(QtCore.Qt.AlignRight)
        self.lA1.setAlignment(QtCore.Qt.AlignRight)
        
        
        layout = pyqt5widget.QGridLayout()
        layout.addWidget(self.canvas, 0, 0, 1, 2)
        layout.addWidget(self.infolabel, 1, 0, 1, 2)
        
        layout.addWidget(self.bprev, 2, 0)
        layout.addWidget(self.bnext, 2, 1)
        layout.addWidget(self.bvalid, 3, 0)
        layout.addWidget(self.bsave, 3, 1)
        layout.addWidget(self.bswitchGamma, 4, 0)
        layout.addWidget(self.lgamma1, 5, 0)
        layout.addWidget(self.tgamma1, 5, 1)
        layout.addWidget(self.lgamma2, 6, 0)
        layout.addWidget(self.tgamma2, 6, 1)
        layout.addWidget(self.lA1, 7, 0)
        layout.addWidget(self.tA1, 7, 1)
        layout.addWidget(self.bplot, 8, 0)
        layout.addWidget(self.brefit, 8, 1)
        layout.addWidget(self.bsinglemodefit, 9, 0, 1, 2)
        
        
        self.main_widget.setLayout(layout)
        self.show_result_tab_plot()
        self.update_plots()
        
        self.save_data(None)

    def show_result_tab_plot(self):
        self.ax12.cla()
        self.ax12.set_xlabel(r"$\mathit{q^2} \, / \, \AA^{-2}$")
        self.ax12.set_ylabel(r"$\Gamma \, / \, s^{-1}$")
        
        for dataname in self.data:
            datafile = self.data[dataname]
            if datafile["valid"]:
                plot_color = 'black'
            else:
                plot_color = 'red'
            self.ax12.errorbar(datafile["q"]**2, datafile["Gamma1"], datafile["sGamma1"], color=plot_color, marker='.')
            #self.ax12.errorbar(datafile["q"]**2, datafile["Gamma2"], datafile["sGamma2"], color=plot_color2, marker='.')
        
        self.ax12.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        self.current_point12, = self.ax12.plot([], [], marker=r'$\circ$', color='red', markersize=10)
        self.fig.tight_layout()
        self.canvas.draw()

    def update_info_label(self):
        self.current_data = self.data[self.datlist[self.index]]
        self.infolabel.setText("Opened file: " + self.datlist[self.index] + "\n" +
                                "2 th = " + str(self.current_data["ang"]) + " deg\n"+
                                "n = " + str(self.current_data["n"]) + "\n"+
                                "wavelength = " + str(self.current_data["wavelength"]) + " nm\n"+
                                "T  = " + str(self.current_data["T"]) + " K\n"+
                                "Measurement Time = " + str(self.current_data["dur"]) + " s\n"+
                                "q = " + "{:.3e}".format(self.current_data["q"]) + " A-1\n"+
                                "A1 = " + "{:.3e}".format(self.current_data["A1"]) + " +/- " + "{:.3e}".format(self.current_data["sA1"]) + "s-1\n"+
                                "Gamma1 = " + "{:.3e}".format(self.current_data["Gamma1"]) + " +/- " + "{:.3e}".format(self.current_data["sGamma1"]) + "s-1\n"+
                                "Gamma2 = " + "{:.3e}".format(self.current_data["Gamma2"]) + " +/- " + "{:.3e}".format(self.current_data["sGamma2"]) + "s-1\n"+
                                "f = " + "{:.3e}".format(self.current_data["f"]) + " +/- " + "{:.3e}".format(self.current_data["sf"]) + "\n"+
                                "Valid: " + str(self.current_data["valid"]))
        self.tgamma1.setText("{:.3e}".format(self.current_data["Gamma1"]))
        self.tgamma2.setText("{:.3e}".format(self.current_data["Gamma2"]))
        self.tA1.setText("{:.3e}".format(self.current_data["A1"]))
            
            
    def update_plots(self):
        self.current_data = self.data[self.datlist[self.index]]
        self.plot11.set_data(self.current_data["t_freq"], self.current_data["freq"])
        self.ax11.set_xlim([self.current_data["t_freq"][0], self.current_data["t_freq"][-1]])
        self.ax11.set_ylim([0.8*min(self.current_data["freq"]), 1.2*max(self.current_data["freq"])])
        
        self.current_point12.set_data(self.current_data["q"]**2, self.current_data["Gamma1"])
        
        #self.plot13.set_data(self.current_data["SON_Gamma"],self.current_data["SON_A"])
        #self.ax13.set_xlim([self.current_data["SON_Gamma"][0], self.current_data["SON_Gamma"][-1]])
        #self.ax13.set_ylim([0, 1.2])
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
        self.data[self.datlist[self.index]]["valid"] = not self.data[self.datlist[self.index]]["valid"]
        self.update_plots()
        self.show_result_tab_plot()

    def switch_gamma(self, event):
        self.data[self.datlist[self.index]]["Gamma1"], self.data[self.datlist[self.index]]["Gamma2"] =\
                self.data[self.datlist[self.index]]["Gamma2"], self.data[self.datlist[self.index]]["Gamma1"]
        self.data[self.datlist[self.index]]["sGamma1"], self.data[self.datlist[self.index]]["sGamma2"] =\
                self.data[self.datlist[self.index]]["sGamma2"], self.data[self.datlist[self.index]]["sGamma1"]
        self.data[self.datlist[self.index]]["A1"] =\
                1-self.data[self.datlist[self.index]]["A1"]
        self.update_plots()
        self.show_result_tab_plot()
    
    
    def save_data(self, event):
        save_file = open(self.data_folder+"/TwoModeRESULT.TAB", "w")
        precision = "{:.4e}"
        if "-saveth" in sys.argv:
            save_file.write("#2th/° \tGamma1/s-1 \tsGamma1\tGamma2/s-1 \tsGamma2/s-1\tA1 \tsA1\t f\t sf\n")
            for dataname in self.datlist:
                dataset = self.data[dataname]
                
                if dataset["valid"]:
                    save_file.write(precision.format(dataset["ang"]) +" \t"+\
                                    precision.format(dataset["Gamma1"]) + " \t" + precision.format(dataset["sGamma1"])+ " \t"+\
                                    precision.format(dataset["Gamma2"]) + " \t" + precision.format(dataset["sGamma2"])+ " \t"+\
                                    precision.format(dataset["A1"]) + " \t" + precision.format(dataset["sA1"])+"\t"+\
                                    precision.format(dataset["f"]) + " \t" + precision.format(dataset["sf"])+"\n")
            save_file.close()
        else:
            save_file.write("#q/A-1 \t\tGamma1/s-1 \tsGamma1 \tGamma2/s-1 \tsGamma2/s-1 \tA1 \t\tsA1\t f\t sf\n")
            for dataname in self.datlist:
                dataset = self.data[dataname]
                
                if dataset["valid"]:
                        save_file.write(precision.format(dataset["q"]) + " \t" +\
                                        precision.format(dataset["Gamma1"]) + " \t" + precision.format(dataset["sGamma1"])+ " \t"+\
                                        precision.format(dataset["Gamma2"]) + " \t" + precision.format(dataset["sGamma2"])+ " \t"+\
                                        precision.format(dataset["A1"]) + " \t" + precision.format(dataset["sA1"])+"\t"+\
                                        precision.format(dataset["f"]) + " \t" + precision.format(dataset["sf"])+"\n")
            save_file.close()
        
        pickle.dump ( (self.datlist, self.data), open(self.data_folder+"/TwoModeFittingData.p", "wb"))
        print("Saved data to " + self.data_folder+"/TwoModeRESULT.TAB")
        print("All data used for fitting is binary stored in  " + self.data_folder+"/TwoModeFittingData.p")
            
    def replot(self, event):
        try:
            gamma1 = float(self.tgamma1.text())
            gamma2 = float(self.tgamma2.text())
            A1 = float(self.tA1.text())
            f = self.data[self.datlist[self.index]]["f"]
            init_p = lmfit.Parameters()
            init_p.add("f", f, min=0)
            init_p.add("Gamma1", gamma1)
            init_p.add("Gamma2", gamma2)
            init_p.add("A1", A1, min=0, max=1)

            #fit_result = lmfit.minimize(residuum, init_p, args=(dataset["tau"], dataset["g2"]), method="leastsq")

            self.data[self.datlist[self.index]]["A1"] = A1
            self.data[self.datlist[self.index]]["Gamma1"] = gamma1
            self.data[self.datlist[self.index]]["Gamma2"] = gamma2
            self.data[self.datlist[self.index]]["g2_fit"] = calc_g2(init_p, self.data[self.datlist[self.index]]["tau"])
            self.show_result_tab_plot()
            self.update_plots()
        except TypeError:
            print("Non float number entered in box")
    
    def refit(self, event):
        try:
            gamma1 = float(self.tgamma1.text())
            gamma2 = float(self.tgamma2.text())
            A1 = float(self.tA1.text())
            f = self.data[self.datlist[self.index]]["f"]
            init_p = lmfit.Parameters()
            init_p.add("f", f, min=0)
            init_p.add("Gamma1", gamma1, min=0)
            init_p.add("Gamma2", gamma2, min=0)
            init_p.add("A1", A1, min=0, max=1)
            tau = self.data[self.datlist[self.index]]["tau"]
            fit_result = lmfit.minimize(residuum, init_p, args=(tau, self.data[self.datlist[self.index]]["g2"]), method="leastsq")

            self.data[self.datlist[self.index]]["A1"] = fit_result.params["A1"].value
            self.data[self.datlist[self.index]]["sA1"] = fit_result.params["A1"].stderr
            self.data[self.datlist[self.index]]["Gamma1"] = fit_result.params["Gamma1"].value
            self.data[self.datlist[self.index]]["sGamma1"] = fit_result.params["Gamma1"].stderr
            self.data[self.datlist[self.index]]["Gamma2"] = fit_result.params["Gamma2"].value
            self.data[self.datlist[self.index]]["sGamma2"] = fit_result.params["Gamma2"].stderr
            self.data[self.datlist[self.index]]["f"] = fit_result.params["f"].value
            self.data[self.datlist[self.index]]["sf"] = fit_result.params["f"].stderr
            self.data[self.datlist[self.index]]["g2_fit"] = calc_g2(fit_result.params, tau)
            self.show_result_tab_plot()
            self.update_plots()
        except TypeError:
            print("Non float number entered in box")
    
    def singlemodefit(self, event):
        try:
            gamma1 = float(self.tgamma1.text())
            gamma2 = 0
            A1 = 1
            f = self.data[self.datlist[self.index]]["f"]
            init_p = lmfit.Parameters()
            init_p.add("f", f, min=0)
            init_p.add("Gamma", gamma1, min=0)
            
            tau = self.data[self.datlist[self.index]]["tau"]
            fit_result = lmfit.minimize(residuum_single, init_p, args=(tau, self.data[self.datlist[self.index]]["g2"]), method="leastsq")

            self.data[self.datlist[self.index]]["A1"] = 1
            self.data[self.datlist[self.index]]["sA1"] = 0
            self.data[self.datlist[self.index]]["Gamma1"] = fit_result.params["Gamma"].value
            self.data[self.datlist[self.index]]["sGamma1"] = fit_result.params["Gamma"].stderr
            self.data[self.datlist[self.index]]["Gamma2"] = 0
            self.data[self.datlist[self.index]]["sGamma2"] = 0
            self.data[self.datlist[self.index]]["f"] = fit_result.params["f"].value
            self.data[self.datlist[self.index]]["sf"] = fit_result.params["f"].stderr
            self.data[self.datlist[self.index]]["g2_fit"] = calc_g2_single(fit_result.params, tau)
            self.show_result_tab_plot()
            self.update_plots()
        except TypeError:
            print("Non float number entered in box")
if __name__=='__main__':
    numargs = len(sys.argv) -1
    if numargs == 0:
        print("Usage: python fit_dls_two_mode.py dls_folder_path [..]")
        print("Possible Parameters:")
        print("-saveth\tSave with two theta in first column.")
            
    else:
        data_folder_path = sys.argv[1]
        if not os.path.isdir(data_folder_path):
                sys.exit("Folder path: " + str(data_folder_path) + " does not exist.")
        
        if os.path.exists(data_folder_path+"/TwoModeFittingData.p"):
                print("Loading previous fitting results from " + data_folder_path+"/TwoModeFittingData.p")
                dat_files_list, dat_files_data = pickle.load( open( data_folder_path+"/TwoModeFittingData.p", "rb" ) )
        else:
                list_of_dat_files = sorted(glob.glob(data_folder_path + "/*.DAT"))
                if len(list_of_dat_files) == 0:
                        sys.exit("No *.DAT files in " + str(data_folder_path))
                
                dat_files_list = []
                dat_files_data = {}
                for dat_file in list_of_dat_files:
                        datname = os.path.basename(dat_file).split(".DAT")[0]
                        dat_files_list.append(datname)
                        dat_files_data[datname] = load_dat_file(dat_file)
                        dat_files_data[datname] = fit_gamma( dat_files_data[datname] )
                
        qApp = pyqt5widget.QApplication(sys.argv)
        main_window = DLSViewer(dat_files_data, dat_files_list, data_folder_path)
        main_window.resize(1248, 900)
        main_window.show()
        sys.exit(qApp.exec_())   
