# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\droxl\Documents\EPFL\MA2\SHS\values.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from Indicator_function import *
from Graphic_output import *
from Table_output import *
from Graphic_output import *

class Ui_ValuesWindow(object):

    def setupUi(self, MainWindow, complete_dictionnary, indicator_dictionnary,t):
        self.image_width = 800
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 776)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, self.image_width, 731))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("node_style.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")

        # Have the data from previous tree in self
        self.t = t
        self.complete_dictionnary = complete_dictionnary
        self.indicator_dictionnary = indicator_dictionnary

        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(self.image_width + 20, 0, 111, 631))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.doubleSpinBox = []
        self.doubleSpinBox2 = []
        self.doubleSpinBox3 = []

        self.nb_column = 1
        self.nb_output = 0
        self.name_indic = []
        self.minmax_dictionnary = {}

        for node  in self.t.traverse("postorder"):
            if node.is_leaf():
                # ROMAIN
                if self.indicator_dictionnary[node.name]["binary"] == False:
                    self.minmax_dictionnary[node.name] = (indicator_dictionnary[node.name]["x_min"], indicator_dictionnary[node.name]["x_max"])
                else :
                    self.minmax_dictionnary[node.name] = (0,1)


        for i, (min, max) in self.minmax_dictionnary.items():
            self.doubleSpinBox.append(QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget))
            self.doubleSpinBox[-1].setMinimum(float(min))
            self.doubleSpinBox[-1].setMaximum(float(max))
            self.doubleSpinBox[-1].setObjectName("doubleSpinBox_"+i)
            self.verticalLayout.addWidget(self.doubleSpinBox[-1])
        
        self.nb_column = self.nb_column + 1

        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(self.image_width+ 140, 0, 111, 631))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(self.image_width+ 260, 0, 111, 631))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")

        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        self.New = QtWidgets.QPushButton(self.centralwidget)
        self.New.setGeometry(QtCore.QRect(self.image_width + 60, 660, 75, 23))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.New.setFont(font)
        self.New.setObjectName("New")
        self.New.clicked.connect(self.new_column)

        self.Copy = QtWidgets.QPushButton(self.centralwidget)
        self.Copy.setGeometry(QtCore.QRect(self.image_width + 150, 660, 75, 23))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.Copy.setFont(font)
        self.Copy.setObjectName("Copy")
        self.Copy.clicked.connect(self.copy)

        self.Redo = QtWidgets.QPushButton(self.centralwidget)
        self.Redo.setGeometry(QtCore.QRect(self.image_width + 240, 660, 75, 23))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.Redo.setFont(font)
        self.Redo.setObjectName("Reset")
        self.Redo.clicked.connect(self.reset)

        self.Next = QtWidgets.QPushButton(self.centralwidget)
        self.Next.setGeometry(QtCore.QRect(self.image_width + 200,700, 121, 31))
        # self.Next.setGeometry(QtCore.QRect(self.image_width + 200,0, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Next.setFont(font)
        self.Next.setIconSize(QtCore.QSize(56, 56))
        self.Next.setCheckable(True)
        self.Next.setObjectName("Next")
        self.Next.clicked.connect(self.next_page)

        self.Previous = QtWidgets.QPushButton(self.centralwidget)
        self.Previous.setGeometry(QtCore.QRect(self.image_width + 50, 700, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Previous.setFont(font)
        self.Previous.setIconSize(QtCore.QSize(56, 56))
        self.Previous.setCheckable(True)
        self.Previous.setObjectName("Previous")
        self.Previous.clicked.connect(MainWindow.close)


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 935, 22))
        self.menubar.setObjectName("menubar")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MIVES"))
        self.Next.setText(_translate("MainWindow", "Next"))
        self.Previous.setText(_translate("MainWindow", "Previous"))
        self.New.setText(_translate("MainWindow", "New"))
        self.Copy.setText(_translate("MainWindow", "Copy"))
        self.Redo.setText(_translate("MainWindow", "Redo"))


    def new_column(self):

        if(self.nb_column == 2):
    
            for i, (min, max) in self.minmax_dictionnary.items():
                self.doubleSpinBox2.append(QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget_2))
                self.doubleSpinBox2[-1].setMinimum(float(min))
                self.doubleSpinBox2[-1].setMaximum(float(max))
                self.doubleSpinBox2[-1].setObjectName("doubleSpinBox2_"+i)
                self.verticalLayout_2.addWidget(self.doubleSpinBox2[-1])
            self.nb_column = self.nb_column + 1
            return
        
        if(self.nb_column == 3): 

            for i, (min, max) in self.minmax_dictionnary.items():
                self.doubleSpinBox3.append(QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget_3))
                self.doubleSpinBox3[-1].setMinimum(float(min))
                self.doubleSpinBox3[-1].setMaximum(float(max))
                self.doubleSpinBox3[-1].setObjectName("doubleSpinBox3_"+i)
                self.verticalLayout_3.addWidget(self.doubleSpinBox3[-1])
            self.nb_column = self.nb_column + 1
            return 

    def copy(self):
        
        if(self.nb_column == 2):
    
            for i, (crit, (min, max)) in enumerate(self.minmax_dictionnary.items()):
                self.doubleSpinBox2.append(QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget_2))
                self.doubleSpinBox2[-1].setMinimum(float(min))
                self.doubleSpinBox2[-1].setMaximum(float(max))
                self.doubleSpinBox2[-1].setObjectName("doubleSpinBox2_"+crit)
                self.verticalLayout_2.addWidget(self.doubleSpinBox2[-1])
                self.doubleSpinBox2[-1].setValue(self.doubleSpinBox[i].value())

            self.nb_column = self.nb_column + 1
            return
        
        if(self.nb_column == 3): 

            for i, (crit, (min, max)) in enumerate(self.minmax_dictionnary.items()):
                self.doubleSpinBox3.append(QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget_3))
                self.doubleSpinBox3[-1].setMinimum(float(min))
                self.doubleSpinBox3[-1].setMaximum(float(max))
                self.doubleSpinBox3[-1].setObjectName("doubleSpinBox3_"+crit)
                self.verticalLayout_3.addWidget(self.doubleSpinBox3[-1])
                self.doubleSpinBox3[-1].setValue(self.doubleSpinBox2[i].value())

            self.nb_column = self.nb_column + 1
            return 

    def reset(self):

        if(self.nb_column == 3 or self.nb_column == 4):
            for i, crit in enumerate(self.minmax_dictionnary.items()):
                self.doubleSpinBox2[i].deleteLater() 
                
            self.doubleSpinBox2 = []
        
        if(self.nb_column == 4):
            for i, crit in enumerate(self.minmax_dictionnary.items()): 
                self.doubleSpinBox3[i].deleteLater() 
                
            self.doubleSpinBox3 = []

        for i, (crit, (min, max)) in enumerate(self.minmax_dictionnary.items()): 
                self.doubleSpinBox[i].setValue(float(min))

        self.nb_column = 2
        return

    def next_page(self):

        self.output_dict = {}

        value1 = []
        value2 = []
        value3 = []

        for i, crit in enumerate (self.minmax_dictionnary.items()):
            self.output_dict[crit[0]] = (self.doubleSpinBox[i].value())
            if(self.nb_column == 3 or self.nb_column == 4):
                self.output_dict[crit[0]] = (self.doubleSpinBox[i].value(), self.doubleSpinBox2[i].value())
            if(self.nb_column == 4):
                self.output_dict[crit[0]] = (self.doubleSpinBox[i].value(), self.doubleSpinBox2[i].value(), self.doubleSpinBox3[i].value())
        # Here we should get the dictionnary from Coline's work: matching each indicator with its value input
            
        values_dict = self.output_dict
        computed_value_for_indicator_dict = {}

        # We need to check if the number of columns is greater than 1 or not
        if self.nb_column-1 > 1:
            for node in self.t.traverse("postorder"):
                if  node.is_leaf()== True:
                    # It is an indicator
                    indicator_dict  = self.complete_dictionnary[node.name]
                    indicator_value = values_dict[node.name]
                    weight = float(indicator_dict["weight"])
                    x_min = float(indicator_dict["x_min"])
                    x_max = float(indicator_dict["x_max"])
                    geometric_P = float(indicator_dict["geometric_P"])
                    geometric_K = float(indicator_dict["geometric_K"])
                    geometric_C = float(indicator_dict["geometric_C"])
                    infl_point_coord = [geometric_C,geometric_K]
                    binary = int(indicator_dict["binary"])
                    descending = int(indicator_dict["descending"])
                    computed_value = []
                    computed_value_for_indicator_dict[node.name] = []
                    for ind_val in indicator_value:
                        if binary:
                            if descending:
                                if ind_val == 0:
                                    computed_value.append(1)
                                else:
                                    computed_value.append(0)
                            else:
                                if ind_val == 0:
                                    computed_value.append(0)
                                else:
                                    computed_value.append(1)
                        else:
                            computed_value = evaluate_function(geometric_P,infl_point_coord,x_min,x_max,ind_val,descending)
                        computed_value_for_indicator_dict[node.name].append(computed_value)

            computed_value_for_criteria_dict = {}
            for node in self.t.traverse("postorder"):
                if node.is_leaf() == False and node.is_root() == False and node.up.up!= None : #It's a criteria
                    computed_value_for_criteria_dict[node.name] = []
                    for i in range (0,self.nb_column-1):
                        criteria_value = 0
                        for ind in node.get_children(): #Indicators are children of criterias
                            criteria_value = criteria_value + float(computed_value_for_indicator_dict[ind.name])*float(self.complete_dictionnary[ind.name]["weight"])
                        computed_value_for_criteria_dict[node.name].append(criteria_value)

            computed_value_for_pillars_dict = {}
            final_score = np.zeros(self.nb_column-1)
            for node in self.t.traverse("postorder"):
                if  node.is_root() == False and node.up.is_root(): # Then it's a pillar
                    computed_value_for_pillars_dict[node.name] = []
                    for i in range (0,self.nb_column-1):
                        pillar_value = 0
                        for crit in node.get_children():
                            pillar_value = pillar_value + computed_value_for_criteria_dict[crit.name][i]*self.complete_dictionnary[crit.name]
                        computed_value_for_pillars_dict[node.name].append(pillar_value)
                        final_score[i] = final_score[i] + pillar_value*self.complete_dictionnary[node.name]


        else:
            for node in self.t.traverse("postorder"):
                if  node.is_leaf()== True:
                    # It is an indicator
                    indicator_dict  = self.complete_dictionnary[node.name]
                    indicator_value = values_dict[node.name]
                    weight = float(indicator_dict["weight"])
                    x_min = float(indicator_dict["x_min"])
                    x_max = float(indicator_dict["x_max"])
                    geometric_P = float(indicator_dict["geometric_P"])
                    geometric_K = float(indicator_dict["geometric_K"])
                    geometric_C = float(indicator_dict["geometric_C"])
                    infl_point_coord = [geometric_C,geometric_K]
                    binary = int(indicator_dict["binary"])
                    descending = int(indicator_dict["descending"])
                    if binary:
                        if descending:
                            if indicator_value == 0:
                                computed_value = 1
                            else:
                                computed_value = 0
                        else:
                            if indicator_value == 0:
                                computed_value = 0
                            else:
                                computed_value = 1
                    else:
                        computed_value = evaluate_function(geometric_P,infl_point_coord,x_min,x_max,indicator_value,descending)
                    computed_value_for_indicator_dict[node.name] = computed_value
            computed_value_for_criteria_dict = {}
            for node in self.t.traverse("postorder"):
                if node.is_leaf() == False and node.is_root() == False and node.up.up!= None : #It's a criteria
                    criteria_value = 0
                    for ind in node.get_children(): #Indicators are children of criterias
                        criteria_value = criteria_value + float(computed_value_for_indicator_dict[ind.name])*float(self.complete_dictionnary[ind.name]["weight"])
                    computed_value_for_criteria_dict[node.name] = criteria_value

            computed_value_for_pillars_dict = {}
            final_score = 0
            for node in self.t.traverse("postorder"):
                if  node.is_root() == False and node.up.is_root(): # Then it's a pillar
                    pillar_value = 0
                    for crit in node.get_children():
                        pillar_value = pillar_value + computed_value_for_criteria_dict[crit.name]*self.complete_dictionnary[crit.name]
                    computed_value_for_pillars_dict[node.name] = pillar_value
                    final_score = final_score + pillar_value*self.complete_dictionnary[node.name]



        # Puts everything in the graph
        pilar_dictionnary = computed_value_for_pillars_dict
        criteria_dictionnary = computed_value_for_criteria_dict
        indicator_dictionnary = computed_value_for_indicator_dict
        final_value = final_score
        complete_dictionnary = self.complete_dictionnary
        t = self.t

        self.window=QtWidgets.QMainWindow()
        self.ui=Ui_Dialog_for_graph()      #------------->creating an object
        self.ui.setupUi_for_graph(self.window,pilar_dictionnary,criteria_dictionnary,indicator_dictionnary,final_value,t,complete_dictionnary)
        self.window.show()
        # self.ui=Ui_Dialog_for_table()      #------------->creating an object
        # self.ui.setupUi_for_table(self.window,pilar_dictionnary,criteria_dictionnary,indicator_dictionnary,final_value)
        # self.window.show()


if __name__ == "__main__":
    import sys

    complete_dictionnary = {'Construction Cost': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Indirect Cost': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Rehabilitation Cost': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Dismantling Cost': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Production & Assembly': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Co2-eq Emissions': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Energy Consumption': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Index of Efficiency': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Index of risks': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Social Benefits': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Disturbances': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Cost': 0.61, 
                  'Time': 0.39, 
                  'Economic': 0.36,
                  'Emissions': 0.55, 
                  'Energy': 0.19, 
                  'Materials': 0.26, 
                  'Environmental': 0.39, 
                  'Safety': 0.6, 
                  '3rd Party affect': 0.4, 
                  'Social': 0.25, 
                  '': 0} 

    indicator_dictionnary = {'Construction Cost': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Indirect Cost': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Rehabilitation Cost': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Dismantling Cost': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Production & Assembly': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Co2-eq Emissions': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Energy Consumption': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Index of Efficiency': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Index of risks': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Social Benefits': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False},
                  'Disturbances': {'x_min': 1, 'x_max': 10, 'geometric_P': 1, 'geometric_K': 0, 'geometric_C': 1, 'binary': False, 'descending': False}} 

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_ValuesWindow()
    ui.setupUi(MainWindow, complete_dictionnary, indicator_dictionnary)
    MainWindow.show()
    sys.exit(app.exec_())
