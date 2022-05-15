# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\33621\Documents\Romain\Cours_EPFL\Semestre_3\How_people_learn_2\Code\Graphic_output.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from typing import final
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import numpy as np
import sys

class Table(QWidget):

    def __init__(self,complete_dict,input_dict,pilar_dictionnary,criteria_dictionnary,indicator_dictionnary,final_value):
        super().__init__()
        self.left = 0
        self.top = 0
        self.width = 500
        self.height = 500
        self.initUI(complete_dict,input_dict,pilar_dictionnary,criteria_dictionnary,indicator_dictionnary,final_value)
        
    def initUI(self,complete_dict,input_dict,pilar_dictionnary,criteria_dictionnary,indicator_dictionnary,final_value):

        # Add box layout, add table to box layout and add box layout to widget
        self.title = 'Summary table'
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.createTable(complete_dict,input_dict,pilar_dictionnary,criteria_dictionnary,indicator_dictionnary,final_value)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.show()

        # Show widget
        # self.show()

    def createTable(self,complete_dict,input_dict,pilar_dictionnary,criteria_dictionnary,indicator_dictionnary,final_value):
        # Create table

        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(complete_dict)+1)
        self.tableWidget.setColumnCount(4*len(final_value))
        horHeaders = ["Name "+str(1),"Weight "+str(1),"Input value "+str(1),"Computed value "+str(1),"Name "+str(2),"Weight "+str(2),"Input value "+str(2),"Computed value "+str(2),"Name "+str(3),"Weight "+str(3),"Input value "+str(3),"Computed value "+str(3)]
        for n in range(len(final_value)):
            self.tableWidget.setHorizontalHeaderLabels(horHeaders)
            i = 0
            for ind in indicator_dictionnary:
                self.tableWidget.setItem(i,4*n,QTableWidgetItem(str(ind)))
                self.tableWidget.setItem(i,4*n+1,QTableWidgetItem(str(complete_dict[ind]["weight"])))
                self.tableWidget.setItem(i,4*n+2,QTableWidgetItem(str(input_dict[ind][n])))
                self.tableWidget.setItem(i,4*n+3,QTableWidgetItem(str(round(indicator_dictionnary[ind][n],3))))
                i = i+1
            for crit in criteria_dictionnary:
                self.tableWidget.setItem(i,4*n,QTableWidgetItem(str(crit)))
                self.tableWidget.setItem(i,4*n+1,QTableWidgetItem(str(complete_dict[crit])))
                self.tableWidget.setItem(i,4*n+3,QTableWidgetItem(str(round(criteria_dictionnary[crit][n],3))))
                i = i+1
            for pil in pilar_dictionnary:
                self.tableWidget.setItem(i,4*n,QTableWidgetItem(str(pil)))
                self.tableWidget.setItem(i,4*n+1,QTableWidgetItem(str(complete_dict[pil])))
                self.tableWidget.setItem(i,4*n+3,QTableWidgetItem(str(round(pilar_dictionnary[pil][n],3))))
                i = i+1
            self.tableWidget.setItem(i,4*n,QTableWidgetItem("Final score"))
            self.tableWidget.setItem(i,4*n+3,QTableWidgetItem(str(round(final_value[n],3))))


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = App(input_dict,pilar_dictionnary,criteria_dictionnary,indicator_dictionnary,final_value)
#     sys.exit(app.exec_())  
