# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\33621\Documents\Romain\Cours_EPFL\Semestre_3\How_people_learn_2\Code\Graphic_output.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
import matplotlib.pyplot as plt


class Ui_Dialog_for_graph(object):
    def setupUi_for_graph(self, Dialog,pilar_dictionnary,criteria_dictionnary,indicator_dictionnary,final_value):
        self.final_value = final_value
        self.pilar_dictionnary = pilar_dictionnary
        self.criteria_dictionnary = criteria_dictionnary
        self.indicator_dictionnary = indicator_dictionnary
        Dialog.setObjectName("Dialog")
        Dialog.resize(945, 806)
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(30, 30, 891, 721))
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 0, 731, 21))
        self.label_text ="PILLARS VIEW - final sustainability score: "+str(final_value)
        self.label.setObjectName(self.label_text)
        self.number_of_times_clicked = 0 # to know which graph to be in
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(720, 760, 221, 41))
        self.pushButton.setObjectName("Criteria")
        self.pushButton.clicked.connect(self.button_clicked)


        #Plot the initial figure: pillars for one output
        X = []
        Y = []
        for key in pilar_dictionnary:
            X.append(key)
            Y.append(pilar_dictionnary[key])
        plt.bar(X,Y,width=0.1)
        plt.savefig("Pillars_plot.png",bbox_inches = "tight")
        self.plot_figure("Pillars_plot.png")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog",self.label_text))
        self.pushButton.setText(_translate("Dialog", "PushButton"))


    def plot_figure(self,image):
        plt.clf()
        if self.number_of_times_clicked>=1: # We need to clear the graph
            pass
        self.grview = QtWidgets.QGraphicsView(self.graphicsView)
        scene = QtWidgets.QGraphicsScene(self.graphicsView)
        image = QPixmap(image)

        # Set it at the scale of the image to fit the window
        width = self.graphicsView.width()
        height = self.graphicsView.height()
        resized_image = image.scaled(width,height,QtCore.Qt.KeepAspectRatio) # To scale automaticall
        resized_image_1 = image.scaledToWidth(width) # To scale to the witdth of the window
        resized_image_2 = image.scaledToHeight(height) #To scale to the height of the window

        scene.addPixmap(resized_image)
        self.grview.setScene(scene)
        self.grview.show()

    def button_clicked(self):
        self.number_of_times_clicked = self.number_of_times_clicked+1
        if (self.number_of_times_clicked%2) != 0: # Pair number, we should be in the criteria stuff
                    self.label_text ="CRITERIA VIEW - final sustainability score: "+str(self.final_value)
                    self.label.setObjectName(self.label_text)
                    X = []
                    Y = []
                    for key in self.criteria_dictionnary:
                        X.append(key)
                        Y.append(self.criteria_dictionnary[key])
                    plt.bar(X,Y,width=0.1)
                    plt.savefig("Criteria_plot.png",bbox_inches = "tight")
                    self.plot_figure("Criteria_plot.png")
        else:
                    self.label_text ="PILLAR VIEW - final sustainability score: "+str(self.final_value)
                    self.label.setObjectName(self.label_text)
                    self.plot_figure("Pillars_plot.png")





if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog_for_graph()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

##

