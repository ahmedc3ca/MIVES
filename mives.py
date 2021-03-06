# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ete3 import Tree, faces, AttrFace, TreeStyle, NodeStyle, TextFace
from dialog import Ui_Dialog
from Indicator_updated import *
from values import Ui_ValuesWindow 
import copy
import math


class Ui_MainWindow(QMainWindow):
    def setupUi(self, MainWindow):

        # For the indicator function
        self.indicator_dictionnary = {}

        #initialize_tree
        self.image_width = 800
        self.weights = {}
        self.name_faces = {}
        self.weight_faces = {}
        self.t, self.ts , self.style, self.style1, self.style2 = self.get_example_tree()
        self.t.render("node_style.png", w=self.image_width, tree_style=self.ts)
        

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1015, 776)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")


        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, self.image_width, 731))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("node_style.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")


        self.buttony = 0
        self.children_buttons = []
        

        clear_button = QtWidgets.QPushButton(self.centralwidget)
        clear_button.setGeometry(QtCore.QRect(810, self.buttony, 200, 30))
        self.buttony = self.buttony + 30
        clear_button.setObjectName("Clear")
        clear_button.setText("Clear tree")
        clear_button.clicked.connect(self.clear_popup)


        rem_button = QtWidgets.QPushButton(self.centralwidget)
        rem_button.setGeometry(QtCore.QRect(810, self.buttony, 200, 30))
        self.buttony = self.buttony + 30
        rem_button.setObjectName("Remove")
        rem_button.setText("Remove leaves")
        rem_button.clicked.connect(self.remove_popup)


        pil_button = QtWidgets.QPushButton(self.centralwidget)
        pil_button.setGeometry(QtCore.QRect(810, self.buttony, 200, 30))
        self.buttony = self.buttony + 30
        pil_button.setObjectName("Add pillar")
        pil_button.setText("Add pillar")
        pil_button.clicked.connect(self.weight_selection_popup_pil)


        crit_button = QtWidgets.QPushButton(self.centralwidget)
        crit_button.setGeometry(QtCore.QRect(810, self.buttony, 200, 30))
        self.buttony = self.buttony + 30
        crit_button.setObjectName("Add criterion")
        crit_button.setText("Add criterion")
        crit_button.clicked.connect(self.weight_selection_popup_crit)


        ind_button = QtWidgets.QPushButton(self.centralwidget)
        ind_button.setGeometry(QtCore.QRect(810, self.buttony, 200, 30))
        self.buttony = self.buttony + 30
        ind_button.setObjectName("Add indicator")
        ind_button.setText("Add indicator")
        ind_button.clicked.connect(self.weight_selection_popup_ind)


        edit_button = QtWidgets.QPushButton(self.centralwidget)
        edit_button.setGeometry(QtCore.QRect(810, self.buttony, 200, 30))
        self.buttony = self.buttony + 30
        edit_button.setObjectName("Edit branch")
        edit_button.setText("Edit branch")
        edit_button.clicked.connect(self.edit_popup)


        value_window_button = QtWidgets.QPushButton(self.centralwidget)
        value_window_button.setGeometry(QtCore.QRect(810, 500, 200, 30))
        value_window_button.setObjectName("Next")
        value_window_button.setText("Next")
        value_window_button.clicked.connect(self.create_dictionnary)
        value_window_button.clicked.connect(self.open_values_window)


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1015, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)


        self.numClicked = 0
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def weight_selection_popup_crit(self):
        list = []
        for node in self.t.traverse("postorder"):
            if  node.up != None and node.up.up == None:
                    list.append(node.name)
        item,ok = QInputDialog.getItem(self,"Add Criterion","To which pillar do you want to add the new criterion?",list,0,False)
        
        if ok:
            Dialog = QtWidgets.QDialog()
            ui = Ui_Dialog()
            ui.setupUi(Dialog)
            Dialog.show()
            rsp = Dialog.exec_()
            if rsp == QtWidgets.QDialog.Accepted:
                if(self.check_user_input(ui.branch_name.text(),ui.weight.text())):
                    self.weights[ui.branch_name.text()] = ui.weight.text()
                    self.add_child_crit(ui.branch_name.text(), ui.weight.text(), item)
                else:
                    QMessageBox.about(self, "Error", "Can't have two branches with the same name and weights must be a number between 0 and 1")
            else:
                pass
        else:
            pass

    
    def weight_selection_popup_pil(self):
            Dialog = QtWidgets.QDialog()
            ui = Ui_Dialog()
            ui.setupUi(Dialog)
            Dialog.show()
            rsp = Dialog.exec_()
            if rsp == QtWidgets.QDialog.Accepted:
                if(self.check_user_input(ui.branch_name.text(),ui.weight.text())):
                    self.weights[ui.branch_name.text()] = ui.weight.text()
                    self.add_child_pil(ui.branch_name.text(), ui.weight.text())
                else:
                    QMessageBox.about(self, "Error", "Can't have two branches with the same name and weights must be a number between 0 and 1")
            else:
                pass

    
    def check_user_input(self,nameinput, weightinput):
        try:
            # Convert it into integer
            val = float(weightinput)
            if(val < 0 or val > 1):
                return False
            else:
                for node in self.t.traverse("postorder"):
                    if(node.name == nameinput):
                        return False
                return True
        except ValueError:
            return False

    def weight_selection_popup_ind(self):
        list = []
        for node in self.t.traverse("postorder"):
            if  node.up != None and node.up.up != None and node.up.up.up == None:
                list.append(node.name)
        crit,ok = QInputDialog.getItem(self,"Add Indicator","To which criterion do you want to add the new indicator?",list,0,False)

        if ok:
            Dialog = QtWidgets.QDialog()
            ui = Ui_Dialog()
            ui.setupUi(Dialog)
            Dialog.show()
            rsp = Dialog.exec_()
            if rsp == QtWidgets.QDialog.Accepted:
                if(self.check_user_input(ui.branch_name.text(),ui.weight.text())):
                    self.weights[ui.branch_name.text()] = ui.weight.text()
                    self.add_child_ind(ui.branch_name.text(), ui.weight.text(), crit)
                    Dialog_2 = QtWidgets.QDialog()
                    indicator_updated_dialog = indicator_updated()
                    indicator_updated_dialog.setupUi(Dialog_2)
                    Dialog_2.show()
                    rsp_2 = Dialog_2.exec_()
                    # Get the values from this indicator_updated_dialog function
                    if rsp_2 == QtWidgets.QDialog.Accepted:
                        # It's the indicator_updated_dialog that has all the values and variables
                        # We need to store x_min, x_max, the geometrical parameters and the binary and descending boxes values
                        # So that we can compute the function later. So we have to store one set of variables for each indicator.
                        # Maybe in a dictionnary ?
                        x_min = indicator_updated_dialog.min_value_input.text()
                        x_max = indicator_updated_dialog.max_value_input.text()
                        geometric_P = indicator_updated_dialog.geometrical_P_input.text()
                        geometric_C = indicator_updated_dialog.geometrical_C_input.text()
                        geometric_K = indicator_updated_dialog.geometrical_K_input.text()
                        binary = indicator_updated_dialog.binary_checkbox.isChecked()
                        descending = indicator_updated_dialog.descending_checkbox.isChecked()
                        name_of_indicator = ui.branch_name.text()
                        unit = indicator_updated_dialog.units_input.text()
                        # Put all the values in a dictionnary in which the names of the indicator will be the key
                        # Hence we need to be careful not to have 2 indicators with the same name
                        self.indicator_dictionnary[name_of_indicator] = {"weight":self.weights[name_of_indicator],"x_min":x_min,"x_max":x_max,"geometric_P":geometric_P,
                        "geometric_K":geometric_K,"geometric_C":geometric_C,"binary":binary,"descending":descending, "unit": unit}
                    else:
                        pass
                else:
                    QMessageBox.about(self, "Error", "Can't have two branches with the same name and weights must be a number between 0 and 1")
            else:
                pass
        else:
            pass
    
    
    def add_child_pil(self, branch_name, weight):
        for node in self.t.traverse("postorder"):
            if node.up == None:
                cost = node
        new_node = cost.add_child(name = branch_name)
        temp_button = QtWidgets.QPushButton(self.centralwidget)
        self.name_faces[new_node.name] = TextFace(new_node.name)
        self.name_faces[new_node.name].margin_left = 2
        self.name_faces[new_node.name].margin_right = 120-8*len(new_node.name)
        new_node.add_face(self.name_faces[new_node.name], column=0, position="branch-top")
        self.weight_faces[new_node.name] = TextFace(weight)
        self.weight_faces[new_node.name].margin_left = 2
        self.weight_faces[new_node.name].margin_right = 120-8*len(str(weight))
        new_node.add_face(self.weight_faces[new_node.name], column=0, position='branch-bottom')
        new_node.img_style = self.style1
        self.update_tree_display()


    def add_child_crit(self, branch_name, weight, pil):
        for node in self.t.traverse("postorder"):
            if node.name == pil:
                cost = node
        new_node = cost.add_child(name = branch_name)
        temp_button = QtWidgets.QPushButton(self.centralwidget)
        self.name_faces[new_node.name] = TextFace(new_node.name)
        self.name_faces[new_node.name].margin_left = 2
        self.name_faces[new_node.name].margin_right = 120-8*len(new_node.name)
        new_node.add_face(self.name_faces[new_node.name], column=0, position="branch-top")
        self.weight_faces[new_node.name] = TextFace(weight)
        self.weight_faces[new_node.name].margin_left = 2
        self.weight_faces[new_node.name].margin_right = 120-8*len(str(weight))
        new_node.add_face(self.weight_faces[new_node.name], column=0, position='branch-bottom')
        new_node.img_style = self.style1
        self.update_tree_display()

    
    def add_child_ind(self, branch_name, weight, crit):
        for node in self.t.traverse("postorder"):
            if node.name == crit:
                cost = node
        new_node = cost.add_child(name = branch_name)
        temp_button = QtWidgets.QPushButton(self.centralwidget)
        self.name_faces[new_node.name] = TextFace(new_node.name)
        self.name_faces[new_node.name].margin_left = 2
        self.name_faces[new_node.name].margin_right = 120-8*len(new_node.name)
        new_node.add_face(self.name_faces[new_node.name], column=0, position="branch-top")
        self.weight_faces[new_node.name] = TextFace(weight)
        self.weight_faces[new_node.name].margin_left = 2
        self.weight_faces[new_node.name].margin_right = 120-8*len(str(weight))
        new_node.add_face(self.weight_faces[new_node.name], column=0, position='branch-bottom')
        new_node.img_style = self.style2
        self.update_tree_display()


    def remove_popup(self):
            list = []
            for leaf in self.t:
                if leaf.up == None:
                    continue
                list.append(leaf.name)

            item,ok = QInputDialog.getItem(self,"Remove Branch","Select the branch you want to remove",list,0,False)

            if ok:
                for leaf in self.t:
                    if leaf.name == item:
                        leaf.detach()
                        self.update_tree_display()
            else :
                pass


    def edit_popup(self):
            list = []
            for node in self.t.traverse("levelorder"):
                if  node.up != None:
                    list.append(node.name)

            item,ok = QInputDialog.getItem(self,"Edit Branch","Select the branch you want to edit",list,0,False)

            if ok:
                for node in self.t.traverse("levelorder"):
                    if node.name == item:
                        Dialog = QtWidgets.QDialog()
                        ui = Ui_Dialog()
                        ui.setupUi(Dialog)
                        ui.branch_name.setText(node.name)
                        ui.weight.setText(str(self.weights[node.name]))
                        Dialog.show()
                        rsp = Dialog.exec_()
                        if rsp == QtWidgets.QDialog.Accepted:
                            if(node.name == ui.branch_name.text()):
                                input_check = self.check_user_input("$testNameThatNooneWouldWrite$__", ui.weight.text())
                            else:
                                input_check = self.check_user_input(ui.branch_name.text(), ui.weight.text())
                            if(input_check):
                                node.write

                                del self.weights[item]

                                new_name = ui.branch_name.text()
                                self.weights[ui.branch_name.text()] = ui.weight.text()
                                self.name_faces[new_name] = self.name_faces[item] 
                                self.name_faces[new_name].text = new_name

                                self.weight_faces[new_name] = self.weight_faces[item]
                                self.weight_faces[new_name].text = self.weights[new_name]
                                node.name = new_name

                                if(item != new_name):
                                    del self.name_faces[item]
                                    del self.weight_faces[item]

                                self.update_tree_display()

                                if item in self.indicator_dictionnary:
                                
                                    del self.indicator_dictionnary[item]
                                    Dialog_2 = QtWidgets.QDialog()
                                    indicator_updated_dialog = indicator_updated()
                                    indicator_updated_dialog.setupUi(Dialog_2)
                                    Dialog_2.show()
                                    rsp_2 = Dialog_2.exec_()
                                    # Get the values from this indicator_updated_dialog function
                                    if rsp_2 == QtWidgets.QDialog.Accepted:
                                        # It's the indicator_updated_dialog that has all the values and variables
                                        # We need to store x_min, x_max, the geometrical parameters and the binary and descending boxes values
                                        # So that we can compute the function later. So we have to store one set of variables for each indicator.
                                        # Maybe in a dictionnary ?
                                        x_min = indicator_updated_dialog.min_value_input.text()
                                        x_max = indicator_updated_dialog.max_value_input.text()
                                        geometric_P = indicator_updated_dialog.geometrical_P_input.text()
                                        geometric_C = indicator_updated_dialog.geometrical_C_input.text()
                                        geometric_K = indicator_updated_dialog.geometrical_K_input.text()
                                        binary = indicator_updated_dialog.binary_checkbox.isChecked()
                                        descending = indicator_updated_dialog.descending_checkbox.isChecked()
                                        name_of_indicator = ui.branch_name.text()
                                        unit = indicator_updated_dialog.units_input.text()
                                        # Put all the values in a dictionnary in which the names of the indicator will be the key
                                        # Hence we need to be careful not to have 2 indicators with the same name
                                        self.indicator_dictionnary[name_of_indicator] = {"weight":self.weights[name_of_indicator],"x_min":x_min,"x_max":x_max,"geometric_P":geometric_P,
                                        "geometric_K":geometric_K,"geometric_C":geometric_C,"binary":binary,"descending":descending, "unit": unit}
                                    else:
                                        pass

                            else:
                                QMessageBox.about(self, "Error", "Can't have two branches with the same name and weights must be a number between 0 and 1")
                            
                        else:
                            pass
            else:
                pass
    

    def clear_popup(self):
            for node in self.t.traverse("postorder"):
                if node!=None:
                    node.detach()
                    self.update_tree_display()


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MIVES"))
        
        
    def create_dictionnary(self):
        self.complete_dictionnary = copy.copy(self.indicator_dictionnary)
        for node in self.t.traverse("postorder"):
            if  node.is_leaf()== False and node.up != None:
                self.complete_dictionnary[node.name] = self.weights[node.name]


    def open_values_window(self):
        crit = []
        criteria_with_no_ind = False
        for node in self.t.traverse("postorder"):
            if  node.up != None and node.up.up != None and node.up.up.up != None:
                continue
            else:
                if node.is_leaf():
                    criteria_with_no_ind = True
        
        if(self.check_weights(self.t)) and criteria_with_no_ind==False:
            #Open second window
            self.window=QtWidgets.QMainWindow()
            self.ui=Ui_ValuesWindow()      #------------->creating an object 
            self.ui.setupUi(self.window, self.complete_dictionnary, self.indicator_dictionnary,self.t)
            self.window.show()
            self.close()

        else:
            if self.check_weights(self.t)==False:
                QMessageBox.about(self, "Weights", "Weights don't sum up to 1")
            if criteria_with_no_ind:
                QMessageBox.about(self, "Missing leaves", "There are branches with no indicators")


    def check_weights(self, tree):
        if tree.get_children():
            summation = 0.0
            for child in tree.get_children():
                summation = summation + float(self.weights[child.name])
            if math.isclose(summation, 1):
                check_children = True
                for child in tree.get_children():
                    check_children = check_children and self.check_weights(child)
                return check_children
            else:
                return False


        else:
            return True


    def update_tree_display(self):
        self.t.render("node_style.png", w=self.image_width, tree_style=self.ts)
        self.label.setPixmap(QtGui.QPixmap("node_style.png"))


    def get_example_tree(self):
        self.weights[''] = 0
        self.weights["Economic"] = 0.36
        self.weights["Social"] = 0.25
        self.weights["Environmental"] = 0.39
        self.weights["Cost"] = 0.61
        self.weights["Time"] = 0.39
        self.weights["Emissions"] = 0.55
        self.weights["Energy"] = 0.19
        self.weights["Materials"] = 0.26
        self.weights["Safety"] = 0.6
        self.weights["3rd Party affect"] = 0.4
        self.weights["Construction Cost"] = 0.58
        self.weights["Indirect Cost"] = 0.09
        self.weights["Rehabilitation Cost"] = 0.13
        self.weights["Dismantling Cost"] = 0.2
        self.weights["Production & Assembly"] = 1
        self.weights["Co2-eq Emissions"] = 1
        self.weights["Energy Consumption"] = 1
        self.weights["Index of Efficiency"] = 1
        self.weights["Index of risks"] = 1
        self.weights["Social Benefits"] = 0.5
        self.weights["Disturbances"] = 0.5

        string = "(((Construction Cost,Indirect Cost,Rehabilitation Cost,Dismantling Cost)Cost,(Production & Assembly)Time)Economic,((Co2-eq Emissions)Emissions,(Energy Consumption)Energy,(Index of Efficiency)Materials)Environmental,((Index of risks)Safety,(Social Benefits,Disturbances)3rd Party affect)Social);"
        t = Tree(string, format = 8)

        # Node style handling is no longer limited to layout functions. You
        # can now create fixed node styles and use them many times, save them
        # or even add them to nodes before drawing (this allows to save and
        # reproduce an tree image design)

        # Set bold red branch to the root node
        style = NodeStyle()
        style["fgcolor"] = "#0f0f0f"
        style["size"] = 0
        style["vt_line_color"] = "#ff0000"
        style["hz_line_color"] = "#ff0000"
        style["vt_line_width"] = 6
        style["hz_line_width"] = 6
        style["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
        style["hz_line_type"] = 0

        #Set less thicker red branch
        style1 = NodeStyle()
        style1["fgcolor"] = "#0f0f0f"
        style1["size"] = 0
        style1["vt_line_color"] = "#ff0000"
        style1["hz_line_color"] = "#ff0000"
        style1["vt_line_width"] = 4
        style1["hz_line_width"] = 4
        style1["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
        style1["hz_line_type"] = 0

        # Set dashed blue lines in all leaves
        style2 = NodeStyle()
        style2["fgcolor"] = "#0f0f0f"
        style2["size"] = 0
        style2["vt_line_color"] = "#ff0000"
        style2["hz_line_color"] = "#ff0000"
        style2["vt_line_width"] = 2
        style2["hz_line_width"] = 2
        style2["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
        style2["hz_line_type"] = 0

        t.set_style(style2)
        for node in t.traverse("postorder"):
            node.img_style = style1
        t.children[0].img_style = style
        t.children[1].img_style = style
        t.children[2].img_style = style

        for l in t.iter_leaves():
            l.img_style = style2
        
        # We input a geometry for the predefined indicators
        for node in t.traverse("postorder"):
            if  node.is_leaf()== True:
                self.indicator_dictionnary[node.name] = {"weight":self.weights[node.name],"x_min":1,"x_max":10,"geometric_P":1,
                    "geometric_K":0,"geometric_C":1,"binary":False,"descending":False, "unit": "-"}

        for node in t.traverse("postorder"):
            # Add text on top of tree nodes
            name_face = TextFace(node.name)
            name_face.margin_left = 2
            name_face.margin_right = 120-8*len(node.name)
            weight_face = TextFace(str(self.weights[node.name]))
            weight_face.margin_left = 2
            weight_face.margin_right = 120-8*len(str(self.weights[node.name]))
            self.name_faces[node.name] = name_face
            self.weight_faces[node.name] = weight_face
            node.add_face(name_face, column=0, position="branch-top")
            node.add_face(weight_face, column=0, position="branch-bottom")

        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.force_topology = True
        return t, ts, style, style1, style2

class inputdialogdemo(QWidget):
   def __init__(self, parent = None):
      super(inputdialogdemo, self).__init__(parent)

      layout = QFormLayout()
      self.btn = QPushButton("Choose from list")
      self.btn.clicked.connect(self.getItem)

      self.le = QLineEdit()
      layout.addRow(self.btn,self.le)
      self.btn1 = QPushButton("get name")
      self.btn1.clicked.connect(self.gettext)

      self.le1 = QLineEdit()
      layout.addRow(self.btn1,self.le1)
      self.btn2 = QPushButton("Enter an integer")
      self.btn2.clicked.connect(self.getint)

      self.le2 = QLineEdit()
      layout.addRow(self.btn2,self.le2)
      self.setLayout(layout)
      self.setWindowTitle("Input Dialog demo")

   def getItem(self):
      items = ("C", "C++", "Java", "Python")

      item, ok = QInputDialog.getItem(
         self, "select input dialog", "list of languages", items, 0, False
      )

      if ok and item:
         self.le.setText(item)

   def gettext(self):
      text, ok = QInputDialog.getText(self, 'Text Input Dialog', 'Enter your name:')

      if ok:
         self.le1.setText(str(text))

   def getint(self):
      num,ok = QInputDialog.getInt(self,"integer input dualog","enter a number")

      if ok:
         self.le2.setText(str(num))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
