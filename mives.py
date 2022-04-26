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


class Ui_MainWindow(QMainWindow):
    def setupUi(self, MainWindow):



        #initialize_tree
        self.image_width = 800
        self.weights = {}
        self.name_faces = {}
        self.weight_faces = {}
        self.t, self.ts , self.style, self.style1, self.style2 = self.get_example_tree()
        self.t.render("node_style.png", w=self.image_width, tree_style=self.ts)
        
        # For the indicator function
        self.indicator_dictionnary = {}

        

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1015, 776)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 0, self.image_width, 731))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("node_style.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")

        self.buttony = 0
        self.children_buttons = []
        

        rem_button = QtWidgets.QPushButton(self.centralwidget)
        rem_button.setGeometry(QtCore.QRect(810, self.buttony, 200, 30))
        self.buttony = self.buttony + 30
        rem_button.setObjectName("Remove")
        rem_button.setText("Remove leaves")
        rem_button.clicked.connect(self.remove_popup)


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


        check_weights_button = QtWidgets.QPushButton(self.centralwidget)
        check_weights_button.setGeometry(QtCore.QRect(810, self.buttony, 200, 30))
        self.buttony = self.buttony + 30
        check_weights_button.setObjectName("Check Weights")
        check_weights_button.setText("Check weights")
        check_weights_button.clicked.connect(self.check_weight_popup)


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


    def check_weight_popup(self):
        if(self.check_weights(self.t)):
            QMessageBox.about(self, "Weights", "Weights sum up to 1")
        else:
            QMessageBox.about(self, "Weights", "Weights don't sum up to 1")


    def weight_selection_popup_crit(self):
        list = ["Economic","Environmental","Social"]
        item,ok = QInputDialog.getItem(self,"Add Criterion","To which pillar do you want to add the new criterion?",list,0,False)
        if item == "Economic":
            pil = 0
        else:
            if item == "Environmental":
                pil = 1
            else:
                pil = 2

        Dialog = QtWidgets.QDialog()
        ui = Ui_Dialog()
        ui.setupUi(Dialog)
        Dialog.show()
        rsp = Dialog.exec_()
        if rsp == QtWidgets.QDialog.Accepted:
            if(self.check_user_input(ui.weight.toPlainText())):
                self.weights[ui.branch_name.toPlainText()] = ui.weight.toPlainText()
                self.add_child_crit(ui.branch_name.toPlainText(), ui.weight.toPlainText(), pil)
            else:
                QMessageBox.about(self, "Error", "Weight must be a number between 0 and 1")
        else:
            pass

    
    def check_user_input(self, input):
        try:
            # Convert it into integer
            val = float(input)
            if(val < 0 or val > 1):
                return False
            else:
                return True
        except ValueError:
            return False

    def weight_selection_popup_ind(self):
        list = []
        print(self.t)
        for node in self.t.traverse("postorder"):
            if  node.up != None and node.name != "Economic" and node.name != "Environmental" and node.name != "Social":
                parent = node.up
                if parent.name == "Economic" or parent.name == "Environmental" or parent.name == "Social":
                    list.append(node.name)
        print(list)
        crit,ok = QInputDialog.getItem(self,"Add Indicator","To which criterion do you want to add the new indicator?",list,0,False)

        Dialog = QtWidgets.QDialog()
        ui = Ui_Dialog()
        ui.setupUi(Dialog)
        Dialog.show()
        rsp = Dialog.exec_()
        if rsp == QtWidgets.QDialog.Accepted:
            if(self.check_user_input(ui.weight.toPlainText())):
                self.weights[ui.branch_name.toPlainText()] = ui.weight.toPlainText()
                self.add_child_ind(ui.branch_name.toPlainText(), ui.weight.toPlainText(), crit)
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
                    print(x_min,x_max,geometric_P,geometric_K, geometric_C, binary,descending)
                    name_of_indicator = ui.branch_name.toPlainText()
                    # Put all the values in a dictionnary in which the names of the indicator will be the key
                    # Hence we need to be careful not to have 2 indicators with the same name
                    self.indicator_dictionnary[name_of_indicator] = {"x_min":x_min,"x_max":x_max,"geometric_P":geometric_P,
                    "geometric_K":geometric_K,"binary":binary,"descending":descending}
                else:
                    pass
            else:
                QMessageBox.about(self, "Error", "Weight must be a number between 0 and 1")
        else:
            pass


    def add_child_crit(self, branch_name, weigth, pil):
        cost = self.t.children[pil]
        new_node = cost.add_child(name = branch_name)
        temp_button = QtWidgets.QPushButton(self.centralwidget)
        new_node.add_face(TextFace(new_node.name), column=0, position="branch-top")
        new_node.add_face(TextFace(weigth), column=0, position='branch-bottom')
        new_node.img_style = self.style1
        self.update_tree_display()

    
    def add_child_ind(self, branch_name, weigth, crit):
        for node in self.t.traverse("postorder"):
            if node.name == crit:
                cost = node
        new_node = cost.add_child(name = branch_name)
        temp_button = QtWidgets.QPushButton(self.centralwidget)
        new_node.add_face(TextFace(new_node.name), column=0, position="branch-top")
        new_node.add_face(TextFace(weigth), column=0, position='branch-bottom')
        new_node.img_style = self.style2
        self.update_tree_display()


    def remove_popup(self):
            list = []
            for leaf in self.t:
                if leaf.name == "Eco" or leaf.name == "Env" or leaf.name == "Soc":
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
                        # edit the node
                        Dialog = QtWidgets.QDialog()
                        ui = Ui_Dialog()
                        ui.setupUi(Dialog)
                        Dialog.show()
                        rsp = Dialog.exec_()
                        if rsp == QtWidgets.QDialog.Accepted:
                            if(self.check_user_input(ui.weight.toPlainText())):
                                node.write

                                new_name = ui.branch_name.toPlainText()
                                self.weights[new_name] = ui.weight.toPlainText()
                                self.name_faces[new_name] = self.name_faces[item] 
                                self.name_faces[new_name].text = new_name

                                self.weight_faces[new_name] = self.weight_faces[item]
                                self.weight_faces[new_name].text = self.weights[new_name]
                                node.name = new_name
                                self.update_tree_display()

                                del self.weights[item]
                                del self.name_faces[item]
                                del self.weight_faces[item]
                            else:
                                QMessageBox.about(self, "Error", "Weight must be a number between 0 and 1")
                        else:
                            pass
            else:
                pass
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))


    def check_weights(self, tree):
        if tree.get_children():
            summation = 0.0
            for child in tree.get_children():
                summation = summation + float(self.weights[child.name])
            if summation == 1:
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





        for node in t.traverse("postorder"):
            # Add text on top of tree nodes
            name_face = TextFace(node.name)
            weight_face = TextFace(str(self.weights[node.name]))
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
