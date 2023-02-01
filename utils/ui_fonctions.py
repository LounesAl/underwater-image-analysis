import sys
from PySide2 import QtCore, QtGui

from PySide2.QtWidgets import (QApplication, QMainWindow, QAction, QToolBar, QGridLayout,
                               QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QMenu, QToolButton, QFileDialog, QWidget, QMessageBox, QFileDialog)

    
def browse_folder(self, button_id):
    folder_path = QFileDialog.getExistingDirectory(self, 'Sélectionner un dossier', '', QFileDialog.ShowDirsOnly)
    if folder_path:
        if button_id == 1:
            self.path1 = folder_path
        elif button_id == 2:
            self.path2 = folder_path
            
def browse_file(self, button_id):
    file_path, _ = QFileDialog.getOpenFileName(self, 'Sélectionner un fichier', '', "Tous les fichiers (*)")
    if file_path:
        if button_id == 1:
            self.path1 = file_path
        elif button_id == 2:
            self.path2 = file_path

            
def nouveau_fichier(self):
    print("Call nouveau")

def ouvrir_fichier(self):
    print("Call ouviir")

def enregistrer_fichier(self):
    print("Call enregistrer")
    
def exit(self):
    QApplication.exit()

def demarrer(self):
    print("Call demarrer")

def a_propos(self):
    dlg = QMessageBox(self)
    dlg.setWindowTitle("A propos de l'application")
    dlg.setText("Cette application est créer par un groupe d'étudiants de \n                         Sorbonne Sniversité.\n                         Tous droits reservés")
    button = dlg.exec_()
    if button == QMessageBox.Ok:
        print("OK!")