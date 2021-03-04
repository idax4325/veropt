from veropt.gui.gui_setup import Ui_MainWindow
from PySide2.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QWidget
from PySide2.QtGui import QTextCursor
from PySide2.QtCore import QThread, QObject, Signal
from veropt import BayesOptimiser
import sys
from queue import Queue
import torch


class BayesOptWindow(QMainWindow):
    def __init__(self, optimiser: BayesOptimiser, opt_worker):
        super(BayesOptWindow, self).__init__()

        self.optimiser = optimiser
        self.opt_worker = opt_worker

        self.opt_worker.window = self

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.textEdit_main_out.setReadOnly(True)

        self.last_string_was_false = False
        self.inserted_newline_since_last_string = False
        self.last_string_was_r = False

        self.set_up_plot_pred_group()

        self.connect_signals_to_slots()

        self.ls_tabs = []
        self.ls_tab_widgets = []
        self.set_up_ls_tabs()

        self.cc_wrappers = []
        for obj_no in range(self.optimiser.n_objs):
            self.cc_wrappers.append(self.ChangeConstraintsWrapper(obj_no, self.change_constraints))
            self.ls_tab_widgets[obj_no]["pushButton_set_constraints"].clicked.connect(
                self.cc_wrappers[obj_no].run)
        self.update_status_labels()

        self.update_labels_and_buttons()

        # TODO: Remove once 3d plots have been updated and the gui supports choosing two variables
        self.ui.checkBox_plotp_2d.setDisabled(True)

    def connect_signals_to_slots(self):
        self.ui.pushButton_refit_kernel.clicked.connect(self.opt_worker.refit_model)
        self.ui.pushButton_run_opt_step.clicked.connect(self.opt_worker.do_opt_steps)
        self.ui.pushButton_suggest_point.clicked.connect(self.opt_worker.suggest_bayes_steps)

        self.opt_worker.working_signal.connect(self.freeze_buttons)
        self.opt_worker.finished_signal.connect(self.unfreeze_buttons)
        self.opt_worker.finished_signal.connect(self.update_labels_and_buttons)
        self.opt_worker.update_signal.connect(self.update_labels_and_buttons)

        self.ui.pushButton_plot_prediction.clicked.connect(self.plot_prediction)
        # self.ui.pushButton_plot_prediction_real_units.clicked.connect(self.plot_prediction_real_units)
        self.ui.pushButton_plot_progress.clicked.connect(self.optimiser.plot_progress)
        self.ui.pushButton_plot_variables.clicked.connect(self.optimiser.plot_variable_values)
        self.ui.pushButton_close_plots.clicked.connect(self.optimiser.close_plots)

        self.ui.pushButton_save_points.clicked.connect(self.optimiser.save_suggested_steps)
        self.ui.pushButton_add_points.clicked.connect(self.optimiser.load_new_data)

        self.ui.pushButton_set_beta.clicked.connect(self.change_beta_val)
        self.ui.pushButton_set_gamma.clicked.connect(self.change_gamma_val)

        self.ui.pushButton_save_optimiser.clicked.connect(self.optimiser.save_optimiser)

    def update_labels_and_buttons(self):
        self.update_labels()
        self.update_buttons()

    def update_labels(self):
        self.update_status_labels()
        self.update_length_scale_label()
        self.update_constraint_labels()
        self.update_beta_label()
        self.update_gamma_label()
        self.update_local_best_val()

    def update_buttons(self):

        if self.optimiser.obj_func.function is None:
            self.ui.checkBox_keep_running.setDisabled(True)
        else:
            self.ui.checkBox_keep_running.setDisabled(False)

        if self.optimiser.obj_func.saver is None:
            self.ui.pushButton_save_points.setDisabled(True)

        if self.optimiser.obj_func.loader is None:
            self.ui.pushButton_add_points.setDisabled(True)

        if self.optimiser.data_fitted:
            self.ui.pushButton_plot_prediction.setDisabled(False)
            # self.ui.pushButton_plot_prediction_real_units.setDisabled(False)
            self.ui.pushButton_suggest_point.setDisabled(False)
        else:
            self.ui.pushButton_plot_prediction.setDisabled(True)
            # self.ui.pushButton_plot_prediction_real_units.setDisabled(True)
            self.ui.pushButton_suggest_point.setDisabled(True)

    def set_up_plot_pred_group(self):

        if self.optimiser.n_objs > 1:
            self.ui.comboBox_plotp_obj.addItem("All")
        else:
            self.ui.comboBox_plotp_obj.setDisabled(True)

        for obj_no in range(self.optimiser.n_objs):
            obj_name = self.optimiser.obj_func.obj_names[obj_no] if self.optimiser.obj_func.obj_names is not None \
                else f"Obj {obj_no+1}"
            self.ui.comboBox_plotp_obj.addItem(obj_name)

        if self.optimiser.n_params > 1:
            self.ui.comboBox_plotp_var.addItem("All")
        else:
            self.ui.comboBox_plotp_var.setDisabled(True)

        for var_no in range(self.optimiser.n_params):
            var_name = self.optimiser.obj_func.var_names[var_no] if self.optimiser.obj_func.var_names is not None \
                else f"Var {var_no+1}"
            self.ui.comboBox_plotp_var.addItem(var_name)

    def set_up_ls_tabs(self):

        for obj_no in range(self.optimiser.n_objs):

            if self.optimiser.obj_func.obj_names is not None:
                obj_name = self.optimiser.obj_func.obj_names[obj_no]
            else:
                obj_name = f"Obj {obj_no}"

            if obj_no == 0:
                self.ls_tabs.append(self.ui.ls_tab_1)
                self.ui.length_scale_tabs.setTabText(obj_no, obj_name)

            else:
                self.ls_tabs.append(QWidget())
                self.ui.length_scale_tabs.addTab(self.ls_tabs[obj_no], obj_name)

            if obj_no == 0:
                self.ls_tab_widgets.append({
                    "label_length_scale": self.ui.label_length_scale,
                    "label_constraints": self.ui.label_constraints,
                    "lineEdit_constraint_0": self.ui.lineEdit_constraint_0,
                    "lineEdit_constraint_1": self.ui.lineEdit_constraint_1,
                    "pushButton_set_constraints": self.ui.pushButton_set_constraints,
                    "label_obj_best_val": self.ui.label_obj_best_val
                })
            else:
                label_length_scale = QLabel(parent=self.ls_tabs[obj_no])
                label_length_scale.setGeometry(self.ui.label_length_scale.geometry())
                label_length_scale.setText(self.ui.label_length_scale.text())

                label_constraints = QLabel(parent=self.ls_tabs[obj_no])
                label_constraints.setGeometry(self.ui.label_constraints.geometry())

                lineEdit_constraint_0 = QLineEdit(parent=self.ls_tabs[obj_no])
                lineEdit_constraint_0.setGeometry(self.ui.lineEdit_constraint_0.geometry())

                lineEdit_constraint_1 = QLineEdit(parent=self.ls_tabs[obj_no])
                lineEdit_constraint_1.setGeometry(self.ui.lineEdit_constraint_1.geometry())

                pushButton_set_constraints = QPushButton(parent=self.ls_tabs[obj_no])
                pushButton_set_constraints.setGeometry(self.ui.pushButton_set_constraints.geometry())
                pushButton_set_constraints.setText(self.ui.pushButton_set_constraints.text())

                label_obj_best_val = QLabel(parent=self.ls_tabs[obj_no])
                label_obj_best_val.setGeometry(self.ui.label_obj_best_val.geometry())
                label_obj_best_val.setText(self.ui.label_obj_best_val.text())

                self.ls_tab_widgets.append({
                    "label_length_scale": label_length_scale,
                    "label_constraints": label_constraints,
                    "lineEdit_constraint_0": lineEdit_constraint_0,
                    "lineEdit_constraint_1": lineEdit_constraint_1,
                    "pushButton_set_constraints": pushButton_set_constraints,
                    "label_obj_best_val": label_obj_best_val
                })

    def plot_prediction(self):

        obj_no_list = [int(self.ui.comboBox_plotp_obj.currentIndex()) - 1]

        var_no_list = [int(self.ui.comboBox_plotp_var.currentIndex()) - 1]

        if obj_no_list[0] == -1:
            obj_no_list = torch.arange(self.optimiser.n_objs)

        if var_no_list[0] == -1:
            var_no_list = torch.arange(self.optimiser.n_params)

        normalised = bool(self.ui.checkBox_plotp_norm.checkState())
        two_dims = bool(self.ui.checkBox_plotp_2d.checkState())

        if not two_dims:
            for obj_no in obj_no_list:
                for var_no in var_no_list:
                    self.optimiser.plot_prediction(int(obj_no), int(var_no), in_real_units=(not normalised))

        elif not normalised and two_dims:
            print("Not implemented! Plotting two variables at once is currently only supported with real units. Check"
                  " the \"Normalised\" box to make a 3D plot.")
            
        # else:
            # TODO: Update this plot function to support MO
            # for obj_no in obj_no_list:
            #     for var_no in var_no_list:
            #         self.optimiser.plot_prediction_2d_real_units(int(obj_no), int(var_no))

    def plot_prediction_real_units(self):
        for var_ind in range(self.optimiser.n_params):
            self.optimiser.plot_prediction_real_units(var_ind)

    def update_beta_label(self):
        if 'beta' in self.optimiser.acq_func.params:
            self.ui.label_beta.setText(f"Beta: {self.optimiser.acq_func.params['beta']}")
        else:
            self.ui.label_beta.setText(f"Beta not used ")
            self.ui.pushButton_set_beta.setDisabled(True)

        self.ui.label_beta.repaint()

    def update_gamma_label(self):
        if 'gamma' in self.optimiser.acq_func.params:
            self.ui.label_gamma.setText(f"Gamma: {self.optimiser.acq_func.params['gamma']}")
        else:
            self.ui.label_gamma.setText(f"Gamma not used ")
            self.ui.pushButton_set_gamma.setDisabled(True)

        self.ui.label_gamma.repaint()

    def change_beta_val(self):
        try:
            new_val = float(self.ui.lineEdit_beta.text())
            self.optimiser.set_acq_func_params('beta', new_val)
            self.update_beta_label()

        except ValueError:
            self.write_to_textfield("Invalid value encountered. The input must be a float.")
        self.ui.lineEdit_beta.setText('')
        self.ui.lineEdit_beta.repaint()

    def change_gamma_val(self):
        try:
            new_val = float(self.ui.lineEdit_gamma.text())
            self.optimiser.set_acq_func_params('gamma', new_val)
            self.update_gamma_label()
        except ValueError:
            self.write_to_textfield("Invalid value encountered. The input must be a float.")
        self.ui.lineEdit_gamma.setText('')
        self.ui.lineEdit_gamma.repaint()

    def update_length_scale_label(self):
        for obj_no in range(self.optimiser.n_objs):
            if 'Matern' in self.optimiser.model.model_class_list[obj_no].__name__ or 'RBF' in self.optimiser.model.model_class_list[obj_no].__name__:
                if self.optimiser.data_fitted:
                    if self.optimiser.multi_obj:
                        vals = self.optimiser.model.\
                            view_model_hyperparameter('covar_module', 'raw_lengthscale', obj_no).squeeze(0).tolist()
                    else:
                        vals = self.optimiser.model.\
                            view_model_hyperparameter('covar_module', 'raw_lengthscale', 0).squeeze(0).tolist()

                    string_vals = ""
                    for val_no, val in enumerate(vals):
                        string_vals += f"{val:.2f}"
                        if val_no < len(vals) - 1:
                            string_vals += ", "

                    self.ls_tab_widgets[obj_no]["label_length_scale"].setText(f"Lengthscale: [{string_vals}]")

                    self.ls_tab_widgets[obj_no]["label_length_scale"].repaint()

                    # self.ui.label_length_scale.setText(f"Lengthscale: [{string_vals}]")
                    #
                    # self.ui.label_length_scale.repaint()

    def update_local_best_val(self):
        if self.optimiser.points_evaluated > 0:
            for obj_no in range(self.optimiser.n_objs):
                self.ls_tab_widgets[obj_no]["label_obj_best_val"].setText(
                    f"Best value: {self.optimiser.best_val(max_for_single_obj_ind=obj_no):.2f}")

    def update_constraint_labels(self):
        for obj_no in range(self.optimiser.n_objs):
            if 'Matern' in self.optimiser.model.model_class_list[obj_no].__name__ or 'RBF' in self.optimiser.model.model_class_list[obj_no].__name__:
                constraint_vals = self.optimiser.model.constraint_dict_list[obj_no]["covar_module"]["raw_lengthscale"]
                self.ls_tab_widgets[obj_no]["label_constraints"].setText(
                    f"Constraints: [{constraint_vals[0]}, {constraint_vals[1]}]")

                self.ls_tab_widgets[obj_no]["label_constraints"].repaint()

                # self.ui.label_constraints.repaint()

    class ChangeConstraintsWrapper:
        def __init__(self, obj_no, change_constraints):
            self.obj_no = obj_no
            self.change_constraints = change_constraints

        def run(self):
            self.change_constraints(self.obj_no)

    def change_constraints(self, obj_no):
        try:
            new_val_0 = float(self.ls_tab_widgets[obj_no]["lineEdit_constraint_0"].text())
            new_val_1 = float(self.ls_tab_widgets[obj_no]["lineEdit_constraint_1"].text())
            self.optimiser.model.constraint_dict_list[obj_no]["covar_module"]["raw_lengthscale"] = [new_val_0, new_val_1]
            self.update_constraint_labels()
            self.write_to_textfield("Constraints changed! The model will now be refitted.\n")
            self.opt_worker.refit_model()
        except ValueError:
            self.write_to_textfield("Invalid value encountered. Write a float in both constraint fields.")
        self.ls_tab_widgets[obj_no]["lineEdit_constraint_0"].setText('')
        self.ls_tab_widgets[obj_no]["lineEdit_constraint_1"].setText('')

        self.ls_tab_widgets[obj_no]["lineEdit_constraint_0"].repaint()
        self.ls_tab_widgets[obj_no]["lineEdit_constraint_1"].repaint()

    def update_status_labels(self):
        self.ui.label_current_point.setText(f"Step {self.optimiser.current_step} of {self.optimiser.n_steps} "
                                            f"({self.optimiser.points_evaluated} of {self.optimiser.n_points} "
                                            f"points evaluated)")
        if self.optimiser.points_evaluated > 0:
            if self.optimiser.multi_obj:
                self.ui.label_best_val.setText(f"Best summed value: {self.optimiser.best_val(weighted_best=True)[0]:.2f}")
            else:
                self.ui.label_best_val.setText(f"Best value: {float(self.optimiser.best_val()):.2f}")

        if self.optimiser.opt_mode == "init":
            self.ui.label_opt_mode.setText("Optimisation mode: Initial")
        else:
            self.ui.label_opt_mode.setText("Optimisation mode: Bayes")

    def write_to_textfield(self, string):
        if self.last_string_was_r:

            string = string.replace("\r", "").replace("\n", "")

            if string:

                self.ui.textEdit_main_out.moveCursor(QTextCursor.End)
                self.ui.textEdit_main_out.moveCursor(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
                self.ui.textEdit_main_out.moveCursor(QTextCursor.Up, QTextCursor.KeepAnchor)

                self.ui.textEdit_main_out.insertPlainText(string + "\n")

                self.last_string_was_r = False
            else:
                self.ui.textEdit_main_out.insertPlainText("\n")

        elif "\r" in string:

            self.last_string_was_r = True

        else:

            self.ui.textEdit_main_out.moveCursor(QTextCursor.End)

            string = string.replace("\n", "")

            if string:
                self.ui.textEdit_main_out.insertPlainText(string + "\n")
                self.inserted_newline_since_last_string = False
                self.last_string_was_false = False
            else:
                if self.last_string_was_false and (self.inserted_newline_since_last_string is False):
                    self.ui.textEdit_main_out.insertPlainText("\n")
                    self.last_string_was_false = True
                    self.inserted_newline_since_last_string = True
                else:
                    self.last_string_was_false = True

            self.last_string_was_r = False

    def save_points(self):
        self.optimiser.save_suggested_steps()

    def freeze_buttons(self):

        for dict_item in self.ui.__dict__:
            if "pushButton" in dict_item:
                self.ui.__getattribute__(dict_item).setDisabled(True)

        for widget_dict in self.ls_tab_widgets:
            for dict_item in widget_dict:
                if "pushButton" in dict_item:
                    widget_dict[dict_item].setDisabled(True)

    def unfreeze_buttons(self):

        for dict_item in self.ui.__dict__:
            if "pushButton" in dict_item:
                self.ui.__getattribute__(dict_item).setDisabled(False)

        for widget_dict in self.ls_tab_widgets:
            for dict_item in widget_dict:
                if "pushButton" in dict_item:
                    widget_dict[dict_item].setDisabled(False)

        self.update_buttons()


class OptWorker(QObject):
    working_signal = Signal()
    update_signal = Signal()
    finished_signal = Signal()

    def __init__(self, optimiser: BayesOptimiser):
        self.optimiser = optimiser
        self.window = None
        super(OptWorker, self).__init__()

    def suggest_bayes_steps(self):
        self.working_signal.emit()

        self.optimiser.suggest_bayes_steps()

        self.finished_signal.emit()

    def do_opt_steps(self):

        self.run_opt_step()

        while self.window.ui.checkBox_keep_running.checkState() \
                and self.optimiser.current_point < self.optimiser.n_points:

            self.run_opt_step()

        self.finished_signal.emit()

    def run_opt_step(self):
        self.working_signal.emit()

        self.optimiser.run_opt_step()

        self.update_signal.emit()

    def refit_model(self):
        self.working_signal.emit()

        self.optimiser.refit_model()

        self.finished_signal.emit()


class WriteStream(object):
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)


class Receiver(QObject):
    signal = Signal(str)

    def __init__(self, queue):
        QObject.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            text = self.queue.get()
            self.signal.emit(text)


def run(optimiser):
    app = QApplication(sys.argv)

    # Thread for running slow parts of the optimiser without pausing GUI

    opt_worker = OptWorker(optimiser)
    opt_thread = QThread()

    opt_worker.moveToThread(opt_thread)

    app.aboutToQuit.connect(opt_thread.quit)

    opt_thread.start()

    # Queue and thread for updating text field
    queue = Queue()
    sys.stdout = WriteStream(queue)

    window = BayesOptWindow(optimiser, opt_worker)
    window.show()

    write_thread = QThread()
    receiver = Receiver(queue)
    receiver.signal.connect(window.write_to_textfield)
    receiver.moveToThread(write_thread)
    write_thread.started.connect(receiver.run)
    app.aboutToQuit.connect(write_thread.quit)

    write_thread.start()

    # app.exec_()

    sys.exit(app.exec_())
