import sys
import json
import csv
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
                             QCheckBox, QPushButton, QFrame, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QGridLayout, QGroupBox, QStackedWidget, QMessageBox,
                             QComboBox, QFileDialog, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import core

class UAVEnergyModelApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV Energy Model Selection Framework")
        self.setGeometry(100, 100, 500, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Header
        self.header = QLabel("UAV Energy Model Selection Framework")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setFont(QFont("Arial", 18, QFont.Bold))
        self.main_layout.addWidget(self.header)
        
        # Navigation
        self.nav_layout = QHBoxLayout()
        self.btn_feasibility = QPushButton("1. Feasibility")
        self.btn_ahp = QPushButton("2. Criteria Comparison")
        self.btn_results = QPushButton("3. Results")
        
        for btn in [self.btn_feasibility, self.btn_ahp, self.btn_results]:
            btn.setCheckable(True)
            self.nav_layout.addWidget(btn)
        
        self.btn_group = [self.btn_feasibility, self.btn_ahp, self.btn_results]
        self.btn_feasibility.clicked.connect(lambda: self.switch_page(0))
        self.btn_ahp.clicked.connect(lambda: self.switch_page(1))
        self.btn_results.clicked.connect(lambda: self.switch_page(2))
        
        self.main_layout.addLayout(self.nav_layout)
        
        # Stacked Pages
        self.stack = QStackedWidget()
        self.page_feasibility = FeasibilityPage(self)
        self.page_ahp = AHPPage(self)
        self.page_results = ResultsPage(self)
        
        self.stack.addWidget(self.page_feasibility)
        self.stack.addWidget(self.page_ahp)
        self.stack.addWidget(self.page_results)
        
        self.main_layout.addWidget(self.stack)
        
        # Initial State
        self.switch_page(0)
        self.feasible_alternatives = []
        self.criteria_weights = None
        self.global_scores = None

    def switch_page(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.btn_group):
            btn.setChecked(i == index)

    def set_feasible_alternatives(self, alts):
        self.feasible_alternatives = alts
        self.page_results.update_feasibility_status(alts)
        # Enable AHP page only if alternatives exist? No, AHP is independent of alternatives technically, 
        # but final results depend on both.
        if not alts:
            QMessageBox.warning(self, "No Feasible Options", 
                                "No modeling strategies are feasible with current constraints.\n"
                                "Please reconsider your feasibility inputs.")

    def set_criteria_weights(self, weights, cr, matrix):
        self.criteria_weights = weights
        self.page_results.update_results(self.feasible_alternatives, weights)

class FeasibilityPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent_app = parent
        layout = QVBoxLayout(self)
        
        info_label = QLabel("Stage 1: Feasibility Filtering\n"
                            "Select the conditions that apply to your project to determine valid modeling strategies.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        self.checkboxes = []
        for i, text in enumerate(core.FEASIBILITY_INDICATORS):
            cb = QCheckBox(text)
            self.checkboxes.append(cb)
            layout.addWidget(cb)
            
        btn_check = QPushButton("Check Feasibility")
        btn_check.clicked.connect(self.check_feasibility)
        layout.addWidget(btn_check)
        
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        layout.addStretch()

    def check_feasibility(self):
        f_vector = [1 if cb.isChecked() else 0 for cb in self.checkboxes]
        admissible = core.check_feasibility(f_vector)
        self.parent_app.set_feasible_alternatives(admissible)
        
        if admissible:
            text = "<b>Feasible Alternatives:</b><ul>"
            for idx in admissible:
                text += f"<li>{core.ALTERNATIVES[idx]}</li>"
            text += "</ul>"
            self.result_label.setText(text)
            QMessageBox.information(self, "Success", "Feasibility check passed. Proceed to Criteria Comparison.")
            self.parent_app.switch_page(1)
        else:
            # Provide specific hints based on what was checked
            hints = []
            if f_vector[0] == 1: # F1 checked
                hints.append("• <b>F1 (Widely Recognized)</b> is selected, but it doesn't automatically guarantee a model exists. "
                             "Please check literature: if a white-box model exists, select <b>F2</b>; if a black-box model exists, select <b>F3</b>.")
            
            if sum(f_vector) == 0:
                hints.append("• No conditions selected. You need at least some resources (Infrastructure, Data, or Existing Models) to proceed.")
            
            if not hints:
                 hints.append("• Current resource combination doesn't support any standard modeling path.")

            hint_text = "<br>".join(hints)
            self.result_label.setText(f"<font color='red'>No feasible alternatives found.</font><br><br><b>Hints:</b><br>{hint_text}")

class AHPPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent_app = parent
        layout = QVBoxLayout(self)
        
        info = QLabel("Stage 2: Criteria Comparison (AHP)\n"
                      "Compare the importance of each criterion against others using the 1-9 scale.\n"
                      "1: Equal, 3: Moderate, 5: Strong, 7: Very Strong, 9: Extreme")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Smart Fill Checkbox
        self.smart_fill_cb = QCheckBox("Enable Smart Fill (Auto-suggest values based on logic)")
        self.smart_fill_cb.setToolTip("When enabled, filling one cell will automatically infer other empty cells based on transitivity (A>B & B>C -> A>C).")
        layout.addWidget(self.smart_fill_cb)

        # Table
        self.table = QTableWidget(4, 4)
        self.table.setHorizontalHeaderLabels(["Accuracy", "Interpretability", "Cost/Time", "Customization"])
        self.table.setVerticalHeaderLabels(["Accuracy", "Interpretability", "Cost/Time", "Customization"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Initialize Table
        self.scale_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 
                             1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9]
        self.scale_options = ["-"] + ["1", "2", "3", "4", "5", "6", "7", "8", "9", 
                              "1/2", "1/3", "1/4", "1/5", "1/6", "1/7", "1/8", "1/9"]
        
        # Internal matrix to track state (0 = empty)
        self.matrix = np.zeros((4, 4))
        np.fill_diagonal(self.matrix, 1.0)

        for r in range(4):
            for c in range(4):
                if r == c:
                    item = QTableWidgetItem("1")
                    item.setFlags(Qt.ItemIsEnabled)
                    item.setBackground(QColor(240, 240, 240))
                    self.table.setItem(r, c, item)
                elif r < c:
                    combo = QComboBox()
                    combo.addItems(self.scale_options)
                    combo.setCurrentText("-")
                    # Store row, col in the widget property for easy access
                    combo.setProperty("row", r)
                    combo.setProperty("col", c)
                    combo.currentTextChanged.connect(self.on_cell_changed)
                    self.table.setCellWidget(r, c, combo)
                else:
                    item = QTableWidgetItem("-")
                    item.setFlags(Qt.ItemIsEnabled) # Read only, updated by code
                    self.table.setItem(r, c, item)
                    
        layout.addWidget(self.table)
        
        btn_calc = QPushButton("Calculate Weights")
        btn_calc.clicked.connect(self.calculate)
        layout.addWidget(btn_calc)
        
        self.stats_label = QLabel("Consistency Ratio (CR): -")
        layout.addWidget(self.stats_label)
        
        self.suggestion_label = QLabel("")
        self.suggestion_label.setWordWrap(True)
        self.suggestion_label.setStyleSheet("color: blue; font-style: italic;")
        layout.addWidget(self.suggestion_label)

    def on_cell_changed(self, text):
        sender = self.sender()
        if not sender: return
        
        r = sender.property("row")
        c = sender.property("col")
        
        # Avoid recursion
        self.table.blockSignals(True)
        sender.blockSignals(True)
        
        if text == "-":
            self.matrix[r, c] = 0
            self.matrix[c, r] = 0
            self.table.item(c, r).setText("-")
        else:
            try:
                val = float(eval(text))
                self.matrix[r, c] = val
                self.matrix[c, r] = 1.0 / val
                self.table.item(c, r).setText(f"{1.0/val:.3f}")
                
                # Smart Fill Logic
                if self.smart_fill_cb.isChecked():
                    self.apply_smart_fill()
            except:
                pass
                
        sender.blockSignals(False)
        self.table.blockSignals(False)

    def apply_smart_fill(self):
        # Iterate over empty cells (where r < c)
        for r in range(4):
            for c in range(4):
                if r < c:
                    combo = self.table.cellWidget(r, c)
                    if combo.currentText() == "-":
                        # Try to infer
                        val = core.get_smart_fill_value(self.matrix, r, c)
                        if val:
                            # Find closest scale option
                            closest_text = self.get_closest_scale_text(val)
                            
                            # Set value
                            combo.blockSignals(True)
                            combo.setCurrentText(closest_text)
                            combo.setStyleSheet("color: blue; font-weight: bold;") # Highlight inferred
                            combo.setToolTip(f"Auto-filled based on logic (Exact: {val:.2f})")
                            combo.blockSignals(False)
                            
                            # Update matrix and reciprocal
                            actual_val = float(eval(closest_text))
                            self.matrix[r, c] = actual_val
                            self.matrix[c, r] = 1.0 / actual_val
                            self.table.item(c, r).setText(f"{1.0/actual_val:.3f}")

    def get_closest_scale_text(self, value):
        # Scale values corresponding to scale_options[1:] (skipping "-")
        # We need to match value to one of self.scale_values
        
        # Use log distance
        log_val = np.log(value)
        min_dist = float('inf')
        best_idx = 0
        
        for i, s_val in enumerate(self.scale_values):
            dist = abs(np.log(s_val) - log_val)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
                
        # Map index back to text
        # scale_options has "-" at index 0, so +1
        return self.scale_options[best_idx + 1]

    def update_reciprocal(self, row, col, text):
        # Deprecated by on_cell_changed
        pass

    def calculate(self):
        # Check for completeness
        if np.any(self.matrix == 0):
             QMessageBox.warning(self, "Incomplete Matrix", "Please fill all comparisons before calculating.")
             return

        weights, cr = core.calculate_priority_vector(self.matrix)
        
        status_color = "green" if cr < 0.1 else "red"
        status_text = "Acceptable" if cr < 0.1 else "Inconsistent"
        self.stats_label.setText(f"Consistency Ratio (CR): {cr:.4f} - <font color='{status_color}'>{status_text}</font>")
        
        # Reset styles
        for r in range(4):
            for c in range(4):
                if r < c:
                    self.table.cellWidget(r, c).setStyleSheet("")
        self.suggestion_label.setText("")

        if cr < 0.1:
            self.parent_app.set_criteria_weights(weights, cr, self.matrix)
            QMessageBox.information(self, "Success", "Weights calculated successfully. Proceed to Results.")
            self.parent_app.switch_page(2)
        else:
            # Highlight inconsistencies
            report = core.get_consistency_report(self.matrix, weights)
            if report:
                msg_list = []
                for item in report[:3]: # Top 3 issues
                    r, c = item['row'], item['col']
                    ideal = item['ideal']
                    
                    # Highlight cell
                    widget = self.table.cellWidget(r, c)
                    widget.setStyleSheet("background-color: #ffcccc; border: 1px solid red;")
                    
                    suggested_text = self.get_closest_scale_text(ideal)
                    tooltip = f"Current: {item['actual']:.2f}\nConsistent Value: {ideal:.2f}\nSuggested: {suggested_text}"
                    widget.setToolTip(tooltip)
                    
                    msg_list.append(f"• {core.CRITERIA[r+1].split(':')[0]} vs {core.CRITERIA[c+1].split(':')[0]}: Suggest changing to {suggested_text}")
                
                self.suggestion_label.setText("<b>Suggestions to improve consistency:</b><br>" + "<br>".join(msg_list))
                
            QMessageBox.warning(self, "Consistency Warning", 
                                f"CR is {cr:.4f} > 0.1.\n"
                                "Your judgments contain logical contradictions.\n"
                                "Please review the highlighted cells and suggestions below the table.")

class ResultsPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent_app = parent
        layout = QVBoxLayout(self)
        
        self.label = QLabel("Stage 3: Ranking & Recommendations")
        layout.addWidget(self.label)
        
        # Chart
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Export Buttons
        btn_layout = QHBoxLayout()
        self.btn_export_pdf = QPushButton("Export PDF")
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_json = QPushButton("Export JSON")
        
        self.btn_export_pdf.clicked.connect(self.export_pdf)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_json.clicked.connect(self.export_json)
        
        btn_layout.addWidget(self.btn_export_pdf)
        btn_layout.addWidget(self.btn_export_csv)
        btn_layout.addWidget(self.btn_export_json)
        layout.addLayout(btn_layout)
        
        self.current_scores = {} # {alt_idx: score}

    def update_feasibility_status(self, alts):
        if not alts:
            self.figure.clear()
            self.canvas.draw()
            
    def update_results(self, feasible_alts, weights):
        if not feasible_alts:
            return
            
        # Calculate Global Scores
        # Re-using core logic but we need to pass weights
        # Actually core has calculate_global_scores but that takes matrix
        # We already have weights.
        
        # S_j = sum(w_i * a_j^(i))
        S = np.zeros(5)
        for i in range(4):
            S += weights[i] * core.LOCAL_PRIORITIES[i+1]
            
        # Filter by feasible
        self.current_scores = {}
        for idx in feasible_alts:
            self.current_scores[idx] = S[idx-1] # S is 0-indexed (O1 is at 0)
            
        # Plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        labels = [f"O{idx}" for idx in self.current_scores.keys()]
        values = list(self.current_scores.values())
        colors = ['skyblue' if v != max(values) else 'lightgreen' for v in values]
        
        bars = ax.bar(labels, values, color=colors)
        ax.set_title("Feasible Alternative Scores")
        ax.set_ylabel("Global Priority Score")
        ax.set_ylim(0, max(values)*1.2)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}', ha='center', va='bottom')
            
        # Add legend/description
        desc = "\n".join([f"O{k}: {core.ALTERNATIVES[k]}" for k in self.current_scores.keys()])
        self.figure.text(0.02, 0.02, desc, fontsize=8, verticalalignment='bottom')
        self.figure.subplots_adjust(bottom=0.25)
        
        self.canvas.draw()

    def export_pdf(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save PDF", "", "PDF Files (*.pdf)")
        if filename:
            c = canvas.Canvas(filename, pagesize=letter)
            width, height = letter
            
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "UAV Energy Model Selection Report")
            
            c.setFont("Helvetica", 12)
            y = height - 80
            c.drawString(50, y, "Recommended Strategy:")
            y -= 20
            
            if self.current_scores:
                best_idx = max(self.current_scores, key=self.current_scores.get)
                c.drawString(70, y, f"{core.ALTERNATIVES[best_idx]} (Score: {self.current_scores[best_idx]:.4f})")
                y -= 40
                
                c.drawString(50, y, "All Feasible Scores:")
                y -= 20
                for idx, score in self.current_scores.items():
                    c.drawString(70, y, f"{core.ALTERNATIVES[idx]}: {score:.4f}")
                    y -= 20
            
            c.save()
            QMessageBox.information(self, "Export", "PDF saved successfully.")

    def export_csv(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if filename:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Alternative ID", "Description", "Score"])
                for idx, score in self.current_scores.items():
                    writer.writerow([f"O{idx}", core.ALTERNATIVES[idx], score])
            QMessageBox.information(self, "Export", "CSV saved successfully.")

    def export_json(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON Files (*.json)")
        if filename:
            data = {
                "feasible_alternatives": [
                    {"id": f"O{idx}", "description": core.ALTERNATIVES[idx], "score": score}
                    for idx, score in self.current_scores.items()
                ]
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            QMessageBox.information(self, "Export", "JSON saved successfully.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UAVEnergyModelApp()
    window.show()
    sys.exit(app.exec_())