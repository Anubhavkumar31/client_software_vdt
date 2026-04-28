from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton
)


class ClusterSummaryDialog(QDialog):
    def __init__(self, cluster_df, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Cluster Summary")
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        # Table
        table = QTableWidget(self)
        table.setColumnCount(len(cluster_df.columns))
        table.setRowCount(len(cluster_df))
        table.setHorizontalHeaderLabels(cluster_df.columns.tolist())

        for r in range(len(cluster_df)):
            for c in range(len(cluster_df.columns)):
                table.setItem(
                    r, c,
                    QTableWidgetItem(str(cluster_df.iat[r, c]))
                )

        table.resizeColumnsToContents()
        layout.addWidget(table)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

        # def on_view_clusters_clicked(self):
        #     from PyQt6.QtWidgets import QMessageBox
        #     import pandas as pd
        #
        #     if self.pipe_tally is None or self.pipe_tally.empty:
        #         QMessageBox.warning(self, "Clusters", "No pipe data loaded")
        #         return
        #
        #     self.pipe_tally = self.pipe_tally.loc[:, ~self.pipe_tally.columns.duplicated()]
        #
        #     clusters = run_clustering(self.pipe_tally)
        #
        #     if not clusters:
        #         QMessageBox.information(self, "Clusters", "No clusters formed")
        #         return
        #
        #     cluster_rows = build_cluster_rows(clusters)
        #     cluster_df = pd.DataFrame(cluster_rows)
        #
        #     # 🔥 DIRECT USE (same file)
        #     dlg = ClusterSummaryDialog(cluster_df, self)
        #     dlg.exec()