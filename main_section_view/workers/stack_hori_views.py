from PyQt6.QtCore import Qt


def _apply_heatmap_layout_con(self, mode: str = None):
    """Apply horizontal (side-by-side) or vertical (stacked) layout for dual heatmaps"""
    # Use provided mode or fall back to current mode
    if mode is None:
        mode = getattr(self, '_hm_layout_mode', 'horizontal')

    # Safety checks
    if not hasattr(self, 'top_hsplit'):
        print("Warning: top_hsplit not found, skipping layout change")
        return

    self._hm_layout_mode = mode

    # Change splitter orientation
    if mode == "horizontal":
        self.top_hsplit.setOrientation(Qt.Orientation.Horizontal)
        if hasattr(self, 'btnToggleHmLayout'):
            self.btnToggleHmLayout.setText("stack" if mode == "horizontal" else "side-by-side")
        # Apply 50-50 split
        total = self.top_hsplit.width()
        left = int(total * 0.38)
        right = total - left
        self.top_hsplit.setSizes([left, right])
    else:  # vertical
        self.top_hsplit.setOrientation(Qt.Orientation.Vertical)
        if hasattr(self, 'btnToggleHmLayout'):
            self.btnToggleHmLayout.setText("Side-by-side")
        # Apply 50-50 split
        total = self.top_hsplit.height()
        top = (total // 2) - 95
        bottom = total - top
        self.top_hsplit.setSizes([top, bottom])

    print(f"Heatmap layout changed to: {mode}")