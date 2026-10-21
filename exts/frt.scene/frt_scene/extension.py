import omni.ext
import omni.kit.menu.utils as mu
import omni.ui as ui
from omni.kit.menu.utils import MenuItemDescription

from .window import TITLE, ReviewWindow


class FrtSceneExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._window = None
        self._menu = [MenuItemDescription(name="Factory Review", onclick_fn=self.show)]
        mu.add_menu_items(self._menu, "Window")
        ui.Workspace.set_show_window_fn(TITLE, lambda visible: self.show() if visible else None)

    def show(self, *_):
        if self._window is None:
            self._window = ReviewWindow()
        self._window.window.visible = True
        self._window.window.focus()

    def on_shutdown(self):
        mu.remove_menu_items(self._menu, "Window")
        ui.Workspace.set_show_window_fn(TITLE, None)
        if self._window:
            self._window.destroy()               # or a hot reload leaves a dead copy of the window behind
            self._window = None
