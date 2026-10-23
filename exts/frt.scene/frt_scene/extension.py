import os

import omni.ext
import omni.kit.menu.utils as mu
import omni.ui as ui
import omni.usd
from omni.kit.menu.utils import MenuItemDescription

from frt_core import usdio

from .review_window import TITLE as REVIEW_TITLE, DrawingReviewWindow
from .task_window import TITLE as TASK_TITLE, TaskWindow
from .window import TITLE, ReviewWindow


class FrtSceneExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._window = None
        self._review = None
        self._task = None
        self._menu = [MenuItemDescription(name="Factory Review", onclick_fn=self.show),
                      MenuItemDescription(name="Drawing Review", onclick_fn=self.show_review),
                      MenuItemDescription(name="Task Builder", onclick_fn=self.show_task)]
        mu.add_menu_items(self._menu, "Window")
        ui.Workspace.set_show_window_fn(TITLE, lambda visible: self.show() if visible else None)
        ui.Workspace.set_show_window_fn(REVIEW_TITLE, lambda visible: self.show_review() if visible else None)
        ui.Workspace.set_show_window_fn(TASK_TITLE, lambda visible: self.show_task() if visible else None)
        global _EXT
        _EXT = self
        # leave a <scene>.open marker beside the open scene so command-line builds don't swap it under us
        self._marked = None
        self._stage_sub = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event, name="frt open marker")
        self._mark_current()

    def _mark_current(self):
        self._unmark()
        st = omni.usd.get_context().get_stage()
        path = st.GetRootLayer().realPath if st else ""
        if path and os.path.isfile(path):
            usdio.mark_open(path)
            self._marked = path

    def _unmark(self):
        if self._marked:
            usdio.clear_open(self._marked)
            self._marked = None

    def _on_stage_event(self, e):
        if e.type in (int(omni.usd.StageEventType.OPENED), int(omni.usd.StageEventType.SAVED)):
            self._mark_current()                         # SAVED: a Save As gives the stage a new file
        elif e.type in (int(omni.usd.StageEventType.CLOSING), int(omni.usd.StageEventType.CLOSED)):
            self._unmark()

    def show(self, *_):
        if self._window is None:
            self._window = ReviewWindow()
        self._window.window.visible = True
        self._window.window.focus()

    def show_review(self, *_):
        if self._review is None:
            self._review = DrawingReviewWindow()
        self._review.window.visible = True
        self._review.window.focus()

    def show_task(self, *_):
        if self._task is None:
            self._task = TaskWindow()
        self._task.window.visible = True
        self._task.window.focus()
        return self._task

    def on_shutdown(self):
        global _EXT
        _EXT = None
        self._stage_sub = None
        self._unmark()
        mu.remove_menu_items(self._menu, "Window")
        ui.Workspace.set_show_window_fn(TITLE, None)
        ui.Workspace.set_show_window_fn(REVIEW_TITLE, None)
        ui.Workspace.set_show_window_fn(TASK_TITLE, None)
        if self._task:
            self._task.window.destroy()
            self._task = None
        if self._review:
            self._review.destroy()
            self._review = None
        if self._window:
            self._window.destroy()               # or a hot reload leaves a dead copy of the window behind
            self._window = None


_EXT = None


def show_task():
    """Open Window > Task Builder from a script: frt_scene.extension.show_task() -> the TaskWindow."""
    return _EXT.show_task() if _EXT else None
