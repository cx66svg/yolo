"""项目管理页：建档、拖入数据集、查看项目详情。

设计目标：用户不必理解类别方案，只要给一个符合 YOLO v8 格式的数据集，
系统自己读出类别数量和类名，据此建立网络。类别数完全由数据决定。
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QTableWidget, QTextBrowser,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ... import config
from ...data import load_profile
from ...data.import_tools import extract_archive
from ..state import AppState
from ..widgets import Card, PageHeader, StatRow


class NewProjectDialog(QDialog):
    """新建项目：填名称、选数据集（支持把文件夹或压缩包拖进来）。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("新建项目")
        self.setMinimumWidth(560)
        self.setAcceptDrops(True)
        self._dataset: Path | None = None
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        tip = QLabel("把数据集文件夹或 .zip 压缩包拖到这里，也可以点「浏览」选择。\n"
                     "结构需符合 YOLO v8 格式：train/images、train/labels（valid、test 可选）。")
        tip.setObjectName("Hint")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        self.drop_zone = QLabel("拖入数据集到这里")
        self.drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_zone.setMinimumHeight(72)
        self.drop_zone.setStyleSheet(
            "border:2px dashed #B5C0D0; border-radius:6px; color:#6B7280;"
        )
        layout.addWidget(self.drop_zone)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如：山羊行为_8类")
        form.addRow("项目名称", self.name_edit)

        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("数据集目录或压缩包路径")
        self.path_edit.textChanged.connect(self._on_path_changed)
        btn = QPushButton("浏览…")
        btn.clicked.connect(self._browse)
        row.addWidget(self.path_edit, 1)
        row.addWidget(btn)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("数据集", holder)

        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("可选，例如：老师提供的数据集，4 类行为")
        form.addRow("备注", self.note_edit)
        layout.addLayout(form)

        self.info = QLabel("尚未选择数据集")
        self.info.setObjectName("Hint")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("创建")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------ 拖拽
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.exists():
                self.path_edit.setText(str(path))
                break

    # ------------------------------------------------------------ 交互
    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "或选择压缩包", "", "压缩包 (*.zip)")
        if path:
            self.path_edit.setText(path)

    def _on_path_changed(self, text: str) -> None:
        text = text.strip()
        self._dataset = Path(text) if text else None
        if not self._dataset or not self._dataset.exists():
            self.info.setText("路径不存在")
            return
        try:
            profile = self._preview_profile(self._dataset)
        except (OSError, ValueError) as exc:
            self.info.setText(f"解析失败：{exc}")
            return
        if not profile.splits:
            self.info.setText("没有找到 images 目录，结构可能不符合 YOLO 格式")
            return
        names = "、".join(list(profile.names.values())[:6])
        if profile.num_classes > 6:
            names += " …"
        self.info.setText(
            f"识别到 {profile.num_classes} 类：{names}\n"
            f"数据量：{profile.summary()}"
        )
        if not self.name_edit.text().strip():
            self.name_edit.setText(self._dataset.stem)

    @staticmethod
    def _preview_profile(path: Path):
        """压缩包先解压到临时位置再解析，避免建档失败。"""
        if path.is_file() and path.suffix.lower() == ".zip":
            target = config.OUTPUT_DIR / "datasets" / "imported" / path.stem
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
            if not target.exists() or not any(target.iterdir()):
                extract_archive(path, target)
            return load_profile(target)
        return load_profile(path)

    # ------------------------------------------------------------ 结果
    def values(self) -> tuple[str, Path, str]:
        return (self.name_edit.text().strip(),
                Path(self.path_edit.text().strip()),
                self.note_edit.text().strip())

    def accept(self) -> None:  # noqa: D102
        name, path, _ = self.values()
        if not name:
            QMessageBox.warning(self, "提示", "请填写项目名称")
            return
        if not path.exists():
            QMessageBox.warning(self, "提示", "数据集路径不存在")
            return
        super().accept()


class ProjectPage(QWidget):
    """项目列表与详情。"""

    project_opened = pyqtSignal(object)

    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self._build()
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        root.addWidget(PageHeader(
            "项目管理",
            "每个项目对应一个数据集的完整训练档案。新建项目后拖入符合 YOLO v8 格式的数据集，"
            "系统自动识别类别数量与名称，无需事先定义类别方案。",
        ))

        self.stats = StatRow(["项目数", "当前项目", "类别数", "数据量"])
        root.addWidget(self.stats)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(12)
        body.addLayout(left, 3)
        left.addWidget(self._list_card(), 1)

        right = QVBoxLayout()
        right.setSpacing(12)
        body.addLayout(right, 2)
        right.addWidget(self._detail_card(), 1)

    def _list_card(self) -> Card:
        card = Card("项目列表")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["项目名称", "类别数", "图像数", "创建时间"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(38)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        # 允许多选：Ctrl 逐个点、Shift 连选，供批量删除使用
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.doubleClicked.connect(lambda _: self._open_selected())
        card.add(self.table)

        row = QHBoxLayout()
        self.btn_new = QPushButton("新建项目")
        self.btn_new.setObjectName("Primary")
        self.btn_new.clicked.connect(self._create)
        self.btn_open = QPushButton("打开项目")
        self.btn_open.clicked.connect(self._open_selected)
        self.btn_del = QPushButton("删除选中")
        self.btn_del.setObjectName("Danger")
        self.btn_del.clicked.connect(self._delete_selected)
        self.btn_reload = QPushButton("刷新")
        self.btn_reload.clicked.connect(self.refresh)
        row.addWidget(self.btn_new)
        row.addWidget(self.btn_open)
        row.addWidget(self.btn_del)
        row.addWidget(self.btn_reload)
        row.addStretch(1)
        card.add_layout(row)
        return card

    def _detail_card(self) -> Card:
        card = Card("项目详情")
        self.detail = QTextBrowser()
        self.detail.setOpenExternalLinks(False)
        self.detail.setObjectName("DetailView")
        self.detail.document().setDocumentMargin(14)
        self.detail.setHtml(
            '<div style="color:#9CA3AF;font-size:12px;padding:18px 4px;">'
            "在左侧列表选择一个项目，这里会显示它的数据集、类别与训练记录。</div>"
        )
        card.add(self.detail)
        return card

    # ------------------------------------------------------------ 数据
    def refresh(self) -> None:
        store = self.state.store
        projects = store.all()
        self.table.setRowCount(0)
        for project in projects:
            images = 0
            try:
                profile = load_profile(Path(project.dataset_root))
                images = profile.total_images
            except (OSError, ValueError):
                images = -1
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [project.name, project.num_classes,
                      images if images >= 0 else "不可用", project.created_at]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col in (1, 2):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)
            self.table.item(row, 0).setData(Qt.ItemDataRole.UserRole, project.pid)

        self.stats.set("项目数", str(len(projects)))
        current = self.state.project
        if current:
            self.stats.set("当前项目", current.name)
            self.stats.set("类别数", str(current.num_classes))
        else:
            self.stats.set("当前项目", "未打开")
            self.stats.set("类别数", "—")

    def _selected_pid(self) -> str | None:
        pids = self._selected_pids()
        return pids[0] if pids else None

    def _selected_pids(self) -> list[str]:
        """返回所有被选中行的项目 id（去重，按表格顺序）。"""
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        pids: list[str] = []
        for row in rows:
            cell = self.table.item(row, 0)
            if cell is None:
                continue
            pid = cell.data(Qt.ItemDataRole.UserRole)
            if pid and pid not in pids:
                pids.append(pid)
        return pids

    def _on_select(self) -> None:
        pid = self._selected_pid()
        project = self.state.store.get(pid) if pid else None
        if not project:
            self._clear_detail()
            return
        ok, msg = self.state.store.validate(project)
        self.detail.setHtml(self._detail_html(project, ok, msg))

    @staticmethod
    def _detail_html(project, ok: bool, msg: str) -> str:
        """把项目信息渲染成分节富文本。

        QTextBrowser 只支持 HTML 的一个子集：表格与 inline-block 会塌成一行，
        所以这里只用块级 p 加内联样式，保证在 Qt 里稳定渲染。
        """
        faded = "color:#6B7280;"
        section = ('<p style="font-weight:bold;color:#111827;'
                   'margin:14px 0 2px 0;">{}</p>')
        line = '<p style="margin:2px 0;color:#111827;">{}</p>'

        status_color = "#16A34A" if ok else "#DC2626"
        chips = " · ".join(f"{cid} {name}" for cid, name in sorted(project.class_names.items()))

        parts = [
            f'<p style="font-size:17px;font-weight:bold;color:#111827;margin:0;">{project.name}</p>',
            f'<p style="color:{status_color};margin:2px 0 0 0;">● {"就绪" if ok else msg}</p>',
            section.format("基本信息"),
            line.format(f'<span style="{faded}">创建时间　</span>{project.created_at}'),
            line.format(f'<span style="{faded}">类别数量　</span>{project.num_classes} 类'),
            line.format(f'<span style="{faded}">训练次数　</span>{len(project.runs)} 次'),
            line.format(f'<span style="{faded}">数据集　　</span>{project.dataset_root}'),
        ]
        if project.best_weights:
            parts.append(line.format(
                f'<span style="{faded}">当前权重　</span>{project.best_weights}'))

        parts.append(section.format("类别清单"))
        parts.append(line.format(chips or "（无）"))

        if project.runs:
            parts.append(section.format("训练记录"))
            for name in project.runs[-10:]:
                parts.append(line.format(f"· {name}"))

        if project.note:
            parts.append(section.format("备注"))
            parts.append(line.format(project.note))

        return "".join(parts)

    def _clear_detail(self) -> None:
        self.detail.setHtml(
            '<div style="color:#9CA3AF;font-size:12px;padding:18px 4px;">'
            "在左侧列表选择一个项目，这里会显示它的数据集、类别与训练记录。</div>"
        )

    # ------------------------------------------------------------ 动作
    def _create(self) -> None:
        dialog = NewProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, path, note = dialog.values()
        try:
            project = self.state.store.create(name, path, note)
        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            QMessageBox.critical(self, "创建失败", str(exc))
            return
        self.refresh()
        self.state.open_project(project)
        self.state.notify(f"已创建项目「{project.name}」，{project.num_classes} 类")
        QMessageBox.information(
            self, "创建成功",
            f"项目「{project.name}」已创建。\n"
            f"类别数：{project.num_classes}\n"
            f"类别：{project.class_summary()}\n\n"
            "可以到「模型训练」页开始训练。",
        )

    def _open_selected(self) -> None:
        pid = self._selected_pid()
        if not pid:
            QMessageBox.information(self, "提示", "请先在列表中选择一个项目")
            return
        project = self.state.store.get(pid)
        if not project:
            return
        ok, msg = self.state.store.validate(project)
        if not ok:
            QMessageBox.warning(self, "数据集不可用", msg)
            return
        self.state.open_project(project)
        self.stats.set("当前项目", project.name)
        self.stats.set("类别数", str(project.num_classes))
        self.state.notify(f"已打开项目「{project.name}」")
        self.project_opened.emit(project)

    def _delete_selected(self) -> None:
        """删除选中的项目，支持一次删除多个。"""
        pids = self._selected_pids()
        if not pids:
            QMessageBox.information(self, "提示", "请先在列表中选中要删除的项目（可按住 Ctrl 多选）")
            return
        projects = [p for p in (self.state.store.get(pid) for pid in pids) if p]
        if not projects:
            return

        if len(projects) == 1:
            detail = f"「{projects[0].name}」"
        else:
            shown = "、".join(p.name for p in projects[:5])
            more = f" 等 {len(projects)} 个" if len(projects) > 5 else ""
            detail = f"{len(projects)} 个项目：{shown}{more}"
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定删除{detail}？\n\n"
            "只删除建档记录，不会删除数据集与已训练的权重文件。",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        current = self.state.project
        for project in projects:
            self.state.store.delete(project.pid)
        if current and current.pid in {p.pid for p in projects}:
            self.state.open_project(None)
        self.refresh()
        self.detail.clear()
        self.state.notify(f"已删除 {len(projects)} 个项目")
