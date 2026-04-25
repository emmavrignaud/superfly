import QtQuick
import QtQuick.Window
import QtQuick.Controls

ApplicationWindow {
    id: root
    width: 1400
    height: 850
    visible: true
    title: "Superfly Labeler — frame " + (backend.currentFrame + 1) + " / " + backend.frameCount
    color: "#1e1e2e"

    // Centralized key handling. We handle digit accumulation here because the
    // pendingInput buffer lives on canvas and we want the same key events to
    // drive frame nav too.
    Item {
        id: keyHandler
        anchors.fill: parent
        focus: true

        function isDigitKey(k) { return k >= Qt.Key_0 && k <= Qt.Key_9 }

        Keys.onPressed: (event) => {
            const hasSelection = backend.selectedDetIdx >= 0

            // While a detection is selected, digits accumulate into the buffer.
            if (hasSelection && isDigitKey(event.key)) {
                canvas.pendingInput += String.fromCharCode(event.key)
                event.accepted = true
                return
            }

            switch (event.key) {
            case Qt.Key_Left:
                backend.seek_frame(backend.currentFrame - 1); event.accepted = true; break
            case Qt.Key_Right:
                backend.seek_frame(backend.currentFrame + 1); event.accepted = true; break
            case Qt.Key_PageUp:
                backend.seek_frame(backend.currentFrame - 10); event.accepted = true; break
            case Qt.Key_PageDown:
                backend.seek_frame(backend.currentFrame + 10); event.accepted = true; break
            case Qt.Key_Home:
                backend.seek_frame(0); event.accepted = true; break
            case Qt.Key_End:
                backend.seek_frame(backend.frameCount - 1); event.accepted = true; break
            case Qt.Key_B:
                canvas.showBboxes = !canvas.showBboxes; event.accepted = true; break
            case Qt.Key_Tab:
                if (event.modifiers & Qt.ShiftModifier) backend.select_prev()
                else backend.select_next()
                event.accepted = true; break
            case Qt.Key_Backspace:
                if (hasSelection && canvas.pendingInput.length > 0) {
                    canvas.pendingInput = canvas.pendingInput.slice(0, -1)
                    event.accepted = true
                }
                break
            case Qt.Key_Return:
            case Qt.Key_Enter:
                if (hasSelection && canvas.pendingInput.length > 0) {
                    const tid = parseInt(canvas.pendingInput, 10)
                    if (!isNaN(tid) && tid > 0) backend.assign_to_selection(tid)
                    canvas.pendingInput = ""
                    event.accepted = true
                }
                break
            case Qt.Key_Delete:
                if (hasSelection) {
                    backend.clear_selection_annotation()
                    event.accepted = true
                }
                break
            case Qt.Key_Escape:
                if (canvas.pendingInput.length > 0) canvas.pendingInput = ""
                else backend.clear_selection()
                event.accepted = true; break
            case Qt.Key_Z:
                if (event.modifiers & Qt.ControlModifier) {
                    backend.undo(); event.accepted = true
                }
                break
            }
        }
    }

    // ── Layout ───────────────────────────────────────────────────────────
    // Top row = canvas (flex) + track panel (fixed). Bottom = HUD strip.

    VideoCanvas {
        id: canvas
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.right: trackPanel.left
        anchors.bottom: hud.top
        anchors.margins: 12
    }

    TrackPanel {
        id: trackPanel
        width: 220
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: shortcutsPanel.top
        anchors.margins: 12
        anchors.leftMargin: 0
        anchors.bottomMargin: 8
    }

    ShortcutsPanel {
        id: shortcutsPanel
        width: 220
        height: 280
        anchors.right: parent.right
        anchors.bottom: hud.top
        anchors.rightMargin: 12
        anchors.bottomMargin: 12
    }

    Rectangle {
        id: hud
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 32
        color: "#181825"

        Text {
            anchors.left: parent.left
            anchors.leftMargin: 16
            anchors.verticalCenter: parent.verticalCenter
            color: "#cdd6f4"
            font.family: "JetBrains Mono, Consolas, Courier New"
            font.pixelSize: 13
            text: {
                const sel = backend.selectedDetIdx
                const base = "frame " + (backend.currentFrame + 1) + " / " + backend.frameCount
                if (sel < 0) return base
                return base + "    selected: det #" + sel
            }
        }

        Text {
            anchors.right: parent.right
            anchors.rightMargin: 16
            anchors.verticalCenter: parent.verticalCenter
            color: backend.selectedDetIdx < 0 ? "#6c7086" : "#cba6f7"
            font.family: "Segoe UI, sans-serif"
            font.pixelSize: 12
            text: backend.selectedDetIdx < 0
                ? "click a fly to start"
                : "type digits → Enter to commit"
        }
    }
}
