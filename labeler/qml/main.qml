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
            case Qt.Key_S:
                if (event.modifiers & Qt.ControlModifier) {
                    backend.save(); event.accepted = true
                }
                break
            case Qt.Key_E:
                if (event.modifiers & Qt.ControlModifier) {
                    backend.export_csv(); event.accepted = true
                }
                break
            case Qt.Key_Space:
                // Toggle the ±1s context loop on the main canvas.
                backend.toggle_playback()
                event.accepted = true
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

    // Autosave pulse indicator: fades in/out when autosave fires.
    Row {
        id: autosaveIndicator
        spacing: 6
        anchors.right: canvas.right
        anchors.top: canvas.top
        anchors.rightMargin: 8
        anchors.topMargin: 8
        opacity: 0

        Rectangle {
            width: 8; height: 8; radius: 4
            color: "#a6e3a1"   // green
            anchors.verticalCenter: parent.verticalCenter
        }
        Text {
            text: "autosaved"
            color: "#a6e3a1"
            font.family: "JetBrains Mono, Consolas, Courier New"
            font.pixelSize: 11
            anchors.verticalCenter: parent.verticalCenter
        }

        // Two visible pulses across ~2.6s: fade-in, hold, dip, hold, fade-out.
        SequentialAnimation {
            id: autosaveAnim
            NumberAnimation { target: autosaveIndicator; property: "opacity"; to: 1.0; duration: 280; easing.type: Easing.OutCubic }
            PauseAnimation  { duration: 700 }
            NumberAnimation { target: autosaveIndicator; property: "opacity"; to: 0.35; duration: 220; easing.type: Easing.InOutSine }
            NumberAnimation { target: autosaveIndicator; property: "opacity"; to: 1.0; duration: 220; easing.type: Easing.InOutSine }
            PauseAnimation  { duration: 700 }
            NumberAnimation { target: autosaveIndicator; property: "opacity"; to: 0.0; duration: 500; easing.type: Easing.InCubic }
        }

        Connections {
            target: backend
            function onAutosavePulse() { autosaveAnim.restart() }
        }
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
        height: 340
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

        Row {
            anchors.left: parent.left
            anchors.leftMargin: 16
            anchors.verticalCenter: parent.verticalCenter
            spacing: 12

            // Tiny play/pause indicator (mauve when playing, dim when paused).
            Text {
                anchors.verticalCenter: parent.verticalCenter
                color: backend.isPlaying ? "#cba6f7" : "#6c7086"
                font.family: "JetBrains Mono, Consolas, Courier New"
                font.pixelSize: 13
                text: backend.isPlaying ? "▶ PLAY" : "■ PAUSE"
            }

            Text {
                anchors.verticalCenter: parent.verticalCenter
                color: "#cdd6f4"
                font.family: "JetBrains Mono, Consolas, Courier New"
                font.pixelSize: 13
                text: {
                    const sel = backend.selectedDetIdx
                    const showFrame = backend.isPlaying ? backend.displayFrame : backend.currentFrame
                    const base = "frame " + (showFrame + 1) + " / " + backend.frameCount
                    if (sel < 0) return base
                    return base + "    selected: det #" + sel
                }
            }
        }

        // Center: transient status (saves, exports, autosaves, errors)
        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter
            color: "#a6e3a1"
            font.family: "JetBrains Mono, Consolas, Courier New"
            font.pixelSize: 12
            text: backend.statusText
            visible: text.length > 0
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
