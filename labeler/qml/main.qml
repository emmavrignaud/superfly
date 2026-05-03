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

        // Auto-follow: backend asks the bubble to prefill an ID after a
        // ←/→ that landed on a nearest-fly auto-selection.
        Connections {
            target: backend
            function onPrefillRequested(trackId) {
                canvas.pendingInput = String(trackId)
            }
        }

        function isDigitKey(k) { return k >= Qt.Key_0 && k <= Qt.Key_9 }

        Keys.onPressed: (event) => {
            // -1 = no selection. Negative IDs other than -1 are synthetic dets.
            const hasSelection = backend.selectedDetIdx !== -1

            // Digits accumulate when a detection is selected OR when a segment
            // range is active (ready for bulk-assign).
            const hasSegment = backend.mode === "track"
                               && backend.segStart >= 0 && backend.segEnd >= 0
            if ((hasSelection || hasSegment) && isDigitKey(event.key)) {
                canvas.pendingInput += String.fromCharCode(event.key)
                event.accepted = true
                return
            }

            // Shift+arrow keys resize the selected synthetic detection's bbox.
            // Only fires for synthetics (backend slot is a no-op otherwise).
            if (event.modifiers & Qt.ShiftModifier) {
                switch (event.key) {
                case Qt.Key_Right: backend.resize_selected_synthetic(+4, 0); event.accepted = true; return
                case Qt.Key_Left:  backend.resize_selected_synthetic(-4, 0); event.accepted = true; return
                case Qt.Key_Down:  backend.resize_selected_synthetic(0, +4); event.accepted = true; return
                case Qt.Key_Up:    backend.resize_selected_synthetic(0, -4); event.accepted = true; return
                }
            }

            switch (event.key) {
            case Qt.Key_Left:
                backend.seek_frame(backend.currentFrame - 1); event.accepted = true; break
            case Qt.Key_Right:
                backend.seek_frame(backend.currentFrame + 1); event.accepted = true; break
            case Qt.Key_PageUp:
                backend.jump_frame(backend.currentFrame - 10); event.accepted = true; break
            case Qt.Key_PageDown:
                backend.jump_frame(backend.currentFrame + 10); event.accepted = true; break
            case Qt.Key_Home:
                backend.jump_frame(0); event.accepted = true; break
            case Qt.Key_End:
                backend.jump_frame(backend.frameCount - 1); event.accepted = true; break
            case Qt.Key_B:
                canvas.showBboxes = !canvas.showBboxes; event.accepted = true; break
            case Qt.Key_Plus:
            case Qt.Key_Equal:
                canvas.zoomIn()
                event.accepted = true
                break
            case Qt.Key_Minus:
            case Qt.Key_Underscore:
                canvas.zoomOut()
                event.accepted = true
                break
            case Qt.Key_0:
                canvas.resetZoomAndPan()
                event.accepted = true
                break
            case Qt.Key_L:
                canvas.loupeVisible = !canvas.loupeVisible
                event.accepted = true
                break
            case Qt.Key_QuoteLeft:
                canvas.loupeFrozen = !canvas.loupeFrozen
                event.accepted = true
                break
            // Shift+Tab is often delivered as Key_Backtab (not Tab+Shift) on Windows/Qt.
            case Qt.Key_Backtab:
                backend.select_prev()
                event.accepted = true; break
            case Qt.Key_Tab:
                if (event.modifiers & Qt.ShiftModifier) backend.select_prev()
                else backend.select_next()
                event.accepted = true; break
            case Qt.Key_Backspace:
                if (canvas.pendingInput.length > 0) {
                    canvas.pendingInput = canvas.pendingInput.slice(0, -1)
                    event.accepted = true
                }
                break
            case Qt.Key_Return:
            case Qt.Key_Enter:
                if (canvas.pendingInput.length > 0) {
                    const tid2 = parseInt(canvas.pendingInput, 10)
                    if (!isNaN(tid2) && tid2 > 0) {
                        if (backend.mode === "track"
                                && backend.segStart >= 0 && backend.segEnd >= 0) {
                            backend.bulk_assign_segment(tid2)
                        } else if (hasSelection) {
                            backend.assign_to_selection(tid2)
                        }
                    }
                    canvas.pendingInput = ""
                    event.accepted = true
                }
                break
            case Qt.Key_Delete:
                if (hasSelection) {
                    if (event.modifiers & Qt.ShiftModifier) {
                        // Shift+Del removes the selected synthetic detection
                        // entirely (and any annotation on it). Backend no-ops
                        // for real dets — they're not ours to delete.
                        backend.delete_selected_synthetic()
                    } else {
                        backend.clear_selection_annotation()
                    }
                    event.accepted = true
                }
                break
            case Qt.Key_Escape:
                if (canvas.pendingInput.length > 0) canvas.pendingInput = ""
                else if (backend.cropMode) backend.toggle_crop_mode()
                else if (backend.segStart >= 0) backend.clear_segment()
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
            case Qt.Key_T:
                // Toggle Frame <-> Track Mode.
                backend.toggle_track_mode()
                event.accepted = true
                break
            case Qt.Key_C:
                if ((event.modifiers & Qt.ControlModifier)
                        && (event.modifiers & Qt.ShiftModifier)) {
                    // Ctrl+Shift+C — toggle segment-crop mode (track mode only).
                    if (backend.mode === "track") {
                        backend.toggle_crop_mode()
                        event.accepted = true
                    }
                } else if (backend.mode === "track" && backend.focusedTrackId > 0) {
                    // C alone — confirm / lock the focused track.
                    backend.toggle_confirm_track(backend.focusedTrackId)
                    event.accepted = true
                }
                break
            case Qt.Key_Up:
                if (backend.mode === "track") {
                    backend.cycle_focused_track(-1)
                    event.accepted = true
                }
                break
            case Qt.Key_Down:
                if (backend.mode === "track") {
                    backend.cycle_focused_track(+1)
                    event.accepted = true
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
        height: 420
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

            // Mode indicator (FRAME / TRACK #N [confirmed]).
            Text {
                anchors.verticalCenter: parent.verticalCenter
                color: backend.cropMode ? "#a6e3a1"
                       : (backend.mode === "track" ? "#cba6f7" : "#a6adc8")
                font.family: "JetBrains Mono, Consolas, Courier New"
                font.pixelSize: 13
                font.bold: true
                text: {
                    if (backend.mode !== "track") return "MODE: FRAME"
                    let s = "MODE: TRACK #" + backend.focusedTrackId
                    if (backend.cropMode) {
                        s += "  CROP"
                        if (backend.segStart >= 0) {
                            s += " [" + (backend.segStart + 1)
                            if (backend.segEnd >= 0) s += "→" + (backend.segEnd + 1)
                            s += "]"
                        } else {
                            s += " — click start point"
                        }
                    }
                    return s
                }
            }

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
                    if (sel === -1) return base
                    // Synthetic dets have negative idx (-2, -3, ...); show "synth" label.
                    return base + "    selected: " + (sel < 0 ? "synth " + sel : "det #" + sel)
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
            color: backend.selectedDetIdx === -1 ? "#6c7086" : "#cba6f7"
            font.family: "Segoe UI, sans-serif"
            font.pixelSize: 12
            text: backend.selectedDetIdx === -1
                ? "click a fly to start"
                : "type digits → Enter to commit"
        }
    }
}
