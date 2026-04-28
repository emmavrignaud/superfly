import QtQuick
import QtQuick.Controls

Item {
    id: root

    // detections list: array of dicts from backend.detections_for_frame()
    property var detections: []
    // trajectory: array of {frame, x, y} for the focused track (Track Mode only)
    property var trajectory: []
    property bool showBboxes: false

    // Pending track-id buffer typed by the user while a detection is selected.
    // Cleared on selection change, Esc, or commit.
    property string pendingInput: ""

    signal commitRequested()  // wired from main.qml's key handler

    readonly property real fitScale: Math.min(
        width  / Math.max(1, backend.videoWidth),
        height / Math.max(1, backend.videoHeight)
    )
    readonly property real videoDrawW: backend.videoWidth  * fitScale
    readonly property real videoDrawH: backend.videoHeight * fitScale

    // Find the detection record matching the currently-selected det_idx.
    function selectedDet() {
        const idx = backend.selectedDetIdx
        if (idx < 0) return null
        for (let i = 0; i < detections.length; ++i) {
            if (detections[i].det_idx === idx) return detections[i]
        }
        return null
    }

    function refreshDetections() {
        root.detections = backend.detections_for_frame(backend.displayFrame)
        overlay.requestPaint()
    }

    function refreshTrajectory() {
        if (backend.mode === "track" && backend.focusedTrackId > 0) {
            root.trajectory = backend.track_positions(backend.focusedTrackId)
        } else {
            root.trajectory = []
        }
        overlay.requestPaint()
    }

    // Background fill (Catppuccin mantle)
    Rectangle {
        anchors.fill: parent
        color: "#181825"
    }

    // The video frame, centered.
    Image {
        id: videoImg
        anchors.centerIn: parent
        width:  root.videoDrawW
        height: root.videoDrawH
        source: "image://videoframes/" + backend.displayFrame + "?v=" + backend.frameTick
        cache: false
        fillMode: Image.Stretch
        smooth: true
        asynchronous: false
    }

    // Click capture: hit-test against bboxes. Empty-space click deselects.
    // Shift+click on empty space: create a synthetic detection there
    // (for flies the detector missed entirely).
    MouseArea {
        anchors.fill: videoImg
        acceptedButtons: Qt.LeftButton
        onClicked: (mouse) => {
            const sx = root.fitScale
            const x_video = mouse.x / sx
            const y_video = mouse.y / sx
            const idx = backend.hit_test_bbox(x_video, y_video)
            if (idx >= 0) {
                backend.select(idx)
            } else if (mouse.modifiers & Qt.ShiftModifier) {
                backend.create_synthetic_at(x_video, y_video)
            } else {
                backend.clear_selection()
            }
        }
    }

    // Detection markers overlay, sized exactly to the video image.
    Canvas {
        id: overlay
        anchors.fill: videoImg
        renderTarget: Canvas.Image
        antialiasing: true

        readonly property real sx: root.fitScale

        onPaint: {
            const ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)

            const dets = root.detections
            const r = 7
            const tickInner = r * 1.3
            const tickOuter = r * 2.1
            const selIdx = backend.selectedDetIdx
            const trackMode = backend.mode === "track"
            const focusId = backend.focusedTrackId

            // Trajectory polyline for focused track (drawn first, behind markers)
            if (trackMode && root.trajectory.length > 1) {
                ctx.lineWidth = 1.5
                ctx.globalAlpha = 0.6
                // Pull color from any detection that has the focused track,
                // else fall back to mauve.
                let trajColor = "#cba6f7"
                for (let i = 0; i < dets.length; ++i) {
                    if (dets[i].track_id === focusId) { trajColor = dets[i].color; break }
                }
                ctx.strokeStyle = trajColor
                ctx.beginPath()
                ctx.moveTo(root.trajectory[0].x * sx, root.trajectory[0].y * sx)
                for (let i = 1; i < root.trajectory.length; ++i) {
                    ctx.lineTo(root.trajectory[i].x * sx, root.trajectory[i].y * sx)
                }
                ctx.stroke()
                ctx.globalAlpha = 1.0
            }

            // Optional annotated-bbox layer (B toggle). Unannotated detections
            // get their own always-on red bbox below; this layer only adds
            // bboxes for the annotated ones.
            if (root.showBboxes) {
                ctx.lineWidth = 1.0
                for (let i = 0; i < dets.length; ++i) {
                    const d = dets[i]
                    if (d.track_id < 0) continue
                    ctx.strokeStyle = d.color
                    ctx.globalAlpha = 0.35
                    ctx.strokeRect(d.x1 * sx, d.y1 * sx,
                                   (d.x2 - d.x1) * sx, (d.y2 - d.y1) * sx)
                }
                ctx.globalAlpha = 1.0
            }

            // Markers
            for (let i = 0; i < dets.length; ++i) {
                const d = dets[i]
                const cx = d.x * sx
                const cy = d.y * sx
                const isUnannotated = d.track_id < 0
                // In Track Mode, dim everything that isn't the focused track.
                // Unannotated red bboxes stay full-alpha (they're "needs attention").
                ctx.globalAlpha = (trackMode && !isUnannotated && d.track_id !== focusId) ? 0.25 : 1.0

                if (isUnannotated) {
                    // Distinct shape: red bbox + tiny hollow centroid dot.
                    // Dashed border if synthetic (human-placed for detector miss).
                    ctx.strokeStyle = d.color   // red
                    ctx.fillStyle = d.color
                    ctx.lineWidth = 2.0
                    ctx.setLineDash(d.is_synthetic ? [4, 3] : [])
                    ctx.strokeRect(d.x1 * sx, d.y1 * sx,
                                   (d.x2 - d.x1) * sx, (d.y2 - d.y1) * sx)
                    ctx.setLineDash([])
                    ctx.lineWidth = 1.5
                    ctx.beginPath()
                    ctx.arc(cx, cy, 3, 0, 2 * Math.PI)
                    ctx.stroke()
                } else {
                    // Annotated: 4 diagonal ticks + central circle (filled if human).
                    ctx.strokeStyle = d.color
                    ctx.fillStyle = d.color
                    ctx.lineWidth = 1.5
                    ctx.beginPath()
                    ctx.moveTo(cx + tickInner * 0.707, cy - tickInner * 0.707)
                    ctx.lineTo(cx + tickOuter * 0.707, cy - tickOuter * 0.707)
                    ctx.moveTo(cx - tickInner * 0.707, cy - tickInner * 0.707)
                    ctx.lineTo(cx - tickOuter * 0.707, cy - tickOuter * 0.707)
                    ctx.moveTo(cx + tickInner * 0.707, cy + tickInner * 0.707)
                    ctx.lineTo(cx + tickOuter * 0.707, cy + tickOuter * 0.707)
                    ctx.moveTo(cx - tickInner * 0.707, cy + tickInner * 0.707)
                    ctx.lineTo(cx - tickOuter * 0.707, cy + tickOuter * 0.707)
                    ctx.stroke()

                    ctx.beginPath()
                    ctx.arc(cx, cy, r, 0, 2 * Math.PI)
                    if (d.filled) ctx.fill(); else ctx.stroke()

                    // Annotated synthetic: dashed bbox outline so the
                    // origin (human-drawn) stays visible after labeling.
                    if (d.is_synthetic) {
                        ctx.lineWidth = 1.0
                        ctx.setLineDash([4, 3])
                        ctx.globalAlpha = 0.5
                        ctx.strokeRect(d.x1 * sx, d.y1 * sx,
                                       (d.x2 - d.x1) * sx, (d.y2 - d.y1) * sx)
                        ctx.globalAlpha = 1.0
                        ctx.setLineDash([])
                        ctx.lineWidth = 1.5
                    }
                }

                // Selection ring (white, 2px) on top of either shape.
                if (d.det_idx === selIdx) {
                    ctx.strokeStyle = "#ffffff"
                    ctx.lineWidth = 2.0
                    ctx.beginPath()
                    ctx.arc(cx, cy, r + 4, 0, 2 * Math.PI)
                    ctx.stroke()
                    ctx.lineWidth = 1.5
                }
            }
            ctx.globalAlpha = 1.0
        }

        Connections {
            target: backend
            // displayFrame fires for both static seeks and every playback tick.
            function onDisplayFrameChanged() { root.refreshDetections() }
            function onAnnotationsChanged() {
                root.refreshDetections()
                root.refreshTrajectory()
            }
            function onSelectionChanged() {
                root.pendingInput = ""
                overlay.requestPaint()
            }
            function onModeChanged() { root.refreshTrajectory() }
            function onFocusedTrackChanged() { root.refreshTrajectory() }
        }

        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
        Connections {
            target: root
            function onShowBboxesChanged() { overlay.requestPaint() }
            function onDetectionsChanged() { overlay.requestPaint() }
        }

        Component.onCompleted: root.refreshDetections()
    }

    // Floating input near the selected detection: shows "ID: ___"
    // Display-only; key handling for digits/Enter/Esc lives in main.qml.
    Rectangle {
        id: inputBubble
        visible: backend.selectedDetIdx >= 0
        color: "#313244"
        border.color: wouldDuplicate ? "#ef4444" : "#cba6f7"
        border.width: 1
        radius: 4
        width: Math.max(bubbleText.implicitWidth, warnText.visible ? warnText.implicitWidth : 0) + 16
        height: bubbleText.implicitHeight + (warnText.visible ? warnText.implicitHeight + 4 : 0) + 10

        property var sel: root.selectedDet()

        // True when the ID currently typed would put the same track on two
        // different detections in the current frame. Recomputes on every
        // pendingInput change because of the binding.
        readonly property bool wouldDuplicate: {
            const buf = root.pendingInput
            if (buf.length === 0) return false
            const tid = parseInt(buf, 10)
            if (isNaN(tid) || tid <= 0) return false
            return backend.would_duplicate_in_current_frame(tid)
        }

        x: {
            if (!sel) return 0
            const cx = videoImg.x + sel.x * root.fitScale
            const px = cx - width / 2
            return Math.max(videoImg.x, Math.min(px, videoImg.x + videoImg.width - width))
        }
        y: {
            if (!sel) return 0
            const cy = videoImg.y + sel.y * root.fitScale
            const desired = cy - 18 - height
            return desired < videoImg.y ? cy + 18 : desired
        }

        Column {
            anchors.centerIn: parent
            spacing: 2

            Text {
                id: bubbleText
                anchors.horizontalCenter: parent.horizontalCenter
                color: "#cdd6f4"
                font.family: "JetBrains Mono, Consolas, Courier New"
                font.pixelSize: 12
                text: {
                    const sel = inputBubble.sel
                    if (!sel) return ""
                    const cur = sel.track_id > 0 ? ("->" + sel.track_id) : "(unset)"
                    let line
                    if (root.pendingInput.length === 0) {
                        line = "ID _   " + cur + "   next free: " + backend.nextFreeTrackId
                    } else {
                        line = "ID " + root.pendingInput + "   " + cur
                    }
                    if (sel.is_synthetic) {
                        const w = Math.round(sel.x2 - sel.x1)
                        const h = Math.round(sel.y2 - sel.y1)
                        line += "   [synth " + w + "x" + h + "  shift+arrows to resize]"
                    }
                    return line
                }
            }

            Text {
                id: warnText
                anchors.horizontalCenter: parent.horizontalCenter
                visible: inputBubble.wouldDuplicate
                color: "#ef4444"
                font.family: "JetBrains Mono, Consolas, Courier New"
                font.pixelSize: 11
                text: "! same fly cannot be in two places at the same time"
            }
        }
    }
}
