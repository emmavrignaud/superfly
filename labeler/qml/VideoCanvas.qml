import QtQuick
import QtQuick.Controls

Item {
    id: root

    property var detections: []
    property var trajectory: []
    property bool showBboxes: false
    property string pendingInput: ""

    signal commitRequested()

    // Fit-to-view scale (letterboxed into this Item).
    readonly property real fitScale: Math.min(
        width / Math.max(1, backend.videoWidth),
        height / Math.max(1, backend.videoHeight)
    )

    // User zoom on top of fit (1 = fit, >1 = magnify). Wheel / +/- adjust this.
    property real zoomLevel: 1.0

    readonly property real screenScale: fitScale * zoomLevel
    readonly property real contentVideoW: backend.videoWidth * screenScale
    readonly property real contentVideoH: backend.videoHeight * screenScale

    property bool loupeVisible: true
    property bool loupeFrozen: false
    readonly property real loupeMag: 2.5
    property real loupeVX: backend.videoWidth > 0 ? backend.videoWidth / 2 : 0
    property real loupeVY: backend.videoHeight > 0 ? backend.videoHeight / 2 : 0
    // Loupe window top-left in VideoCanvas coords (follows hover).
    property real loupePanelX: 24
    property real loupePanelY: 24

    function clampZoom(z) {
        return Math.max(1.0, Math.min(8.0, z))
    }

    function zoomTowardViewportPoint(newZL, ax, ay) {
        const z = clampZoom(newZL)
        if (Math.abs(z - root.zoomLevel) < 1e-5)
            return
        const oldSL = root.screenScale
        const ox = (flick.contentWidth - root.contentVideoW) / 2
        const oy = (flick.contentHeight - root.contentVideoH) / 2
        const vidLocalX = flick.contentX + ax - ox
        const vidLocalY = flick.contentY + ay - oy
        zoomReflowTimer.vx = vidLocalX / oldSL
        zoomReflowTimer.vy = vidLocalY / oldSL
        zoomReflowTimer.ax = ax
        zoomReflowTimer.ay = ay
        root.zoomLevel = z
        zoomReflowTimer.restart()
    }

    function zoomIn() {
        zoomTowardViewportPoint(root.zoomLevel * 1.15, flick.width / 2, flick.height / 2)
    }

    function zoomOut() {
        zoomTowardViewportPoint(root.zoomLevel / 1.15, flick.width / 2, flick.height / 2)
    }

    function resetZoomAndPan() {
        root.zoomLevel = 1.0
        flick.contentX = 0
        flick.contentY = 0
        flick.returnToBounds()
    }

    Timer {
        id: zoomReflowTimer
        interval: 0
        repeat: false
        property real vx: 0
        property real vy: 0
        property real ax: 0
        property real ay: 0
        onTriggered: {
            const newOx = (flick.contentWidth - root.contentVideoW) / 2
            const newOy = (flick.contentHeight - root.contentVideoH) / 2
            const sl = root.screenScale
            flick.contentX = vx * sl + newOx - ax
            flick.contentY = vy * sl + newOy - ay
            flick.returnToBounds()
        }
    }

    function selectedDet() {
        const idx = backend.selectedDetIdx
        if (idx === -1)
            return null
        for (let i = 0; i < detections.length; ++i) {
            if (detections[i].det_idx === idx)
                return detections[i]
        }
        return null
    }

    function refreshDetections() {
        root.detections = backend.detections_for_frame(backend.displayFrame)
        overlay.requestPaint()
    }

    function refreshTrajectory() {
        const newTraj = (backend.mode === "track" && backend.focusedTrackId > 0)
                        ? backend.track_positions(backend.focusedTrackId)
                        : []
        // Only reassign + repaint if something actually changed (length or
        // any coordinate differs). This avoids flicker on unrelated signals.
        // But always repaint if the arrays differ in length.
        root.trajectory = newTraj
        overlay.requestPaint()
    }

    Rectangle {
        anchors.fill: parent
        color: "#181825"
    }

    Flickable {
        id: flick
        anchors.fill: parent
        clip: true
        boundsBehavior: Flickable.StopAtBounds
        boundsMovement: Flickable.StopAtBounds
        contentWidth: Math.max(width, root.contentVideoW)
        contentHeight: Math.max(height, root.contentVideoH)

        Item {
            width: flick.contentWidth
            height: flick.contentHeight

            Image {
                id: videoImg
                x: (parent.width - root.contentVideoW) / 2
                y: (parent.height - root.contentVideoH) / 2
                width: root.contentVideoW
                height: root.contentVideoH
                source: "image://videoframes/" + backend.displayFrame + "?v=" + backend.frameTick
                cache: false
                fillMode: Image.Stretch
                smooth: true
                asynchronous: false
            }

            Canvas {
                id: overlay
                x: videoImg.x
                y: videoImg.y
                width: videoImg.width
                height: videoImg.height
                renderTarget: Canvas.Image
                antialiasing: true

                readonly property real sx: root.screenScale

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
                    const showSelRing = !backend.isPlaying

                    if (trackMode && root.trajectory.length > 1) {
                        let trajColor = "#cba6f7"
                        for (let i = 0; i < dets.length; ++i) {
                            if (dets[i].track_id === focusId) {
                                trajColor = dets[i].color
                                break
                            }
                        }

                        const segS = backend.segStart
                        const segE = backend.segEnd
                        const hasSegRange = segS >= 0 && segE >= 0
                        const segLo = hasSegRange ? Math.min(segS, segE) : -1
                        const segHi = hasSegRange ? Math.max(segS, segE) : -1

                        // Draw trajectory segment-by-segment so highlighted
                        // range can use a distinct colour.
                        ctx.lineWidth = 1.5
                        for (let i = 0; i < root.trajectory.length - 1; ++i) {
                            const f1 = root.trajectory[i].frame
                            const f2 = root.trajectory[i + 1].frame
                            const inSeg = hasSegRange &&
                                          f1 >= segLo && f2 <= segHi
                            ctx.strokeStyle = inSeg ? "#a6e3a1" : trajColor
                            ctx.globalAlpha  = inSeg ? 0.9 : 0.6
                            ctx.beginPath()
                            ctx.moveTo(root.trajectory[i].x * sx,
                                       root.trajectory[i].y * sx)
                            ctx.lineTo(root.trajectory[i + 1].x * sx,
                                       root.trajectory[i + 1].y * sx)
                            ctx.stroke()
                        }

                        // Segment endpoint markers (green filled circles).
                        if (segS >= 0) {
                            for (let i = 0; i < root.trajectory.length; ++i) {
                                const f = root.trajectory[i].frame
                                if (f === segS || (segE >= 0 && f === segE)) {
                                    ctx.fillStyle = "#a6e3a1"
                                    ctx.globalAlpha = 1.0
                                    ctx.beginPath()
                                    ctx.arc(root.trajectory[i].x * sx,
                                            root.trajectory[i].y * sx,
                                            5, 0, 2 * Math.PI)
                                    ctx.fill()
                                }
                            }
                        }

                        ctx.globalAlpha = 1.0
                    }

                    if (root.showBboxes) {
                        ctx.lineWidth = 1.0
                        for (let i = 0; i < dets.length; ++i) {
                            const d = dets[i]
                            if (d.track_id < 0)
                                continue
                            ctx.strokeStyle = d.color
                            ctx.globalAlpha = 0.35
                            ctx.strokeRect(d.x1 * sx, d.y1 * sx,
                                           (d.x2 - d.x1) * sx, (d.y2 - d.y1) * sx)
                        }
                        ctx.globalAlpha = 1.0
                    }

                    for (let i = 0; i < dets.length; ++i) {
                        const d = dets[i]
                        const cx = d.x * sx
                        const cy = d.y * sx
                        const isUnannotated = d.track_id < 0
                        ctx.globalAlpha = (trackMode && !isUnannotated && d.track_id !== focusId) ? 0.25 : 1.0

                        if (isUnannotated) {
                            ctx.strokeStyle = d.color
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
                            if (d.filled)
                                ctx.fill()
                            else
                                ctx.stroke()

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

                        if (showSelRing && d.det_idx === selIdx) {
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
                    function onDisplayFrameChanged() {
                        root.refreshDetections()
                    }
                    function onAnnotationsChanged() {
                        root.refreshDetections()
                        root.refreshTrajectory()
                    }
                    function onSelectionChanged() {
                        root.pendingInput = ""
                        overlay.requestPaint()
                    }
                    function onModeChanged() {
                        root.refreshTrajectory()
                    }
                    function onFocusedTrackChanged() {
                        root.refreshTrajectory()
                    }
                    function onIsPlayingChanged() {
                        overlay.requestPaint()
                    }
                    function onSegmentChanged() {
                        overlay.requestPaint()
                    }
                    function onCropModeChanged() {
                        overlay.requestPaint()
                    }
                }

                onWidthChanged: requestPaint()
                onHeightChanged: requestPaint()
                Connections {
                    target: root
                    function onShowBboxesChanged() {
                        overlay.requestPaint()
                    }
                    function onDetectionsChanged() {
                        overlay.requestPaint()
                    }
                    function onZoomLevelChanged() {
                        overlay.requestPaint()
                    }
                }

                Component.onCompleted: root.refreshDetections()
            }

            MouseArea {
                id: videoMouse
                anchors.fill: videoImg
                hoverEnabled: true
                acceptedButtons: Qt.LeftButton
                preventStealing: false
                cursorShape: backend.cropMode ? Qt.CrossCursor : Qt.ArrowCursor

                // Drag tracking state (managed purely in QML; backend notified
                // only once a real drag exceeds the threshold).
                property bool _dragActive: false
                property real _pressX: 0
                property real _pressY: 0
                // Threshold in screen-pixels before we commit to a drag.
                readonly property real _dragThreshold: 5

                onPressed: (mouse) => {
                    _pressX = mouse.x
                    _pressY = mouse.y
                    _dragActive = false
                }

                onPositionChanged: (mouse) => {
                    if (!root.loupeFrozen) {
                        root.loupeVX = mouse.x / root.screenScale
                        root.loupeVY = mouse.y / root.screenScale
                        const pr = videoMouse.mapToItem(root, mouse.x, mouse.y)
                        root.loupePanelX = pr.x + 8
                        root.loupePanelY = pr.y + 8
                    }
                    if (mouse.buttons & Qt.LeftButton) {
                        const dx = mouse.x - _pressX
                        const dy = mouse.y - _pressY
                        if (!_dragActive && Math.sqrt(dx*dx + dy*dy) > _dragThreshold) {
                            const pvx = _pressX / root.screenScale
                            const pvy = _pressY / root.screenScale
                            if (backend.start_drag(pvx, pvy))
                                _dragActive = true
                        }
                        if (_dragActive)
                            backend.update_drag(mouse.x / root.screenScale,
                                                mouse.y / root.screenScale)
                    }
                }

                onReleased: (mouse) => {
                    const x_video = mouse.x / root.screenScale
                    const y_video = mouse.y / root.screenScale

                    if (_dragActive) {
                        backend.finish_drag(x_video, y_video)
                        _dragActive = false
                        return
                    }

                    // It was a click — handle normally.
                    if (mouse.modifiers & Qt.ShiftModifier) {
                        backend.create_synthetic_at(x_video, y_video)
                    } else if (backend.cropMode) {
                        // Crop mode: every click sets a segment endpoint.
                        // Use a generous threshold (15 px) since that's the
                        // only thing clicks do in this mode.
                        const nearFrame = backend.nearest_trajectory_frame(
                                              x_video, y_video, 15)
                        if (nearFrame >= 0)
                            backend.add_segment_point(nearFrame)
                    } else {
                        backend.select_at_video_point(x_video, y_video)
                    }
                }
                // Wheel over video: zoom when fitted or when Ctrl/Alt held (pinch emulation).
                // When zoomed without modifiers, we must pan here — MouseArea sits above the
                // Flickable content, so wheel never reaches Flickable for scrolling.
                onWheel: (wheel) => {
                    let dx = wheel.angleDelta.x
                    let dy = wheel.angleDelta.y
                    if (wheel.pixelDelta !== undefined && Math.abs(dy) < 1 && Math.abs(dx) < 1) {
                        dx = wheel.pixelDelta.x
                        dy = wheel.pixelDelta.y
                    }
                    if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5)
                        return

                    const ctrl = (wheel.modifiers & Qt.ControlModifier) !== 0
                    const alt = (wheel.modifiers & Qt.AltModifier) !== 0
                    const fitted = root.zoomLevel <= 1.001

                    if (ctrl || alt || fitted) {
                        wheel.accepted = true
                        const zDir = Math.abs(dy) >= Math.abs(dx) ? dy : dx
                        const factor = zDir > 0 ? 1.1 : 1.0 / 1.1
                        const p = videoMouse.mapToItem(flick, wheel.x, wheel.y)
                        root.zoomTowardViewportPoint(root.zoomLevel * factor, p.x, p.y)
                        return
                    }

                    wheel.accepted = true
                    flick.contentX -= dx / 8
                    flick.contentY -= dy / 8
                    flick.returnToBounds()
                }
            }
        }

        onContentWidthChanged: overlay.requestPaint()
        onContentHeightChanged: overlay.requestPaint()
    }

    Connections {
        target: backend
        function onVideoSizeChanged() {
            overlay.requestPaint()
        }
        // Also refresh trajectory at root level — belt-and-suspenders against
        // any case where the nested Canvas Connections block misses the signal.
        function onAnnotationsChanged() {
            root.refreshTrajectory()
        }
        function onTracksChanged() {
            root.refreshTrajectory()
        }
    }

    // Floating magnifier beside cursor (~40% of original 200px — 60% smaller).
    Rectangle {
        id: loupe
        visible: root.loupeVisible && backend.videoWidth > 0
        width: 80
        height: 80
        radius: 4
        color: "#11111b"
        border.color: "#cba6f7"
        border.width: 1
        x: Math.min(Math.max(8, root.loupePanelX), root.width - width - 8)
        y: Math.min(Math.max(8, root.loupePanelY), root.height - height - 8)
        clip: true
        z: 10

        readonly property real loupePixelScale: root.screenScale * root.loupeMag

        Image {
            id: loupeImg
            width: backend.videoWidth * loupe.loupePixelScale
            height: backend.videoHeight * loupe.loupePixelScale
            source: "image://videoframes/" + backend.displayFrame + "?v=" + backend.frameTick
            cache: false
            fillMode: Image.Stretch
            smooth: true
            asynchronous: false
            x: loupe.width / 2 - root.loupeVX * loupe.loupePixelScale
            y: loupe.height / 2 - root.loupeVY * loupe.loupePixelScale
        }

        Text {
            anchors.left: parent.left
            anchors.bottom: parent.bottom
            anchors.margins: 3
            text: root.loupeFrozen ? "off" : "on"
            color: "#6c7086"
            font.family: "JetBrains Mono, Consolas, Courier New"
            font.pixelSize: 7
        }
    }

    Rectangle {
        id: inputBubble
        readonly property bool segmentReady: backend.segStart >= 0 && backend.segEnd >= 0
        visible: backend.selectedDetIdx !== -1 || segmentReady
        color: "#313244"
        border.color: wouldDuplicate ? "#ef4444" : (segmentReady ? "#a6e3a1" : "#cba6f7")
        border.width: 1
        radius: 4
        width: Math.max(bubbleText.implicitWidth, warnText.visible ? warnText.implicitWidth : 0) + 16
        height: bubbleText.implicitHeight + (warnText.visible ? warnText.implicitHeight + 4 : 0) + 10
        z: 20

        property var sel: root.selectedDet()

        readonly property bool wouldDuplicate: {
            const buf = root.pendingInput
            if (buf.length === 0)
                return false
            const tid = parseInt(buf, 10)
            if (isNaN(tid) || tid <= 0)
                return false
            return backend.would_duplicate_in_current_frame(tid)
        }

        // Centre of video canvas when no detection is selected (segment mode).
        x: {
            if (sel) {
                const ox = (flick.contentWidth - root.contentVideoW) / 2
                const cx = ox + sel.x * root.screenScale - flick.contentX
                return Math.max(0, Math.min(cx - width / 2, root.width - width))
            }
            return Math.max(0, (root.width - width) / 2)
        }
        y: {
            if (sel) {
                const oy = (flick.contentHeight - root.contentVideoH) / 2
                const cy = oy + sel.y * root.screenScale - flick.contentY
                const desired = cy - 18 - height
                return desired < 0 ? cy + 18 : desired
            }
            return 20
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
                    // Segment-crop mode: no detection selected, but a range is ready.
                    if (!inputBubble.sel && inputBubble.segmentReady) {
                        const lo = backend.segStart + 1
                        const hi = backend.segEnd + 1
                        if (root.pendingInput.length === 0)
                            return "seg [" + lo + "→" + hi + "]   ID _   next free: " + backend.nextFreeTrackId
                        return "seg [" + lo + "→" + hi + "]   ID " + root.pendingInput
                    }
                    const sel = inputBubble.sel
                    if (!sel)
                        return ""
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
