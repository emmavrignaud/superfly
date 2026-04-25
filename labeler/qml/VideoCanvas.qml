import QtQuick
import QtQuick.Controls

Item {
    id: root

    // detections list: array of dicts from backend.detections_for_frame()
    property var detections: []
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
        root.detections = backend.detections_for_frame(backend.currentFrame)
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
        source: "image://videoframes/" + backend.currentFrame + "?v=" + backend.frameTick
        cache: false
        fillMode: Image.Stretch
        smooth: true
        asynchronous: false
    }

    // Click capture: hit-test against bboxes. Empty-space click deselects.
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
            ctx.lineWidth = 1.5

            const dets = root.detections
            const r = 7
            const tickInner = r * 1.3
            const tickOuter = r * 2.1
            const selIdx = backend.selectedDetIdx

            // Optional bbox layer (drawn first, behind markers)
            if (root.showBboxes) {
                ctx.lineWidth = 1.0
                for (let i = 0; i < dets.length; ++i) {
                    const d = dets[i]
                    ctx.strokeStyle = d.color
                    ctx.globalAlpha = 0.35
                    ctx.strokeRect(d.x1 * sx, d.y1 * sx,
                                   (d.x2 - d.x1) * sx, (d.y2 - d.y1) * sx)
                }
                ctx.globalAlpha = 1.0
                ctx.lineWidth = 1.5
            }

            // Markers
            for (let i = 0; i < dets.length; ++i) {
                const d = dets[i]
                const cx = d.x * sx
                const cy = d.y * sx
                ctx.strokeStyle = d.color
                ctx.fillStyle = d.color

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

                // Selection ring (white, 2px) on top
                if (d.det_idx === selIdx) {
                    ctx.strokeStyle = "#ffffff"
                    ctx.lineWidth = 2.0
                    ctx.beginPath()
                    ctx.arc(cx, cy, r + 4, 0, 2 * Math.PI)
                    ctx.stroke()
                    ctx.lineWidth = 1.5
                }
            }
        }

        Connections {
            target: backend
            function onFrameChanged(f) { root.refreshDetections() }
            function onAnnotationsChanged() { root.refreshDetections() }
            function onSelectionChanged() {
                root.pendingInput = ""
                overlay.requestPaint()
            }
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
        border.color: "#cba6f7"
        border.width: 1
        radius: 4
        width: bubbleText.implicitWidth + 16
        height: bubbleText.implicitHeight + 10

        // Position above the selected detection's centroid, clamped to canvas.
        property var sel: root.selectedDet()
        x: {
            if (!sel) return 0
            const cx = videoImg.x + sel.x * root.fitScale
            const px = cx - width / 2
            return Math.max(videoImg.x, Math.min(px, videoImg.x + videoImg.width - width))
        }
        y: {
            if (!sel) return 0
            const cy = videoImg.y + sel.y * root.fitScale
            // 18px above the marker; flip below if too high
            const desired = cy - 18 - height
            return desired < videoImg.y ? cy + 18 : desired
        }

        Text {
            id: bubbleText
            anchors.centerIn: parent
            color: "#cdd6f4"
            font.family: "JetBrains Mono, Consolas, Courier New"
            font.pixelSize: 12
            text: {
                const sel = inputBubble.sel
                if (!sel) return ""
                const cur = sel.track_id > 0 ? ("→" + sel.track_id) : "(unset)"
                const buf = root.pendingInput.length > 0 ? root.pendingInput : "_"
                return "ID " + buf + "   " + cur
            }
        }
    }
}
