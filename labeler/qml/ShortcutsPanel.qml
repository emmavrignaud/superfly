import QtQuick
import QtQuick.Controls

Rectangle {
    id: root
    color: "#181825"
    border.color: "#313244"
    border.width: 1

    // Each row: ["key combo", "action"]. Grouped by section header (an empty
    // key string with a non-empty action).
    readonly property var rows: [
        ["",            "NAVIGATION"],
        ["← / →",       "prev / next frame"],
        ["PgUp / PgDn", "± 10 frames"],
        ["Home / End",  "first / last frame"],
        ["",            "SELECTION"],
        ["click",       "select fly (inside bbox, +5px tolerance)"],
        ["⇧+click",     "create synthetic detection at click (detector miss)"],
        ["Tab / ⇧Tab",  "cycle dets (overlap pile after click, else whole frame)"],
        ["Esc",         "clear input / deselect / end follow trail"],
        ["",            "ANNOTATE"],
        ["digits",      "type track ID"],
        ["Enter",       "commit ID (auto-prefilled when following a fly)"],
        ["Backspace",   "erase last digit"],
        ["Del",         "clear annotation"],
        ["⇧+Del",       "delete selected synthetic detection (real dets are immune)"],
        ["Ctrl+Z",      "undo any action (assign / clear / synth create / resize / delete)"],
        ["⇧+arrows",    "resize selected synthetic bbox"],
        ["",            "MODE"],
        ["T",           "toggle frame/track mode"],
        ["↑ / ↓",       "cycle focused track (track mode)"],
        ["click row",   "focus track (TRACKS panel)"],
        ["",            "VIEW"],
        ["Space",       "play/pause ±1s loop"],
        ["B",           "toggle bboxes"],
        ["pinch",       "zoom (trackpad / touch)"],
        ["two-finger drag", "pan when zoomed / wheel scroll"],
        ["+ / −",       "zoom in / out (keyboard)"],
        ["0",           "reset zoom & pan"],
        ["L",           "toggle magnifier panel"],
        ["`",           "freeze / thaw magnifier center"],
        ["",            "FILE"],
        ["Ctrl+S",      "save session"],
        ["Ctrl+E",      "export GT csv"]
    ]

    // Header is pinned to the top so users always see "SHORTCUTS" even after
    // scrolling. The list itself is what scrolls.
    Text {
        id: header
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.topMargin: 10
        anchors.leftMargin: 10
        text: "SHORTCUTS"
        color: "#a6adc8"
        font.family: "Segoe UI, sans-serif"
        font.pixelSize: 11
        font.bold: true
    }

    ScrollView {
        id: scroller
        anchors.top: header.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.topMargin: 6
        anchors.leftMargin: 10
        anchors.rightMargin: 10
        anchors.bottomMargin: 10
        clip: true

        // Wheel-scrolling still works; bars are hidden for a cleaner look.
        ScrollBar.vertical.policy: ScrollBar.AlwaysOff
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        contentWidth: availableWidth

        Column {
            width: scroller.availableWidth
            spacing: 2

            Repeater {
                model: root.rows
                delegate: Item {
                    width: parent.width
                    // Headers are fixed-height; rows grow to fit wrapped action text.
                    height: isHeader ? 20 : Math.max(16, actionText.implicitHeight + 4)

                    property bool isHeader: modelData[0] === ""

                    Text {
                        visible: parent.isHeader
                        anchors.left: parent.left
                        anchors.bottom: parent.bottom
                        text: modelData[1]
                        color: "#cba6f7"
                        font.family: "Segoe UI, sans-serif"
                        font.pixelSize: 10
                        font.bold: true
                    }
                    Text {
                        visible: !parent.isHeader
                        anchors.left: parent.left
                        anchors.top: parent.top
                        anchors.topMargin: 1
                        width: 86
                        text: modelData[0]
                        color: "#cdd6f4"
                        font.family: "JetBrains Mono, Consolas, Courier New"
                        font.pixelSize: 11
                    }
                    Text {
                        id: actionText
                        visible: !parent.isHeader
                        anchors.left: parent.left
                        anchors.leftMargin: 88
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.topMargin: 1
                        text: modelData[1]
                        color: "#a6adc8"
                        font.family: "Segoe UI, sans-serif"
                        font.pixelSize: 11
                        wrapMode: Text.WordWrap
                    }
                }
            }
        }
    }
}
