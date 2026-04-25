import QtQuick

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
        ["click",       "select fly (inside bbox)"],
        ["Tab / ⇧Tab",  "cycle dets in frame"],
        ["Esc",         "clear input / deselect"],
        ["",            "ANNOTATE"],
        ["digits",      "type track ID"],
        ["Enter",       "commit ID"],
        ["Backspace",   "erase last digit"],
        ["Del",         "clear annotation"],
        ["Ctrl+Z",      "undo"],
        ["",            "VIEW"],
        ["B",           "toggle bboxes"]
    ]

    Column {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 2

        Text {
            text: "SHORTCUTS"
            color: "#a6adc8"
            font.family: "Segoe UI, sans-serif"
            font.pixelSize: 11
            font.bold: true
            bottomPadding: 4
        }

        Repeater {
            model: root.rows
            delegate: Item {
                width: parent.width
                height: isHeader ? 18 : 16

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
                    anchors.verticalCenter: parent.verticalCenter
                    width: 78
                    text: modelData[0]
                    color: "#cdd6f4"
                    font.family: "JetBrains Mono, Consolas, Courier New"
                    font.pixelSize: 11
                }
                Text {
                    visible: !parent.isHeader
                    anchors.left: parent.left
                    anchors.leftMargin: 80
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    text: modelData[1]
                    color: "#a6adc8"
                    font.family: "Segoe UI, sans-serif"
                    font.pixelSize: 11
                    elide: Text.ElideRight
                }
            }
        }
    }
}
