import QtQuick
import QtQuick.Controls

Rectangle {
    id: root
    color: "#181825"
    border.color: "#313244"
    border.width: 1

    property var tracks: []

    function refresh() { root.tracks = backend.track_summary() }

    Connections {
        target: backend
        function onTracksChanged() { root.refresh() }
        function onFrameChanged(_) { /* counts are global, no refresh needed */ }
    }
    Component.onCompleted: refresh()

    Column {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 8

        Text {
            text: "TRACKS"
            color: "#a6adc8"
            font.family: "Segoe UI, sans-serif"
            font.pixelSize: 11
            font.bold: true
        }

        Text {
            visible: root.tracks.length === 0
            text: "(none yet)"
            color: "#6c7086"
            font.family: "Segoe UI, sans-serif"
            font.pixelSize: 12
        }

        ListView {
            width: parent.width
            height: parent.height - 40
            clip: true
            model: root.tracks
            spacing: 4

            delegate: Item {
                width: ListView.view.width
                height: 22
                Row {
                    anchors.fill: parent
                    spacing: 8

                    Rectangle {
                        width: 14; height: 14
                        radius: 3
                        color: modelData.color
                        anchors.verticalCenter: parent.verticalCenter
                    }
                    Text {
                        text: "#" + modelData.track_id
                        color: "#cdd6f4"
                        font.family: "JetBrains Mono, Consolas, Courier New"
                        font.pixelSize: 12
                        anchors.verticalCenter: parent.verticalCenter
                        width: 36
                    }
                    Text {
                        text: modelData.human_count + " / " + modelData.count
                        color: "#a6adc8"
                        font.family: "JetBrains Mono, Consolas, Courier New"
                        font.pixelSize: 11
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }
            }
        }
    }
}
