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
        function onConfirmedTracksChanged() { root.refresh() }
        function onFrameChanged(_) { /* counts are global, no refresh needed */ }
    }
    Component.onCompleted: refresh()

    Column {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 8

        Row {
            spacing: 8
            Text {
                text: "TRACKS"
                color: "#a6adc8"
                font.family: "Segoe UI, sans-serif"
                font.pixelSize: 11
                font.bold: true
                anchors.verticalCenter: parent.verticalCenter
            }
            Text {
                text: "next free: " + backend.nextFreeTrackId
                color: "#a6e3a1"
                font.family: "JetBrains Mono, Consolas, Courier New"
                font.pixelSize: 10
                anchors.verticalCenter: parent.verticalCenter
            }
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
                property bool isFocused: backend.mode === "track"
                                         && modelData.track_id === backend.focusedTrackId

                Rectangle {
                    anchors.fill: parent
                    color: modelData.confirmed ? "#1a2e1a"
                           : (parent.isFocused ? "#313244" : "transparent")
                    border.color: modelData.confirmed ? "#a6e3a1"
                                  : (parent.isFocused ? "#cba6f7" : "transparent")
                    border.width: (modelData.confirmed || parent.isFocused) ? 1 : 0
                    radius: 3
                }

                Row {
                    anchors.fill: parent
                    anchors.leftMargin: 4
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

                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: backend.set_focused_track(modelData.track_id)
                }
            }
        }
    }
}
