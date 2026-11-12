import AppKit
@preconcurrency import AVFoundation
import SwiftUI

enum HushTheme: String, CaseIterable, Identifiable {
    case graphite
    case ceramic
    case auto

    var id: String { rawValue }

    var label: String {
        switch self {
        case .graphite: "Graphite"
        case .ceramic: "Ceramic"
        case .auto: "Auto"
        }
    }
}

struct HushPalette {
    let isDark: Bool

    var background: Color {
        isDark ? Color(red: 0.078, green: 0.086, blue: 0.102) : Color(red: 0.969, green: 0.965, blue: 0.949)
    }

    var surface: Color {
        isDark ? Color(red: 0.118, green: 0.129, blue: 0.153) : Color(red: 1.0, green: 0.996, blue: 0.988)
    }

    var surfaceMuted: Color {
        isDark ? Color.white.opacity(0.06) : Color.black.opacity(0.04)
    }

    var border: Color {
        isDark ? Color.white.opacity(0.12) : Color.black.opacity(0.10)
    }

    var textPrimary: Color {
        isDark ? Color(red: 0.945, green: 0.953, blue: 0.969) : Color(red: 0.094, green: 0.094, blue: 0.106)
    }

    var textSecondary: Color {
        isDark ? Color.white.opacity(0.62) : Color.black.opacity(0.55)
    }

    var accent: Color {
        isDark ? Color(red: 0.961, green: 0.620, blue: 0.043) : Color(red: 0.851, green: 0.365, blue: 0.141)
    }

    var activeBar: Color {
        isDark ? Color(red: 0.961, green: 0.620, blue: 0.043) : Color(red: 0.16, green: 0.16, blue: 0.18)
    }

    var success: Color {
        isDark ? Color(red: 0.204, green: 0.827, blue: 0.600) : Color(red: 0.086, green: 0.565, blue: 0.349)
    }

    static func resolve(_ theme: HushTheme, systemScheme: ColorScheme) -> HushPalette {
        switch theme {
        case .graphite: HushPalette(isDark: true)
        case .ceramic: HushPalette(isDark: false)
        case .auto: HushPalette(isDark: systemScheme == .dark)
        }
    }
}

enum WaveformExtractor {
    static func readPeaks(from url: URL?, bars: Int = 52) -> [Float] {
        guard let url, bars > 0, let file = try? AVAudioFile(forReading: url) else {
            return []
        }
        let format = file.processingFormat
        let totalFrames = AVAudioFrameCount(file.length)
        guard totalFrames > 0, let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: totalFrames) else {
            return []
        }
        do {
            try file.read(into: buf)
        } catch {
            return []
        }
        guard let ch = buf.floatChannelData?[0] else { return [] }
        let count = Int(buf.frameLength)
        let chunk = max(1, count / bars)
        var peaks: [Float] = []
        peaks.reserveCapacity(bars)
        for idx in 0..<bars {
            let start = idx * chunk
            if start >= count {
                peaks.append(0.02)
                continue
            }
            let end = min(count, start + chunk)
            var maxVal: Float = 0
            for j in start..<end {
                let v = abs(ch[j])
                if v > maxVal { maxVal = v }
            }
            peaks.append(min(max(maxVal, 0.02), 1.0))
        }
        return peaks
    }
}

@MainActor
final class FloatingPillController {
    nonisolated private static let posXKey = "hush.floatingMic.x"
    nonisolated private static let posYKey = "hush.floatingMic.y"

    private enum PanelMode {
        case orb
        case pill

        var size: NSSize {
            switch self {
            case .orb: NSSize(width: 52, height: 52)
            case .pill: NSSize(width: 196, height: 52)
            }
        }
    }

    private var panel: NSPanel?
    private var alertPanel: NSPanel?
    private weak var state: AppState?
    private var hideWorkItem: DispatchWorkItem?
    private var alertHideWorkItem: DispatchWorkItem?
    private var lastShownAlert: FloatingAlert?
    private var moveObserver: NSObjectProtocol?
    private var openStudioHandler: (() -> Void)?
    private var isProgrammaticResize = false
    private var currentMode: PanelMode = .orb

    func attach(state: AppState, openStudio: @escaping () -> Void) {
        self.state = state
        self.openStudioHandler = openStudio
        syncVisibility()
    }

    func syncVisibility() {
        guard let state else { return }
        syncMicPanel(state: state)
        syncAlertPanel(state: state)
    }

    private func syncMicPanel(state: AppState) {
        hideWorkItem?.cancel()
        hideWorkItem = nil

        let activelyCapturing = state.audio.isStarting || state.audio.isRecording || state.isTranscribing
        if activelyCapturing {
            showMicPanel(state: state, mode: .pill)
            return
        }

        if state.showResultPillBriefly {
            showMicPanel(state: state, mode: .pill)
            let work = DispatchWorkItem { [weak self] in
                guard let self, let state = self.state else { return }
                state.showResultPillBriefly = false
                if state.keepPillVisible {
                    self.showMicPanel(state: state, mode: .orb)
                } else {
                    self.panel?.orderOut(nil)
                }
            }
            hideWorkItem = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.2, execute: work)
        } else if state.keepPillVisible {
            showMicPanel(state: state, mode: .orb)
        } else {
            panel?.orderOut(nil)
        }
    }

    private func syncAlertPanel(state: AppState) {
        guard let alert = state.floatingAlert else {
            alertHideWorkItem?.cancel()
            alertHideWorkItem = nil
            lastShownAlert = nil
            alertPanel?.orderOut(nil)
            return
        }

        let hasAudio = alert.audioURL != nil
        let hasSnippet = !(alert.clipboardText ?? "").isEmpty
        let size = NSSize(width: 320, height: hasAudio ? 228 : (hasSnippet ? 204 : 172))
        showAlertPanel(state: state, size: size)

        if lastShownAlert != alert {
            lastShownAlert = alert
            scheduleAlertAutoDismiss(after: 5.5)
        }
    }

    func scheduleAlertAutoDismiss(after seconds: TimeInterval = 5.5) {
        alertHideWorkItem?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self, let state = self.state else { return }
            if state.audio.playingKind == "alert-preview" {
                self.scheduleAlertAutoDismiss(after: 3.0)
                return
            }
            state.dismissFloatingAlert()
        }
        alertHideWorkItem = work
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds, execute: work)
    }

    private func makeFloatingPanel(size: NSSize) -> NSPanel {
        let p = NSPanel(
            contentRect: NSRect(origin: .zero, size: size),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        p.isFloatingPanel = true
        p.level = .floating
        p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
        p.backgroundColor = .clear
        p.isOpaque = false
        // Disable WindowServer rectangular frame shadow so only the clean SwiftUI surface renders.
        p.hasShadow = false
        p.hidesOnDeactivate = false
        return p
    }

    private func showMicPanel(state: AppState, mode: PanelMode) {
        currentMode = mode
        let targetSize = mode.size

        if panel == nil {
            let newPanel = makeFloatingPanel(size: targetSize)
            let root = AnyView(
                FloatingMicChipView(
                    onBeginDrag: { [weak newPanel] in
                        if let event = NSApp.currentEvent {
                            newPanel?.performDrag(with: event)
                        }
                    },
                    onOpenStudio: { [weak self] in
                        self?.openStudioHandler?()
                    }
                )
                .environment(state)
            )
            let host = NSHostingView(rootView: root)
            host.wantsLayer = true
            host.layer?.backgroundColor = NSColor.clear.cgColor
            host.frame = NSRect(origin: .zero, size: targetSize)
            host.autoresizingMask = [.width, .height]
            newPanel.contentView = host
            self.panel = newPanel

            moveObserver = NotificationCenter.default.addObserver(
                forName: NSWindow.didMoveNotification,
                object: newPanel,
                queue: .main
            ) { [weak self, weak newPanel] _ in
                MainActor.assumeIsolated {
                    guard let self, !self.isProgrammaticResize, let frame = newPanel?.frame else { return }
                    if case .orb = self.currentMode {
                        UserDefaults.standard.set(Double(frame.origin.x), forKey: Self.posXKey)
                        UserDefaults.standard.set(Double(frame.origin.y), forKey: Self.posYKey)
                    }
                }
            }
        }

        if let panel {
            let vis = (NSScreen.main ?? NSScreen.screens.first)?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
            let savedX = UserDefaults.standard.object(forKey: Self.posXKey) as? Double ?? Double(vis.midX - 26)
            let savedY = UserDefaults.standard.object(forKey: Self.posYKey) as? Double ?? Double(vis.minY + 36)

            let anchorMidX = savedX + 26
            var newX = anchorMidX - targetSize.width / 2
            var newY = savedY
            let margin: CGFloat = 12
            newX = max(vis.minX + margin, min(vis.maxX - targetSize.width - margin, newX))
            newY = max(vis.minY + margin, min(vis.maxY - targetSize.height - margin, newY))

            isProgrammaticResize = true
            panel.setFrame(NSRect(x: newX, y: newY, width: targetSize.width, height: targetSize.height), display: true)
            isProgrammaticResize = false
            panel.orderFrontRegardless()
        }
    }

    private func showAlertPanel(state: AppState, size: NSSize) {
        if alertPanel == nil {
            let newPanel = makeFloatingPanel(size: size)
            let root = AnyView(
                FloatingAlertHostView(
                    onOpenStudio: { [weak self] in
                        self?.openStudioHandler?()
                    },
                    onInteraction: { [weak self] in
                        self?.scheduleAlertAutoDismiss(after: 5.5)
                    }
                )
                .environment(state)
            )
            let host = NSHostingView(rootView: root)
            host.wantsLayer = true
            host.layer?.backgroundColor = NSColor.clear.cgColor
            host.frame = NSRect(origin: .zero, size: size)
            host.autoresizingMask = [.width, .height]
            newPanel.contentView = host
            self.alertPanel = newPanel
        }

        if let alertPanel {
            let vis = (NSScreen.main ?? NSScreen.screens.first)?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
            let x = vis.maxX - size.width - 20
            let y = vis.minY + 20
            alertPanel.setFrame(NSRect(x: x, y: y, width: size.width, height: size.height), display: true)
            alertPanel.orderFrontRegardless()
        }
    }
}

private struct DraggableMicHitTarget: NSViewRepresentable {
    var onTap: () -> Void
    var onDragStart: () -> Void

    func makeNSView(context: Context) -> HitView {
        let view = HitView()
        view.onTap = onTap
        view.onDragStart = onDragStart
        return view
    }

    func updateNSView(_ nsView: HitView, context: Context) {
        nsView.onTap = onTap
        nsView.onDragStart = onDragStart
    }

    final class HitView: NSView {
        var onTap: (() -> Void)?
        var onDragStart: (() -> Void)?
        private var downPoint: NSPoint = .zero
        private var didDrag = false

        override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

        override func mouseDown(with event: NSEvent) {
            downPoint = event.locationInWindow
            didDrag = false
        }

        override func mouseDragged(with event: NSEvent) {
            let dx = event.locationInWindow.x - downPoint.x
            let dy = event.locationInWindow.y - downPoint.y
            if hypot(dx, dy) >= 3 && !didDrag {
                didDrag = true
                onDragStart?()
            }
        }

        override func mouseUp(with event: NSEvent) {
            if !didDrag {
                onTap?()
            }
        }
    }
}

private struct FloatingMicChipView: View {
    @Environment(AppState.self) private var state
    @Environment(\.colorScheme) private var systemScheme
    var onBeginDrag: () -> Void
    var onOpenStudio: () -> Void

    private let barCount = 12

    var body: some View {
        let palette = HushPalette.resolve(state.theme, systemScheme: systemScheme)
        let activelyCapturing = state.audio.isStarting || state.audio.isRecording || state.isTranscribing
        let expanded = activelyCapturing || state.showResultPillBriefly

        Group {
            if expanded {
                expandedChip(palette: palette)
            } else {
                compactOrb(palette: palette)
            }
        }
        .overlay {
            DraggableMicHitTarget(
                onTap: {
                    if state.audio.isRecording || state.audio.isStarting {
                        Task { await state.endCaptureAndTranscribe() }
                    } else if !state.isTranscribing {
                        Task { await state.beginCapture() }
                    }
                },
                onDragStart: onBeginDrag
            )
            .contextMenu {
                Button("Open Hush Studio") { onOpenStudio() }
                Button("Hide Floating Mic") { state.keepPillVisible = false }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func compactOrb(palette: HushPalette) -> some View {
        ZStack {
            Circle()
                .fill(palette.background.opacity(0.95))
            Circle()
                .strokeBorder(palette.border, lineWidth: 1)
            Image(systemName: "mic.fill")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(palette.accent)
        }
        .frame(width: 40, height: 40)
        .clipShape(Circle())
        .shadow(color: .black.opacity(palette.isDark ? 0.35 : 0.16), radius: 4, y: 2)
        .help("Click to whisper · Drag anywhere to move · Right-click for Studio")
    }

    private func expandedChip(palette: HushPalette) -> some View {
        let activeBars = Int((min(max(state.audio.normalizedMeter, 0), 1) * Double(barCount)).rounded())
        return HStack(spacing: 8) {
            if state.isTranscribing {
                ProgressView().controlSize(.small)
                Text("Transcribing…")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(palette.textPrimary)
            } else if state.showResultPillBriefly && !state.audio.isRecording {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 13))
                    .foregroundStyle(palette.success)
                Text(state.lastPasteSucceeded ? "Pasted" : "Copied")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(palette.textPrimary)
                if let t = state.lastTimings {
                    Text(String(format: "%.0fms", t.asr))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(palette.textSecondary)
                }
            } else {
                Image(systemName: "mic.fill")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(palette.accent)
                HStack(spacing: 2.5) {
                    ForEach(0..<barCount, id: \.self) { idx in
                        RoundedRectangle(cornerRadius: 1.5)
                            .fill(idx < activeBars ? palette.accent : palette.textSecondary.opacity(0.25))
                            .frame(width: 3, height: idx == 5 ? 16 : 11)
                    }
                }
                Text(String(format: "%.0f", state.audio.levelDBFS))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(palette.textSecondary)
            }
        }
        .padding(.horizontal, 12)
        .frame(width: 184, height: 40)
        .background(Capsule().fill(palette.background.opacity(0.95)))
        .overlay(Capsule().strokeBorder(state.audio.isRecording ? palette.accent.opacity(0.7) : palette.border, lineWidth: 1))
        .clipShape(Capsule())
        .shadow(color: .black.opacity(palette.isDark ? 0.35 : 0.16), radius: 4, y: 2)
    }
}

private struct FloatingAlertHostView: View {
    @Environment(AppState.self) private var state
    @Environment(\.colorScheme) private var systemScheme
    var onOpenStudio: () -> Void
    var onInteraction: () -> Void

    var body: some View {
        let palette = HushPalette.resolve(state.theme, systemScheme: systemScheme)
        Group {
            if let alert = state.floatingAlert {
                FloatingAlertCardView(
                    alert: alert,
                    palette: palette,
                    isPlaying: state.audio.playingKind == "alert-preview",
                    onPlayToggle: {
                        onInteraction()
                        state.audio.togglePlay(kind: "alert-preview", url: alert.audioURL)
                    },
                    onDismiss: {
                        state.dismissFloatingAlert()
                    },
                    onCTA: {
                        handleAlertCTA(alert)
                    }
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func handleAlertCTA(_ alert: FloatingAlert) {
        switch alert.kind {
        case .voiceNotHeard:
            state.dismissFloatingAlert()
            state.openTuningOnShow = true
            onOpenStudio()
        case .pasteFailed:
            if !state.isAccessibilityTrusted {
                PasteHelper.openAccessibilitySettings()
            } else {
                if let text = alert.clipboardText {
                    PasteHelper.copyToClipboard(text)
                }
                state.dismissFloatingAlert()
            }
        case .transcriptionFailed:
            state.dismissFloatingAlert()
            onOpenStudio()
        }
    }
}

private struct FloatingAlertCardView: View {
    var alert: FloatingAlert
    var palette: HushPalette
    var isPlaying: Bool
    var onPlayToggle: () -> Void
    var onDismiss: () -> Void
    var onCTA: () -> Void

    private var iconName: String {
        switch alert.kind {
        case .voiceNotHeard: "mic.slash.fill"
        case .pasteFailed: "doc.on.clipboard.fill"
        case .transcriptionFailed: "exclamationmark.triangle.fill"
        }
    }

    var body: some View {
        let ctaText = palette.isDark ? Color(red: 0.08, green: 0.09, blue: 0.11) : Color.white

        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 10) {
                ZStack {
                    Circle()
                        .fill(palette.accent.opacity(0.16))
                        .frame(width: 28, height: 28)
                    Image(systemName: iconName)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(palette.accent)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(alert.title)
                        .font(.system(size: 14, weight: .bold))
                        .foregroundStyle(palette.textPrimary)
                    Text(alert.message)
                        .font(.system(size: 12))
                        .foregroundStyle(palette.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer(minLength: 4)

                Button(action: onDismiss) {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(palette.textSecondary)
                        .frame(width: 24, height: 24)
                        .background(palette.surfaceMuted, in: Circle())
                }
                .buttonStyle(.plain)
                .help("Dismiss")
            }

            if alert.audioURL != nil {
                HStack(spacing: 12) {
                    Button(action: onPlayToggle) {
                        Image(systemName: isPlaying ? "stop.fill" : "play.fill")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(ctaText)
                            .frame(width: 28, height: 28)
                            .background(palette.accent, in: Circle())
                    }
                    .buttonStyle(.plain)
                    .help(isPlaying ? "Stop playback" : "Play back recording")

                    AlertDottedWaveformView(
                        peaks: alert.peaks,
                        activeColor: palette.accent.opacity(isPlaying ? 0.95 : 0.48)
                    )
                    .frame(height: 20)
                }
            } else if let snippet = alert.clipboardText, !snippet.isEmpty {
                Text(snippet)
                    .font(.system(size: 11))
                    .foregroundStyle(palette.textSecondary)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(palette.surfaceMuted, in: RoundedRectangle(cornerRadius: 8))
            }

            Button(action: onCTA) {
                Text(alert.ctaTitle)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(ctaText)
                    .frame(maxWidth: .infinity)
                    .frame(height: 36)
                    .background(palette.accent, in: Capsule())
            }
            .buttonStyle(.plain)
        }
        .padding(16)
        .frame(width: 312)
        .background(palette.background.opacity(0.97), in: RoundedRectangle(cornerRadius: 20))
        .overlay(
            RoundedRectangle(cornerRadius: 20)
                .strokeBorder(palette.accent.opacity(0.38), lineWidth: 1)
        )
        .shadow(color: .black.opacity(palette.isDark ? 0.45 : 0.16), radius: 12, y: 4)
    }
}

private struct AlertDottedWaveformView: View {
    var peaks: [Float]
    var activeColor: Color
    private let dotCount = 34

    var body: some View {
        HStack(alignment: .center, spacing: 3) {
            ForEach(0..<dotCount, id: \.self) { idx in
                let amp: CGFloat = peaks.isEmpty
                    ? 0.22
                    : CGFloat(max(0.20, min(1.0, peaks[(idx * peaks.count) / dotCount])))
                Capsule()
                    .fill(activeColor)
                    .frame(width: 3, height: 3 + amp * 13)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
