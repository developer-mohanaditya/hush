import AppKit
import SwiftUI

struct MenuBarPopoverView: View {
    @Environment(AppState.self) private var state
    @Environment(\.openWindow) private var openWindow
    @Environment(\.colorScheme) private var systemScheme

    var body: some View {
        @Bindable var state = state
        @Bindable var audio = state.audio
        let palette = HushPalette.resolve(state.theme, systemScheme: systemScheme)

        VStack(alignment: .leading, spacing: 12) {
            headerRow(palette: palette)
            Divider().overlay(palette.border)

            VStack(alignment: .leading, spacing: 6) {
                Text("INPUT MICROPHONE")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(palette.textSecondary)
                HStack(spacing: 6) {
                    Picker("Mic", selection: $audio.selectedDeviceID) {
                        Text("Default (\(CaptureDevices.resolveDevice(uniqueID: nil)?.localizedName ?? "Mic"))")
                            .tag(String?.none)
                        ForEach(audio.devices) { dev in
                            Text(dev.name).tag(Optional(dev.id))
                        }
                    }
                    .labelsHidden()
                    .frame(maxWidth: .infinity)

                    Button {
                        audio.refreshDevices()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("ASR ENGINE")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(palette.textSecondary)
                Picker("Engine", selection: Binding(
                    get: { "\(state.options.engine)|\(state.options.model)" },
                    set: { state.selectEnginePreset($0) }
                )) {
                    ForEach(EnginePreset.all) { preset in
                        Text(preset.menuLabel).tag(preset.tag)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: .infinity)
            }

            VStack(spacing: 8) {
                Toggle(isOn: $state.options.dspEnabled) {
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Whisper DSP (+32 dB)")
                            .font(.system(size: 12, weight: .medium))
                        Text("95 Hz plosive filter · -58 dBFS silence gate")
                            .font(.system(size: 10))
                            .foregroundStyle(palette.textSecondary)
                    }
                }
                .toggleStyle(.switch)
                .controlSize(.small)

                Toggle(isOn: Binding(
                    get: { state.options.cleanupEnabled && state.options.cleanupMode == "rules" },
                    set: { enabled in
                        state.options.cleanupEnabled = enabled
                        if enabled { state.options.cleanupMode = "rules" }
                    }
                )) {
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Basic Polish (<1 ms)")
                            .font(.system(size: 12, weight: .medium))
                        Text("Instant grammar, casing & filler cleanup")
                            .font(.system(size: 10))
                            .foregroundStyle(palette.textSecondary)
                    }
                }
                .toggleStyle(.switch)
                .controlSize(.small)

                Toggle("Show Floating Mic Icon", isOn: $state.keepPillVisible)
                    .font(.system(size: 12, weight: .medium))
                    .toggleStyle(.switch)
                    .controlSize(.small)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("APPEARANCE")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(palette.textSecondary)
                Picker("Appearance", selection: $state.theme) {
                    ForEach(HushTheme.allCases) { item in
                        Text(item.label).tag(item)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            if !state.currentPrompt.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("LAST TAKE")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(palette.textSecondary)
                        Spacer()
                        Button("Copy") {
                            PasteHelper.copyToClipboard(state.currentPrompt)
                        }
                        .font(.system(size: 11, weight: .medium))
                        .buttonStyle(.borderless)
                    }
                    Text(state.currentPrompt)
                        .font(.system(size: 12))
                        .foregroundStyle(palette.textPrimary)
                        .lineLimit(3)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                        .background(palette.surface, in: RoundedRectangle(cornerRadius: 8))
                        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(palette.border, lineWidth: 1))
                }
            }

            Divider().overlay(palette.border)

            HStack {
                Button("Open Acoustic Studio") {
                    WindowRouter.showStudio(openWindow: openWindow)
                }
                .buttonStyle(.borderedProminent)
                .tint(palette.accent)
                .controlSize(.small)

                Spacer()

                if state.autoPaste && !state.isAccessibilityTrusted {
                    Button {
                        PasteHelper.openAccessibilitySettings()
                    } label: {
                        Image(systemName: "hand.raised.fill")
                            .font(.system(size: 11))
                    }
                    .buttonStyle(.bordered)
                    .tint(palette.accent)
                    .controlSize(.small)
                    .help("Grant Accessibility permission for Direct Paste")
                }

                Button("Quit") {
                    NSApp.terminate(nil)
                }
                .buttonStyle(.borderless)
                .foregroundStyle(palette.textSecondary)
                .font(.system(size: 12))
            }
        }
        .padding(14)
        .frame(width: 332)
        .background(palette.background)
        .foregroundStyle(palette.textPrimary)
        .preferredColorScheme(state.theme == .graphite ? .dark : (state.theme == .ceramic ? .light : nil))
        .task { await state.start() }
    }

    private func headerRow(palette: HushPalette) -> some View {
        HStack {
            HStack(spacing: 8) {
                Image(systemName: "waveform")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(palette.accent)
                Text("Hush")
                    .font(.system(size: 14, weight: .bold))
            }
            Spacer()
            HStack(spacing: 6) {
                Circle().fill(statusDotColor(palette: palette)).frame(width: 7, height: 7)
                Text(headerStatusText)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(palette.textSecondary)
            }
        }
    }

    private var headerStatusText: String {
        if let t = state.lastTimings, state.engine.status == .ready {
            return String(format: "Local · %.0f ms", t.asr)
        }
        return state.engine.status.label
    }

    private func statusDotColor(palette: HushPalette) -> Color {
        switch state.engine.status {
        case .ready: palette.success
        case .starting, .warming: palette.accent
        case .offline, .failed: .red
        }
    }
}
