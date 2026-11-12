import AppKit
import SwiftUI

struct EnginePreset: Identifiable, Hashable {
    let engine: String
    let model: String
    let shortName: String
    let menuLabel: String
    let tag: String

    var id: String { tag }

    init(engine: String, model: String, shortName: String, menuLabel: String) {
        self.engine = engine
        self.model = model
        self.shortName = shortName
        self.menuLabel = menuLabel
        self.tag = "\(engine)|\(model)"
    }

    static let all: [EnginePreset] = [
        EnginePreset(
            engine: "sherpa-parakeet",
            model: "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            shortName: "Parakeet INT8",
            menuLabel: "Parakeet INT8 · 1.9 GB RAM · 241 ms"
        ),
        EnginePreset(
            engine: "mlx-whisper",
            model: "mlx-community/whisper-large-v3-turbo",
            shortName: "Turbo MLX",
            menuLabel: "Whisper Turbo · 3.8 GB GPU · 387 ms"
        ),
        EnginePreset(
            engine: "mlx-whisper",
            model: "mlx-community/whisper-large-v3-mlx",
            shortName: "Large-v3 MLX",
            menuLabel: "Whisper Large-v3 · 6.6 GB GPU · 625 ms"
        ),
        EnginePreset(
            engine: "faster-whisper",
            model: "Systran/faster-whisper-base",
            shortName: "Base CPU",
            menuLabel: "Faster Whisper Base · 0.6 GB CPU"
        ),
    ]
}

struct WhisperTake: Identifiable, Hashable {
    let id = UUID()
    let timestamp = Date()
    var deviceName: String
    var rawText: String
    var cleanText: String
    var stats: TranscribeStats
    var timings: TranscribeTimings
    var cleanupApplied: Bool
    var cleanupModel: String?
    var rawAudioURL: URL?
    var boostedAudioURL: URL?
}

struct FloatingAlert: Equatable {
    enum Kind: Equatable {
        case voiceNotHeard
        case pasteFailed
        case transcriptionFailed
    }

    var kind: Kind
    var title: String
    var message: String
    var ctaTitle: String
    var audioURL: URL?
    var peaks: [Float]
    var clipboardText: String?
}

@MainActor @Observable
final class AppState {
    let audio = AudioCaptureService()
    let engine = EngineClient()
    let hotkeys = PushToTalkMonitor()

    var options = TranscribeOptions()
    var autoCopy = true
    var autoPaste = true
    var isAccessibilityTrusted = PasteHelper.isAccessibilityTrusted
    var isInputMonitoringGranted = PushToTalkMonitor.isInputMonitoringGranted
    var lastPasteSucceeded = false
    var theme: HushTheme = .graphite
    var keepPillVisible = true {
        didSet { pillController?.syncVisibility() }
    }
    var showResultPillBriefly = false {
        didSet {
            if showResultPillBriefly {
                pillController?.syncVisibility()
            }
        }
    }
    var floatingAlert: FloatingAlert? {
        didSet { pillController?.syncVisibility() }
    }
    var openTuningOnShow = false

    var isTranscribing = false
    var currentPrompt = ""
    var lastRawText = ""
    var lastStats: TranscribeStats?
    var lastTimings: TranscribeTimings?
    var lastCleanupInfo: String?
    var statusBanner: String?
    var fetchedCustomModels: [String] = []
    var isFetchingModels = false
    var customModelsError: String?
    var rawPeaks: [Float] = []
    var boostedPeaks: [Float] = []
    var history: [WhisperTake] = []

    @ObservationIgnored weak var pillController: FloatingPillController?
    @ObservationIgnored private var started = false
    @ObservationIgnored private var pendingRelease = false
    @ObservationIgnored private var promptedAccessibility = false
    @ObservationIgnored private var activeObserver: NSObjectProtocol?
    @ObservationIgnored private var permissionPollTask: Task<Void, Never>?

    var activeEngineShortName: String {
        let tag = "\(options.engine)|\(options.model)"
        return EnginePreset.all.first(where: { $0.tag == tag })?.shortName ?? options.engine
    }

    func selectEnginePreset(_ tag: String) {
        let parts = tag.split(separator: "|", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return }
        options.engine = parts[0]
        options.model = parts[1]
        Task { await engine.warm(engine: parts[0], model: parts[1]) }
    }

    func refreshPermissions() {
        isAccessibilityTrusted = PasteHelper.isAccessibilityTrusted
        isInputMonitoringGranted = PushToTalkMonitor.isInputMonitoringGranted
        if isAccessibilityTrusted, statusBanner?.contains("Accessibility") == true {
            statusBanner = nil
        }
        if isAccessibilityTrusted, floatingAlert?.kind == .pasteFailed {
            floatingAlert = nil
        }
    }

    func dismissFloatingAlert() {
        if audio.playingKind == "alert-preview" {
            audio.togglePlay(kind: "alert-preview", url: nil)
        }
        floatingAlert = nil
    }

    func persistCustomLLM() {
        UserDefaults.standard.set(options.customBaseURL, forKey: "hush.customBaseURL")
        UserDefaults.standard.set(options.customModel, forKey: "hush.customModel")
        KeychainHelper.saveAPIKey(options.customApiKey)
    }

    func fetchCustomModels() async {
        persistCustomLLM()
        isFetchingModels = true
        customModelsError = nil
        let result = await engine.fetchCustomModels(baseURL: options.customBaseURL, apiKey: options.customApiKey)
        isFetchingModels = false
        fetchedCustomModels = result.models
        customModelsError = result.error
        if let first = result.models.first,
           options.customModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !result.models.contains(options.customModel) {
            options.customModel = first
            persistCustomLLM()
        }
    }

    func start() async {
        guard !started else { return }
        started = true
        refreshPermissions()
        activeObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.refreshPermissions()
            }
        }
        permissionPollTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1.5))
                guard let self else { break }
                if (self.autoPaste && !self.isAccessibilityTrusted) ||
                    (self.hotkeys.globalRightOptionEnabled && !self.isInputMonitoringGranted) {
                    self.refreshPermissions()
                }
            }
        }
        audio.refreshDevices()

        hotkeys.allowLocalSpace = { [weak self] in
            guard let self else { return false }
            return !self.isTranscribing
        }
        hotkeys.onBegin = { [weak self] in
            Task { @MainActor [weak self] in await self?.beginCapture() }
        }
        hotkeys.onEnd = { [weak self] in
            Task { @MainActor [weak self] in await self?.endCaptureAndTranscribe() }
        }
        hotkeys.start()

        await engine.ensureStarted()
        if options.ollamaModel == nil {
            options.ollamaModel = engine.defaultOllamaModel
        }
        pillController?.syncVisibility()
    }

    func beginCapture() async {
        guard !audio.isRecording, !audio.isStarting, !isTranscribing else { return }
        pendingRelease = false
        showResultPillBriefly = false
        dismissFloatingAlert()
        lastPasteSucceeded = false
        statusBanner = nil
        pillController?.syncVisibility()
        let ok = await audio.startRecording()
        pillController?.syncVisibility()
        if !ok {
            pendingRelease = false
            pillController?.syncVisibility()
            return
        }
        if pendingRelease {
            await endCaptureAndTranscribe()
        }
    }

    func cancelCapture() async {
        pendingRelease = false
        if let rawURL = await audio.stopRecording() {
            try? FileManager.default.removeItem(at: rawURL)
        }
        statusBanner = nil
        pillController?.syncVisibility()
    }

    func endCaptureAndTranscribe() async {
        if audio.isStarting {
            pendingRelease = true
            return
        }
        guard audio.isRecording else { return }
        guard let rawURL = await audio.stopRecording() else {
            statusBanner = "Hold slightly longer while whispering (no audio captured)."
            floatingAlert = FloatingAlert(
                kind: .voiceNotHeard,
                title: "We couldn't hear you",
                message: "No audio was captured. Hold slightly longer, then check your microphone.",
                ctaTitle: "Check Microphone",
                audioURL: nil,
                peaks: [],
                clipboardText: nil
            )
            return
        }

        isTranscribing = true
        pillController?.syncVisibility()
        defer {
            isTranscribing = false
            pillController?.syncVisibility()
        }

        do {
            let response = try await engine.transcribe(fileURL: rawURL, options: options)
            let boostedURL = audio.saveBoostedWav(base64: response.boosted_wav_b64, matchingRawURL: rawURL)
            lastStats = response.stats
            lastTimings = response.timings_ms
            lastRawText = response.raw_text
            rawPeaks = WaveformExtractor.readPeaks(from: rawURL, bars: 48)
            boostedPeaks = WaveformExtractor.readPeaks(from: boostedURL, bars: 48)

            if response.stats.too_quiet || response.clean_text.isEmpty {
                statusBanner = String(
                    format: "Signal gated (raw %.1f dBFS vs gate %.0f dBFS). Lower Noise Gate in Tuning or speak closer.",
                    response.stats.raw_rms_dbfs,
                    options.silenceGateDbfs
                )
                floatingAlert = FloatingAlert(
                    kind: .voiceNotHeard,
                    title: "We couldn't hear you",
                    message: String(
                        format: "We didn't pick up speech above your %.0f dBFS gate (raw %.1f dBFS). Play back your recording, then check your microphone.",
                        options.silenceGateDbfs,
                        response.stats.raw_rms_dbfs
                    ),
                    ctaTitle: "Check Microphone",
                    audioURL: rawURL,
                    peaks: rawPeaks,
                    clipboardText: nil
                )
                return
            }

            if response.stats.raw_rms_dbfs < -48.0,
               audio.activeDeviceName.localizedCaseInsensitiveContains("OnePlus") {
                statusBanner = String(
                    format: "Soft input (%.1f dBFS) over Bluetooth ENC — compare with MacBook Pro Microphone in the Mic menu.",
                    response.stats.raw_rms_dbfs
                )
            }

            currentPrompt = response.clean_text
            if response.cleanup_applied {
                switch options.cleanupMode {
                case "ollama":
                    lastCleanupInfo = "AI Rewrite (\(response.cleanup_model ?? "Ollama"))"
                case "custom_openai":
                    lastCleanupInfo = "Custom API (\(response.cleanup_model ?? "Model"))"
                default:
                    lastCleanupInfo = "Basic Polish (<1 ms)"
                }
            } else if let err = response.cleanup_error {
                lastCleanupInfo = "Raw ASR (rewrite skipped: \(err))"
            } else {
                lastCleanupInfo = "Raw ASR"
            }

            if autoPaste {
                let pasted = PasteHelper.pasteIntoActiveApp(response.clean_text)
                lastPasteSucceeded = pasted
                if !pasted {
                    if !promptedAccessibility {
                        promptedAccessibility = true
                        PasteHelper.promptAccessibility()
                    }
                    statusBanner = "Copied to clipboard. Grant Accessibility to Hush (if already ON from an earlier build, toggle it OFF → ON once)."
                    floatingAlert = FloatingAlert(
                        kind: .pasteFailed,
                        title: "Couldn't paste text",
                        message: isAccessibilityTrusted
                            ? "Couldn't send ⌘V to the focused field. Your text is ready on the clipboard."
                            : "Accessibility access is needed to paste automatically. Your text is copied to the clipboard.",
                        ctaTitle: isAccessibilityTrusted ? "Copy Text Again" : "Grant Paste Access",
                        audioURL: nil,
                        peaks: [],
                        clipboardText: response.clean_text
                    )
                }
            } else if autoCopy {
                lastPasteSucceeded = false
                PasteHelper.copyToClipboard(response.clean_text)
            } else {
                lastPasteSucceeded = false
            }

            showResultPillBriefly = true
            let take = WhisperTake(
                deviceName: audio.activeDeviceName,
                rawText: response.raw_text,
                cleanText: response.clean_text,
                stats: response.stats,
                timings: response.timings_ms,
                cleanupApplied: response.cleanup_applied,
                cleanupModel: response.cleanup_model,
                rawAudioURL: rawURL,
                boostedAudioURL: boostedURL
            )
            history.insert(take, at: 0)
            if history.count > 20 {
                let evicted = history.removeLast()
                if let u = evicted.rawAudioURL { try? FileManager.default.removeItem(at: u) }
                if let u = evicted.boostedAudioURL { try? FileManager.default.removeItem(at: u) }
            }
        } catch {
            statusBanner = "Transcription failed: \(error.localizedDescription)"
            floatingAlert = FloatingAlert(
                kind: .transcriptionFailed,
                title: "Transcription failed",
                message: error.localizedDescription,
                ctaTitle: "Open Hush Studio",
                audioURL: rawURL,
                peaks: rawPeaks,
                clipboardText: nil
            )
        }
    }
}

struct ContentView: View {
    @Environment(AppState.self) private var state
    @Environment(\.colorScheme) private var systemScheme
    @State private var showSettings = false
    @FocusState private var outputFocused: Bool

    var body: some View {
        @Bindable var state = state
        let palette = HushPalette.resolve(state.theme, systemScheme: systemScheme)

        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                topBar(palette: palette)
                diagnosticRow(palette: palette)
                transcriptCard(palette: palette)
                holdBar(palette: palette)
                if showSettings { settingsCard(palette: palette) }
                if !state.history.isEmpty { historyCard(palette: palette) }
            }
            .padding(14)
        }
        .background(palette.background)
        .foregroundStyle(palette.textPrimary)
        .preferredColorScheme(state.theme == .graphite ? .dark : (state.theme == .ceramic ? .light : nil))
        .frame(minWidth: 680, minHeight: 430)
        .task {
            await state.start()
            if state.openTuningOnShow {
                showSettings = true
                state.openTuningOnShow = false
            }
        }
        .onChange(of: state.openTuningOnShow) { _, openTuning in
            if openTuning {
                showSettings = true
                state.openTuningOnShow = false
            }
        }
    }

    private func topBar(palette: HushPalette) -> some View {
        @Bindable var state = state
        @Bindable var audio = state.audio

        return HStack(spacing: 8) {
            HStack(spacing: 7) {
                Image(systemName: "waveform")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(palette.accent)
                Text("Hush")
                    .font(.system(size: 16, weight: .bold))
                HStack(spacing: 5) {
                    Circle().fill(statusColor(palette: palette)).frame(width: 6, height: 6)
                    Text(state.engine.status.label)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(palette.textSecondary)
                }
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(palette.surface, in: Capsule())
                .overlay(Capsule().strokeBorder(palette.border, lineWidth: 1))
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Picker("Engine", selection: Binding(
                get: { "\(state.options.engine)|\(state.options.model)" },
                set: { state.selectEnginePreset($0) }
            )) {
                ForEach(EnginePreset.all.prefix(3)) { preset in
                    Text(preset.shortName).tag(preset.tag)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlSize(.small)
            .frame(width: 265)

            HStack(spacing: 6) {
                Picker("Mic", selection: $audio.selectedDeviceID) {
                    Text("Default (\(CaptureDevices.resolveDevice(uniqueID: nil)?.localizedName ?? "Mic"))")
                        .tag(String?.none)
                    ForEach(audio.devices) { dev in
                        Text(dev.name).tag(Optional(dev.id))
                    }
                }
                .labelsHidden()
                .controlSize(.small)
                .frame(maxWidth: 150)

                Button {
                    audio.refreshDevices()
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 11))
                }
                .controlSize(.small)
                .help("Refresh microphones")

                Button {
                    showSettings.toggle()
                } label: {
                    Image(systemName: "slider.horizontal.3")
                        .font(.system(size: 11))
                }
                .controlSize(.small)
                .help("DSP, Model & Hotkey Tuning")
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
        }
    }

    private func diagnosticRow(palette: HushPalette) -> some View {
        let rawLabel = state.lastStats.map { String(format: "%.1f dBFS", $0.raw_rms_dbfs) }
            ?? (state.audio.isRecording ? String(format: "%.0f dBFS", state.audio.levelDBFS) : "-48 dBFS target")
        let boostedLabel = state.lastStats.map { String(format: "%.1f dBFS", $0.processed_rms_dbfs) } ?? "-18 dBFS target"

        return HStack(spacing: 10) {
            waveformCard(
                title: "Raw · \(rawLabel)",
                peaks: state.rawPeaks,
                barColor: palette.textSecondary.opacity(0.75),
                badgeText: nil,
                iconName: state.audio.playingKind == "raw" ? "stop.fill" : "speaker.wave.1.fill",
                playHelp: state.audio.playingKind == "raw" ? "Stop Raw Playback" : "Play Raw Capture",
                playDisabled: state.audio.lastRawURL == nil,
                palette: palette,
                onPlay: { state.audio.togglePlay(kind: "raw", url: state.audio.lastRawURL) }
            )

            waveformCard(
                title: "Boosted · \(boostedLabel)",
                peaks: state.boostedPeaks,
                barColor: palette.textPrimary.opacity(0.92),
                badgeText: state.options.dspEnabled ? "Plosives Filtered (\(Int(state.options.highpassHz)) Hz)" : "DSP Off",
                iconName: state.audio.playingKind == "boosted" ? "stop.fill" : "speaker.wave.3.fill",
                playHelp: state.audio.playingKind == "boosted" ? "Stop Boosted Playback" : "Play Boosted Capture",
                playDisabled: state.audio.lastBoostedURL == nil,
                palette: palette,
                onPlay: { state.audio.togglePlay(kind: "boosted", url: state.audio.lastBoostedURL) }
            )
        }
    }

    private func waveformCard(
        title: String,
        peaks: [Float],
        barColor: Color,
        badgeText: String?,
        iconName: String,
        playHelp: String,
        playDisabled: Bool,
        palette: HushPalette,
        onPlay: @escaping () -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(palette.textSecondary)
                Spacer()
                if let badgeText {
                    Text(badgeText)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(palette.success)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(palette.success.opacity(0.14), in: RoundedRectangle(cornerRadius: 5))
                }
                Button(action: onPlay) {
                    Image(systemName: iconName)
                        .font(.system(size: 11, weight: .semibold))
                        .frame(width: 24, height: 22)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(playDisabled)
                .help(playHelp)
            }

            WaveformBarsView(peaks: peaks, color: barColor, liveLevel: state.audio.isRecording ? state.audio.normalizedMeter : nil)
                .frame(height: 42)
        }
        .padding(11)
        .frame(maxWidth: .infinity)
        .background(palette.surface, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(palette.border, lineWidth: 1))
    }

    private func transcriptCard(palette: HushPalette) -> some View {
        @Bindable var state = state
        let polishBadge: String = {
            if let t = state.lastTimings {
                return String(format: "%.1f ms %@", t.cleanup, state.options.cleanupMode == "rules" ? "Basic Polish" : "Rewrite")
            }
            if !state.options.cleanupEnabled {
                return "Raw Output"
            }
            return switch state.options.cleanupMode {
            case "rules": "Basic Polish (<1 ms)"
            case "ollama": "AI Rewrite"
            case "custom_openai": state.options.customModel.isEmpty ? "Custom API" : state.options.customModel
            default: "Polish"
            }
        }()

        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("Transcript")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(palette.textSecondary)
                if let info = state.lastCleanupInfo {
                    Text("· \(info)")
                        .font(.system(size: 10))
                        .foregroundStyle(palette.textSecondary)
                }
                Spacer()
                Text(polishBadge)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(palette.textSecondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(palette.surfaceMuted, in: RoundedRectangle(cornerRadius: 5))
                    .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(palette.border, lineWidth: 1))

                Toggle("Auto-Copy", isOn: $state.autoCopy)
                    .toggleStyle(.checkbox)
                    .font(.caption2)
                Toggle("Direct Paste (⌘V)", isOn: $state.autoPaste)
                    .toggleStyle(.checkbox)
                    .font(.caption2)
                    .onChange(of: state.autoPaste) { _, newValue in
                        if newValue && !PasteHelper.isAccessibilityTrusted {
                            PasteHelper.openAccessibilitySettings()
                        }
                    }

                if state.autoPaste && !state.isAccessibilityTrusted {
                    Button {
                        PasteHelper.openAccessibilitySettings()
                    } label: {
                        Label("Grant Paste Access", systemImage: "hand.raised.fill")
                            .font(.system(size: 10, weight: .semibold))
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(palette.accent)
                    .controlSize(.mini)
                    .help("Open Privacy & Security → Accessibility to allow Hush to paste into active fields")
                }

                if state.hotkeys.globalRightOptionEnabled && !state.isInputMonitoringGranted {
                    Button {
                        PushToTalkMonitor.openInputMonitoringSettings()
                    } label: {
                        Label("Grant Key Access", systemImage: "keyboard")
                            .font(.system(size: 10, weight: .semibold))
                    }
                    .buttonStyle(.bordered)
                    .tint(palette.accent)
                    .controlSize(.mini)
                    .help("Open Privacy & Security → Input Monitoring so Hold Right ⌥ works across apps")
                }

                Button {
                    PasteHelper.copyToClipboard(state.currentPrompt)
                } label: {
                    Image(systemName: "doc.on.doc")
                        .font(.system(size: 11))
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(state.currentPrompt.isEmpty)
                .help("Copy transcript")
            }

            TextEditor(text: $state.currentPrompt)
                .focused($outputFocused)
                .scrollContentBackground(.hidden)
                .font(.system(size: 14, weight: .regular))
                .frame(minHeight: 56, maxHeight: 88)
                .onExitCommand { outputFocused = false }

            if outputFocused {
                Text("Editing text — press Esc or hold Right ⌥ to dictate.")
                    .font(.caption2)
                    .foregroundStyle(palette.textSecondary)
            }

            if !state.lastRawText.isEmpty && state.lastRawText != state.currentPrompt {
                Text("Raw ASR: \(state.lastRawText)")
                    .font(.caption2)
                    .foregroundStyle(palette.textSecondary)
                    .textSelection(.enabled)
            }

            if state.audio.plosiveWarning {
                Label("Air-blast clipping detected — angle mic slightly to the side of your mouth.", systemImage: "wind")
                    .font(.caption2)
                    .foregroundStyle(palette.accent)
            }

            if let err = state.audio.error ?? state.statusBanner {
                Text(err).font(.caption2).foregroundStyle(.red)
            } else if (state.autoPaste && !state.isAccessibilityTrusted) ||
                        (state.hotkeys.globalRightOptionEnabled && !state.isInputMonitoringGranted) {
                Label(
                    "If Hush is already ON in Accessibility or Input Monitoring from an earlier build, toggle it OFF → ON once for this rebuild.",
                    systemImage: "info.circle"
                )
                .font(.caption2)
                .foregroundStyle(palette.textSecondary)
            }
        }
        .padding(11)
        .background(palette.surface, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(palette.border, lineWidth: 1))
    }

    private func holdBar(palette: HushPalette) -> some View {
        VStack(spacing: 6) {
            HStack(spacing: 8) {
                Text("Level")
                    .font(.system(size: 10))
                    .foregroundStyle(palette.textSecondary)
                WhisperMeter(
                    value: state.audio.normalizedMeter,
                    gateDbfs: state.options.silenceGateDbfs,
                    isAboveGate: Double(state.audio.levelDBFS) >= state.options.silenceGateDbfs,
                    isRecording: state.audio.isRecording,
                    accent: palette.accent,
                    success: palette.success
                )
                Text(
                    state.audio.isRecording
                        ? String(format: "%.0f dBFS", state.audio.levelDBFS)
                        : "gate \(Int(state.options.silenceGateDbfs)) dBFS"
                )
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(palette.textSecondary)
                .frame(width: 96, alignment: .trailing)
            }

            HoldToWhisperBar(
                isStarting: state.audio.isStarting,
                isRecording: state.audio.isRecording,
                isTranscribing: state.isTranscribing,
                elapsed: state.audio.elapsedSeconds,
                deviceLabel: state.audio.targetDeviceLabel,
                palette: palette,
                onBegin: {
                    outputFocused = false
                    Task { await state.beginCapture() }
                },
                onEnd: { Task { await state.endCaptureAndTranscribe() } }
            )
        }
    }

    private func settingsCard(palette: HushPalette) -> some View {
        @Bindable var state = state
        return VStack(alignment: .leading, spacing: 10) {
            Text("DSP, Hotkeys & Text Tuning")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(palette.textSecondary)

            HStack(spacing: 14) {
                Toggle("Whisper DSP", isOn: $state.options.dspEnabled)
                Toggle("Floating Mic Icon", isOn: $state.keepPillVisible)
                Toggle("Hold Right ⌥", isOn: Binding(
                    get: { state.hotkeys.globalRightOptionEnabled },
                    set: { enabled in
                        state.hotkeys.globalRightOptionEnabled = enabled
                        if enabled && !PushToTalkMonitor.isInputMonitoringGranted {
                            PushToTalkMonitor.requestInputMonitoring()
                        }
                    }
                ))
            }
            .toggleStyle(.checkbox)
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 16) {
                HStack(spacing: 8) {
                    Text("Theme")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                        .frame(width: 52, alignment: .leading)
                    Picker("Theme", selection: $state.theme) {
                        ForEach(HushTheme.allCases) { item in
                            Text(item.label).tag(item)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(maxWidth: .infinity)
                }
                .frame(maxWidth: .infinity)

                HStack(spacing: 8) {
                    Text("Polish")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                        .frame(width: 44, alignment: .leading)
                    Picker("Polish", selection: Binding(
                        get: { state.options.cleanupEnabled ? state.options.cleanupMode : "none" },
                        set: { mode in
                            if mode == "none" {
                                state.options.cleanupEnabled = false
                            } else {
                                state.options.cleanupEnabled = true
                                state.options.cleanupMode = mode
                            }
                        }
                    )) {
                        Text("Basic Polish (Instant)").tag("rules")
                        Text("AI Rewrite (Ollama)").tag("ollama")
                        Text("Custom API (Base URL + Key)").tag("custom_openai")
                        Text("Off (Raw)").tag("none")
                    }
                    .labelsHidden()
                    .frame(maxWidth: .infinity)
                }
                .frame(maxWidth: .infinity)
            }
            .frame(maxWidth: .infinity)

            if state.options.cleanupEnabled && state.options.cleanupMode == "ollama" && !state.engine.ollamaModels.isEmpty {
                HStack(spacing: 8) {
                    Text("Ollama Model")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                        .frame(width: 82, alignment: .leading)
                    Picker("Ollama Model", selection: Binding(
                        get: { state.options.ollamaModel ?? state.engine.defaultOllamaModel ?? state.engine.ollamaModels.first ?? "" },
                        set: { state.options.ollamaModel = $0 }
                    )) {
                        ForEach(state.engine.ollamaModels, id: \.self) { model in
                            Text(model).tag(model)
                        }
                    }
                    .labelsHidden()
                    .frame(maxWidth: .infinity)
                }
            }

            if state.options.cleanupEnabled && state.options.cleanupMode == "custom_openai" {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 8) {
                        Text("Base URL")
                            .font(.caption)
                            .foregroundStyle(palette.textSecondary)
                            .frame(width: 56, alignment: .leading)
                        TextField("https://api.openai.com/v1", text: $state.options.customBaseURL)
                            .textFieldStyle(.roundedBorder)
                            .onChange(of: state.options.customBaseURL) { _, _ in state.persistCustomLLM() }

                        Text("API Key")
                            .font(.caption)
                            .foregroundStyle(palette.textSecondary)
                        SecureField("Optional for local endpoints", text: $state.options.customApiKey)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 150)
                            .onChange(of: state.options.customApiKey) { _, _ in state.persistCustomLLM() }

                        Button {
                            Task { await state.fetchCustomModels() }
                        } label: {
                            if state.isFetchingModels {
                                ProgressView().controlSize(.small)
                            } else {
                                Text("Fetch Models")
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(state.isFetchingModels || state.options.customBaseURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }

                    HStack(spacing: 8) {
                        Text("Model")
                            .font(.caption)
                            .foregroundStyle(palette.textSecondary)
                            .frame(width: 56, alignment: .leading)
                        if !state.fetchedCustomModels.isEmpty {
                            Picker("Model", selection: $state.options.customModel) {
                                ForEach(state.fetchedCustomModels, id: \.self) { m in
                                    Text(m).tag(m)
                                }
                            }
                            .labelsHidden()
                            .frame(maxWidth: .infinity)
                            .onChange(of: state.options.customModel) { _, _ in state.persistCustomLLM() }
                        }
                        TextField("Model ID (e.g. gpt-4o-mini)", text: $state.options.customModel)
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: state.fetchedCustomModels.isEmpty ? .infinity : 180)
                            .onChange(of: state.options.customModel) { _, _ in state.persistCustomLLM() }
                    }

                    if let err = state.customModelsError, !err.isEmpty {
                        Text(err)
                            .font(.caption2)
                            .foregroundStyle(.red)
                    }
                }
                .padding(8)
                .background(palette.surfaceMuted, in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(palette.border, lineWidth: 1))
            }

            HStack(spacing: 14) {
                HStack(spacing: 6) {
                    Text("High-pass")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                    Slider(value: $state.options.highpassHz, in: 0...160, step: 5)
                    Text("\(Int(state.options.highpassHz)) Hz")
                        .font(.caption2.monospacedDigit())
                        .frame(width: 40, alignment: .trailing)
                }
                .frame(maxWidth: .infinity)

                HStack(spacing: 6) {
                    Text("Max Boost")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                    Slider(value: $state.options.maxGainDb, in: 6...40, step: 2)
                    Text("+\(Int(state.options.maxGainDb)) dB")
                        .font(.caption2.monospacedDigit())
                        .frame(width: 44, alignment: .trailing)
                }
                .frame(maxWidth: .infinity)

                HStack(spacing: 6) {
                    Text("Noise Gate")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                    Slider(value: $state.options.silenceGateDbfs, in: -75 ... -45, step: 1)
                    Text("\(Int(state.options.silenceGateDbfs)) dBFS")
                        .font(.caption2.monospacedDigit())
                        .frame(width: 54, alignment: .trailing)
                }
                .frame(maxWidth: .infinity)
            }
            .frame(maxWidth: .infinity)
        }
        .font(.caption)
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.surface, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(palette.border, lineWidth: 1))
    }

    private func historyCard(palette: HushPalette) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Recent Takes")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(palette.textSecondary)
            ForEach(state.history) { item in
                HStack(alignment: .center, spacing: 8) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(item.cleanText).font(.caption).textSelection(.enabled)
                        Text(
                            String(
                                format: "%@ · raw %.1f dBFS (+%.1f dB) · %.0f ms",
                                item.deviceName, item.stats.raw_rms_dbfs,
                                item.stats.applied_gain_db, item.timings.asr
                            )
                        )
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(palette.textSecondary)
                    }
                    Spacer()
                    HStack(spacing: 6) {
                        let rawKey = "take-\(item.id)-raw"
                        let boostedKey = "take-\(item.id)-boosted"
                        Button {
                            state.audio.togglePlay(kind: rawKey, url: item.rawAudioURL)
                        } label: {
                            Image(systemName: state.audio.playingKind == rawKey ? "stop.fill" : "speaker.wave.1.fill")
                                .font(.system(size: 11))
                                .frame(width: 22, height: 20)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .disabled(item.rawAudioURL == nil)
                        .help(state.audio.playingKind == rawKey ? "Stop Raw" : "Play Raw")

                        Button {
                            state.audio.togglePlay(kind: boostedKey, url: item.boostedAudioURL)
                        } label: {
                            Image(systemName: state.audio.playingKind == boostedKey ? "stop.fill" : "speaker.wave.3.fill")
                                .font(.system(size: 11))
                                .frame(width: 22, height: 20)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .disabled(item.boostedAudioURL == nil)
                        .help(state.audio.playingKind == boostedKey ? "Stop Boosted" : "Play Boosted")

                        Button {
                            PasteHelper.copyToClipboard(item.cleanText)
                        } label: {
                            Image(systemName: "doc.on.doc")
                                .font(.system(size: 11))
                                .frame(width: 22, height: 20)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .help("Copy")
                    }
                }
                Divider().overlay(palette.border)
            }
        }
        .padding(12)
        .background(palette.surface, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(palette.border, lineWidth: 1))
    }

    private func statusColor(palette: HushPalette) -> Color {
        switch state.engine.status {
        case .ready: palette.success
        case .starting, .warming: palette.accent
        case .offline, .failed: .red
        }
    }
}

private struct WaveformBarsView: View {
    var peaks: [Float]
    var color: Color
    var liveLevel: Double?

    var body: some View {
        GeometryReader { geo in
            let count = peaks.isEmpty ? 44 : peaks.count
            HStack(alignment: .center, spacing: 2) {
                ForEach(0..<count, id: \.self) { idx in
                    let amp: CGFloat = if let liveLevel {
                        CGFloat(max(0.08, liveLevel * (0.45 + 0.55 * sin(Double(idx) * 0.7))))
                    } else if peaks.isEmpty {
                        0.06
                    } else {
                        CGFloat(max(0.06, peaks[idx]))
                    }
                    Capsule()
                        .fill(color)
                        .frame(maxWidth: .infinity)
                        .frame(height: max(2.5, geo.size.height * amp))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

private struct HoldToWhisperBar: View {
    var isStarting: Bool
    var isRecording: Bool
    var isTranscribing: Bool
    var elapsed: TimeInterval
    var deviceLabel: String
    var palette: HushPalette
    var onBegin: () -> Void
    var onEnd: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            if isTranscribing {
                ProgressView().controlSize(.small)
                Text("Decoding whisper…")
                    .font(.system(size: 12, weight: .semibold))
            } else if isStarting {
                ProgressView().controlSize(.small)
                Text("Connecting \(deviceLabel)…")
                    .font(.system(size: 12, weight: .semibold))
            } else {
                Image(systemName: isRecording ? "waveform" : "mic.fill")
                    .font(.system(size: 12))
                    .foregroundStyle(isRecording ? palette.accent : palette.textPrimary)
                Text(isRecording ? String(format: "Whispering · %.1fs (Release to insert)", elapsed) : "Hold Right ⌥ or Space  ·  \(deviceLabel)")
                    .font(.system(size: 12, weight: .semibold))
            }
        }
        .frame(maxWidth: .infinity, minHeight: 34)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isRecording ? palette.accent.opacity(0.18) : palette.surface)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(isRecording ? palette.accent : palette.border, lineWidth: isRecording ? 1.5 : 1)
        )
        .overlay {
            HoldPadMouseOverlay(isEnabled: !isTranscribing, onPress: onBegin, onRelease: onEnd)
        }
        .contentShape(Rectangle())
    }
}

private struct HoldPadMouseOverlay: NSViewRepresentable {
    var isEnabled: Bool
    var onPress: () -> Void
    var onRelease: () -> Void

    func makeNSView(context: Context) -> PadMouseView {
        let view = PadMouseView()
        view.isEnabled = isEnabled
        view.onPress = onPress
        view.onRelease = onRelease
        return view
    }

    func updateNSView(_ nsView: PadMouseView, context: Context) {
        nsView.isEnabled = isEnabled
        nsView.onPress = onPress
        nsView.onRelease = onRelease
    }

    final class PadMouseView: NSView {
        var isEnabled = true
        var onPress: (() -> Void)?
        var onRelease: (() -> Void)?
        private var isPressed = false

        override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

        override func mouseDown(with event: NSEvent) {
            window?.makeFirstResponder(nil)
            guard isEnabled, !isPressed else { return }
            isPressed = true
            onPress?()
        }

        override func mouseUp(with event: NSEvent) {
            guard isPressed else { return }
            isPressed = false
            onRelease?()
        }
    }
}

private struct WhisperMeter: View {
    var value: Double
    var gateDbfs: Double
    var isAboveGate: Bool
    var isRecording: Bool
    var accent: Color
    var success: Color

    var body: some View {
        let gateFraction = min(max((gateDbfs + 75.0) / 75.0, 0), 1)
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary)
                Capsule()
                    .fill((isRecording ? (isAboveGate ? success : accent) : .secondary).gradient)
                    .frame(width: geo.size.width * min(max(value, 0), 1))
                Rectangle()
                    .fill(accent)
                    .frame(width: 2)
                    .offset(x: max(0, min(geo.size.width - 2, geo.size.width * gateFraction - 1)))
            }
        }
        .frame(height: 7)
        .animation(.linear(duration: 0.06), value: value)
    }
}
