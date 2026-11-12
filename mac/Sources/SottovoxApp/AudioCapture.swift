@preconcurrency import AVFoundation
import Foundation
import Observation

struct AudioInputDevice: Identifiable, Hashable {
    let id: String
    let name: String
}

enum CaptureDevices {
    static func inputDevices() -> [AudioInputDevice] {
        let session = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.microphone, .external],
            mediaType: .audio,
            position: .unspecified
        )
        return session.devices.map { AudioInputDevice(id: $0.uniqueID, name: $0.localizedName) }
    }

    static func resolveDevice(uniqueID: String?) -> AVCaptureDevice? {
        if let uniqueID {
            guard let dev = AVCaptureDevice(uniqueID: uniqueID), dev.isConnected, !dev.isSuspended else {
                return nil
            }
            return dev
        }
        return AVCaptureDevice.default(for: .audio)
    }
}

private final class CaptureWriter: NSObject, AVCaptureAudioDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let url: URL
    private let lock = NSLock()
    private var file: AVAudioFile?
    private var wroteFrames = false
    private var lastMeterUpdate = ContinuousClock.now - .milliseconds(100)
    private let onStats: @Sendable (Double, Bool) -> Void

    init(url: URL, onStats: @escaping @Sendable (Double, Bool) -> Void) {
        self.url = url
        self.onStats = onStats
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let numSamples = CMSampleBufferGetNumSamples(sampleBuffer)
        guard numSamples > 0,
              let fmtDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(fmtDesc)?.pointee,
              asbd.mSampleRate > 0,
              asbd.mChannelsPerFrame > 0,
              let avFmt = AVAudioFormat(
                  commonFormat: .pcmFormatFloat32,
                  sampleRate: asbd.mSampleRate,
                  channels: asbd.mChannelsPerFrame,
                  interleaved: false
              ),
              let pcm = AVAudioPCMBuffer(pcmFormat: avFmt, frameCapacity: AVAudioFrameCount(numSamples)) else {
            return
        }
        pcm.frameLength = AVAudioFrameCount(numSamples)

        let copyStatus = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(numSamples),
            into: pcm.mutableAudioBufferList
        )
        guard copyStatus == noErr else { return }

        let stats = Self.measure(pcm)
        lock.withLock {
            if file == nil {
                try? FileManager.default.removeItem(at: url)
                file = try? AVAudioFile(
                    forWriting: url,
                    settings: avFmt.settings,
                    commonFormat: .pcmFormatFloat32,
                    interleaved: false
                )
            }
            if let file {
                do {
                    try file.write(from: pcm)
                    wroteFrames = true
                } catch {
                    // Skip failed buffer write; finalized status checks wroteFrames.
                }
            }
        }

        let now = ContinuousClock.now
        if now - lastMeterUpdate >= .milliseconds(45) {
            lastMeterUpdate = now
            onStats(stats.dbfs, stats.clipped)
        }
    }

    @discardableResult
    func close() -> Bool {
        lock.withLock {
            file = nil
            return wroteFrames
        }
    }

    private static func measure(_ buffer: AVAudioPCMBuffer) -> (dbfs: Double, clipped: Bool) {
        guard let samples = buffer.floatChannelData?[0], buffer.frameLength > 0 else {
            return (-90.0, false)
        }
        let n = Int(buffer.frameLength)
        var sum: Double = 0
        var peak: Float = 0
        for i in 0..<n {
            let s = samples[i]
            sum += Double(s * s)
            let a = abs(s)
            if a > peak { peak = a }
        }
        let rms = (sum / Double(n)).squareRoot()
        let db = 20.0 * log10(max(rms, 1e-6))
        return (db, peak >= 0.95)
    }
}

@MainActor @Observable
final class AudioCaptureService: NSObject, AVAudioPlayerDelegate {
    private(set) var devices: [AudioInputDevice] = []
    var selectedDeviceID: String?
    private(set) var activeDeviceName: String = "System default"
    private(set) var isStarting = false
    private(set) var isRecording = false
    private(set) var levelDBFS: Double = -90.0
    private(set) var plosiveWarning = false
    private(set) var elapsedSeconds: TimeInterval = 0
    private(set) var lastRawURL: URL?
    private(set) var lastBoostedURL: URL?
    private(set) var playingKind: String?
    private(set) var error: String?

    @ObservationIgnored private let captureQueue = DispatchQueue(
        label: "com.aditya.sottovox.capture",
        qos: .userInitiated
    )
    @ObservationIgnored private var session: AVCaptureSession?
    @ObservationIgnored private var writer: CaptureWriter?
    @ObservationIgnored private var activeRecordingURL: URL?
    @ObservationIgnored private var recordingStart: Date?
    @ObservationIgnored private var timer: Timer?
    @ObservationIgnored private var player: AVAudioPlayer?
    @ObservationIgnored private var runtimeErrorObserver: NSObjectProtocol?
    @ObservationIgnored private var deviceObservers: [NSObjectProtocol] = []

    private let tempDir: URL = {
        let dir = FileManager.default.temporaryDirectory.appending(path: "SottovoxClips", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    override init() {
        super.init()
        let nc = NotificationCenter.default
        for name in [AVCaptureDevice.wasConnectedNotification, AVCaptureDevice.wasDisconnectedNotification] {
            let token = nc.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.refreshDevices()
                }
            }
            deviceObservers.append(token)
        }
    }

    var normalizedMeter: Double {
        // Map -75...0 dBFS onto 0...1 so ultra-quiet -65..-45 dBFS whispers and noise-gate thresholds align visually.
        max(0, min(1, (levelDBFS + 75.0) / 75.0))
    }

    var targetDeviceLabel: String {
        if let selectedDeviceID, let dev = devices.first(where: { $0.id == selectedDeviceID }) {
            return dev.name
        }
        return CaptureDevices.resolveDevice(uniqueID: nil)?.localizedName ?? "System default"
    }

    func refreshDevices() {
        devices = CaptureDevices.inputDevices()
        if let selectedDeviceID, !devices.contains(where: { $0.id == selectedDeviceID }) {
            self.selectedDeviceID = nil
        }
    }

    func startRecording() async -> Bool {
        guard !isRecording, !isStarting else { return false }
        isStarting = true
        error = nil
        stopPlayback()

        let authStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        let granted: Bool
        switch authStatus {
        case .authorized:
            granted = true
        case .notDetermined:
            granted = await AVCaptureDevice.requestAccess(for: .audio)
        default:
            granted = false
        }
        guard granted else {
            isStarting = false
            error = "Microphone access denied. Enable Sottovox in System Settings › Privacy & Security › Microphone."
            return false
        }

        let targetUID = selectedDeviceID
        let takeID = UUID().uuidString
        let rawURL = tempDir.appending(path: "take_\(takeID)_raw.wav")
        let queue = captureQueue

        let result: Result<(session: AVCaptureSession, writer: CaptureWriter, deviceName: String), CaptureStartError> =
            await withCheckedContinuation { [weak self] continuation in
                queue.async { [weak self] in
                    try? FileManager.default.removeItem(at: rawURL)
                    guard let device = CaptureDevices.resolveDevice(uniqueID: targetUID) else {
                        continuation.resume(
                            returning: .failure(
                                CaptureStartError(message: "Selected microphone is unavailable or disconnected.")
                            )
                        )
                        return
                    }

                    let session = AVCaptureSession()
                    let input: AVCaptureDeviceInput
                    do {
                        input = try AVCaptureDeviceInput(device: device)
                    } catch {
                        continuation.resume(
                            returning: .failure(
                                CaptureStartError(message: "Could not open mic '\(device.localizedName)': \(error.localizedDescription)")
                            )
                        )
                        return
                    }

                    let writer = CaptureWriter(url: rawURL) { [weak self] db, clipped in
                        Task { @MainActor [weak self] in
                            guard let self, self.isRecording else { return }
                            self.levelDBFS = self.levelDBFS * 0.45 + db * 0.55
                            if clipped { self.plosiveWarning = true }
                        }
                    }

                    let output = AVCaptureAudioDataOutput()
                    output.audioSettings = [
                        AVFormatIDKey: kAudioFormatLinearPCM,
                        AVLinearPCMBitDepthKey: 32,
                        AVLinearPCMIsFloatKey: true,
                        AVLinearPCMIsBigEndianKey: false,
                        AVLinearPCMIsNonInterleaved: true
                    ]
                    output.setSampleBufferDelegate(writer, queue: queue)

                    session.beginConfiguration()
                    guard session.canAddInput(input), session.canAddOutput(output) else {
                        session.commitConfiguration()
                        continuation.resume(
                            returning: .failure(
                                CaptureStartError(message: "Unable to attach audio capture pipeline for '\(device.localizedName)'.")
                            )
                        )
                        return
                    }
                    session.addInput(input)
                    session.addOutput(output)
                    session.commitConfiguration()

                    // Run session startup off @MainActor so Bluetooth HFP negotiation never stalls UI responsiveness.
                    session.startRunning()
                    guard session.isRunning else {
                        continuation.resume(
                            returning: .failure(
                                CaptureStartError(message: "Audio capture did not start for '\(device.localizedName)'.")
                            )
                        )
                        return
                    }

                    continuation.resume(returning: .success((session, writer, device.localizedName)))
                }
            }

        isStarting = false
        switch result {
        case .failure(let err):
            error = err.message
            refreshDevices()
            return false
        case .success(let started):
            self.session = started.session
            self.writer = started.writer
            self.activeRecordingURL = rawURL
            self.activeDeviceName = started.deviceName
            self.plosiveWarning = false
            self.levelDBFS = -70.0
            self.elapsedSeconds = 0
            self.recordingStart = Date()
            self.isRecording = true

            if let runtimeErrorObserver {
                NotificationCenter.default.removeObserver(runtimeErrorObserver)
            }
            runtimeErrorObserver = NotificationCenter.default.addObserver(
                forName: AVCaptureSession.runtimeErrorNotification,
                object: started.session,
                queue: .main
            ) { [weak self] note in
                let err = note.userInfo?[AVCaptureSessionErrorKey] as? NSError
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.error = err?.localizedDescription ?? "Audio capture interrupted."
                }
            }

            timer?.invalidate()
            timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self, let start = self.recordingStart else { return }
                    self.elapsedSeconds = Date().timeIntervalSince(start)
                }
            }
            return true
        }
    }

    func stopRecording() async -> URL? {
        guard isRecording else { return nil }
        timer?.invalidate()
        timer = nil
        isRecording = false
        levelDBFS = -90.0

        if let runtimeErrorObserver {
            NotificationCenter.default.removeObserver(runtimeErrorObserver)
            self.runtimeErrorObserver = nil
        }

        let activeSession = session
        let activeWriter = writer
        let url = activeRecordingURL
        session = nil
        writer = nil
        activeRecordingURL = nil

        let queue = captureQueue
        let wroteAudio: Bool = await withCheckedContinuation { continuation in
            queue.async {
                activeSession?.stopRunning()
                let ok = activeWriter?.close() ?? false
                continuation.resume(returning: ok)
            }
        }

        guard wroteAudio, let url, FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        lastRawURL = url
        return url
    }

    @discardableResult
    func saveBoostedWav(base64: String?, matchingRawURL rawURL: URL) -> URL? {
        guard let base64, let data = Data(base64Encoded: base64) else {
            lastBoostedURL = nil
            return nil
        }
        let stem = rawURL.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "_raw", with: "")
        let boostedURL = tempDir.appending(path: "\(stem)_boosted.wav")
        do {
            try data.write(to: boostedURL, options: .atomic)
            lastBoostedURL = boostedURL
            return boostedURL
        } catch {
            lastBoostedURL = nil
            return nil
        }
    }

    func togglePlay(kind: String, url: URL?) {
        if playingKind == kind {
            stopPlayback()
            return
        }
        guard let url else { return }
        stopPlayback()
        do {
            let p = try AVAudioPlayer(contentsOf: url)
            p.delegate = self
            p.play()
            player = p
            playingKind = kind
        } catch {
            self.error = "Playback failed: \(error.localizedDescription)"
        }
    }

    func stopPlayback() {
        player?.stop()
        player = nil
        playingKind = nil
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor [weak self] in
            self?.playingKind = nil
            self?.player = nil
        }
    }
}

private struct CaptureStartError: Error {
    let message: String
}
