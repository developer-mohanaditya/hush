import Foundation
import Observation
import Security

enum KeychainHelper {
    private static let service = "com.aditya.hush.customllm"
    private static let account = "custom_api_key"

    static func saveAPIKey(_ value: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8) else { return }
        var insert = query
        insert[kSecValueData as String] = data
        SecItemAdd(insert as CFDictionary, nil)
    }

    static func loadAPIKey() -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess,
              let data = result as? Data,
              let key = String(data: data, encoding: .utf8) else {
            return ""
        }
        return key
    }
}

struct LLMModelsResponse: Codable {
    var ok: Bool
    var models: [String]
    var error: String?
}

struct ASRModelInfo: Codable, Identifiable, Hashable {
    var engine: String
    var id: String
    var cached: Bool
}

struct HealthResponse: Codable {
    var ok: Bool
    var warmed: Bool
    var warm_error: String?
    var default_engine: String
    var default_model: String
    var asr_models: [ASRModelInfo]
    var ollama_models: [String]
    var default_ollama_model: String?
}

struct TranscribeStats: Codable, Hashable {
    var duration_sec: Double
    var trimmed_duration_sec: Double
    var sample_rate: Int
    var raw_rms_dbfs: Double
    var raw_peak_dbfs: Double
    var processed_rms_dbfs: Double
    var applied_gain_db: Double
    var clipped_ratio: Double
    var too_quiet: Bool
}

struct TranscribeTimings: Codable, Hashable {
    var dsp: Double
    var asr: Double
    var cleanup: Double
}

struct TranscribeResponse: Codable {
    var raw_text: String
    var clean_text: String
    var stats: TranscribeStats
    var engine: String
    var model: String
    var language: String?
    var cleanup_applied: Bool
    var cleanup_model: String?
    var cleanup_error: String?
    var timings_ms: TranscribeTimings
    var boosted_wav_b64: String?
}

struct TranscribeOptions: Sendable {
    var dspEnabled: Bool = true
    var highpassHz: Double = 95
    var targetRmsDbfs: Double = -18
    var maxGainDb: Double = 32
    var silenceGateDbfs: Double = -65
    var engine: String = "sherpa-parakeet"
    var model: String = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    var language: String = "en"
    var whisperMode: Bool = true
    var cleanupEnabled: Bool = true
    var cleanupMode: String = "rules"
    var ollamaModel: String?
    var customBaseURL: String = UserDefaults.standard.string(forKey: "hush.customBaseURL") ?? "https://api.openai.com/v1"
    var customApiKey: String = KeychainHelper.loadAPIKey()
    var customModel: String = UserDefaults.standard.string(forKey: "hush.customModel") ?? ""
}

@MainActor @Observable
final class EngineClient {
    enum Status: Equatable {
        case offline
        case starting
        case warming
        case ready
        case failed(String)

        var label: String {
            switch self {
            case .offline: "Engine offline"
            case .starting: "Starting local engine…"
            case .warming: "Loading local Parakeet INT8…"
            case .ready: "Local · Ready"
            case .failed(let msg): msg
            }
        }
    }

    let baseURL = URL(string: "http://127.0.0.1:8765")!
    private(set) var status: Status = .offline
    private(set) var asrModels: [ASRModelInfo] = []
    private(set) var ollamaModels: [String] = []
    private(set) var defaultOllamaModel: String?

    @ObservationIgnored private var managedProcess: Process?
    @ObservationIgnored private let session: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 120
        return URLSession(configuration: cfg)
    }()

    func ensureStarted() async {
        if await refreshHealth() {
            if status != .ready { await warmDefault() }
            return
        }

        status = .starting
        var serverBin = resolveServerExecutable()
        if serverBin == nil {
            serverBin = await bootstrapBundledEngineIfNeeded()
        }
        guard let serverBin else {
            status = .failed("Local engine not installed. Install uv or Python 3.12 (`brew install uv python@3.12`) and relaunch Hush.")
            return
        }

        let proc = Process()
        proc.executableURL = serverBin
        proc.arguments = ["--host", "127.0.0.1", "--port", "8765"]
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
            managedProcess = proc
        } catch {
            status = .failed("Failed to launch engine: \(error.localizedDescription)")
            return
        }

        for _ in 0..<30 {
            try? await Task.sleep(for: .milliseconds(250))
            if await refreshHealth() {
                await warmDefault()
                return
            }
        }
        status = .failed("Local engine timed out starting on port 8765.")
    }

    func shutdown() {
        if let proc = managedProcess, proc.isRunning {
            proc.terminate()
        }
        managedProcess = nil
    }

    @discardableResult
    func refreshHealth() async -> Bool {
        do {
            let (data, resp) = try await session.data(from: baseURL.appending(path: "health"))
            guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else { return false }
            let health = try JSONDecoder().decode(HealthResponse.self, from: data)
            asrModels = health.asr_models
            ollamaModels = health.ollama_models
            defaultOllamaModel = health.default_ollama_model
            status = health.warmed ? .ready : .warming
            return true
        } catch {
            return false
        }
    }

    func warmDefault() async {
        await warm(engine: "sherpa-parakeet", model: "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
    }

    func warm(engine: String, model: String) async {
        status = .warming
        var req = URLRequest(url: baseURL.appending(path: "warm"))
        req.httpMethod = "POST"
        req.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        let bodyStr = "engine=\(engine)&model=\(model)"
        req.httpBody = Data(bodyStr.utf8)
        do {
            let (_, resp) = try await session.data(for: req)
            if let http = resp as? HTTPURLResponse, http.statusCode == 200 {
                status = .ready
            }
        } catch {
            status = .failed("Model warm failed: \(error.localizedDescription)")
        }
    }

    func transcribe(fileURL: URL, options: TranscribeOptions) async throws -> TranscribeResponse {
        let wavData = try Data(contentsOf: fileURL)
        let boundary = "Sottovox-\(UUID().uuidString)"
        var body = Data()

        func addField(_ name: String, _ value: String) {
            body.append(Data("--\(boundary)\r\n".utf8))
            body.append(Data("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".utf8))
            body.append(Data("\(value)\r\n".utf8))
        }

        addField("dsp_enabled", options.dspEnabled ? "true" : "false")
        addField("highpass_hz", String(format: "%.1f", options.highpassHz))
        addField("target_rms_dbfs", String(format: "%.1f", options.targetRmsDbfs))
        addField("max_gain_db", String(format: "%.1f", options.maxGainDb))
        addField("silence_gate_dbfs", String(format: "%.1f", options.silenceGateDbfs))
        addField("trim_silence", "true")
        addField("engine", options.engine)
        addField("model", options.model)
        addField("language", options.language)
        addField("whisper_mode", options.whisperMode ? "true" : "false")
        addField("cleanup", options.cleanupEnabled ? "true" : "false")
        addField("cleanup_mode", options.cleanupMode)
        if let ollama = options.ollamaModel, !ollama.isEmpty {
            addField("ollama_model", ollama)
        }
        if options.cleanupMode == "custom_openai" {
            addField("custom_base_url", options.customBaseURL)
            addField("custom_api_key", options.customApiKey)
            addField("custom_model", options.customModel)
        }
        addField("return_wav", "true")

        body.append(Data("--\(boundary)\r\n".utf8))
        body.append(Data("Content-Disposition: form-data; name=\"audio\"; filename=\"clip.wav\"\r\n".utf8))
        body.append(Data("Content-Type: audio/wav\r\n\r\n".utf8))
        body.append(wavData)
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))

        var req = URLRequest(url: baseURL.appending(path: "transcribe"))
        req.httpMethod = "POST"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        req.httpBody = body

        let (data, resp) = try await session.data(for: req)
        guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
            let msg = String(data: data, encoding: .utf8) ?? "HTTP error"
            throw NSError(domain: "Sottovox", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
        }
        return try JSONDecoder().decode(TranscribeResponse.self, from: data)
    }

    func fetchCustomModels(baseURL customBase: String, apiKey: String) async -> (models: [String], error: String?) {
        var req = URLRequest(url: baseURL.appending(path: "llm/models"))
        req.httpMethod = "POST"
        var comps = URLComponents()
        comps.queryItems = [
            URLQueryItem(name: "base_url", value: customBase),
            URLQueryItem(name: "api_key", value: apiKey),
        ]
        req.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        req.httpBody = Data((comps.percentEncodedQuery ?? "").utf8)
        do {
            let (data, _) = try await session.data(for: req)
            let decoded = try JSONDecoder().decode(LLMModelsResponse.self, from: data)
            return (decoded.models, decoded.error)
        } catch {
            return ([], error.localizedDescription)
        }
    }

    private func resolveServerExecutable() -> URL? {
        var candidates: [URL] = []
        let bundleDir = Bundle.main.bundleURL.deletingLastPathComponent()
        candidates.append(bundleDir.deletingLastPathComponent().appending(path: "engine/.venv/bin/sottovox-server"))
        candidates.append(URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appending(path: "engine/.venv/bin/sottovox-server"))
        if let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            candidates.append(appSupport.appending(path: "Hush/venv/bin/sottovox-server"))
        }
        candidates.append(URL(fileURLWithPath: "/Volumes/aditya/hush/engine/.venv/bin/sottovox-server"))
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    private func bootstrapBundledEngineIfNeeded() async -> URL? {
        guard let engineSrc = Bundle.main.resourceURL?.appending(path: "engine"),
              FileManager.default.fileExists(atPath: engineSrc.appending(path: "pyproject.toml").path),
              let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return nil
        }

        let hushSupport = appSupport.appending(path: "Hush")
        let venvDir = hushSupport.appending(path: "venv")
        let serverBin = venvDir.appending(path: "bin/sottovox-server")
        if FileManager.default.isExecutableFile(atPath: serverBin.path) {
            return serverBin
        }

        return await Task.detached(priority: .userInitiated) {
            try? FileManager.default.createDirectory(at: hushSupport, withIntermediateDirectories: true)
            let uvCandidates = [
                "/opt/homebrew/bin/uv",
                "/usr/local/bin/uv",
                NSHomeDirectory() + "/.local/bin/uv",
                NSHomeDirectory() + "/.cargo/bin/uv",
            ]
            if let uvPath = uvCandidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) {
                let venvProc = Process()
                venvProc.executableURL = URL(fileURLWithPath: uvPath)
                venvProc.arguments = ["venv", "--python", "3.12", venvDir.path]
                try? venvProc.run()
                venvProc.waitUntilExit()
                if venvProc.terminationStatus == 0 {
                    let installProc = Process()
                    installProc.executableURL = URL(fileURLWithPath: uvPath)
                    installProc.arguments = ["pip", "install", "--python", venvDir.appending(path: "bin/python").path, engineSrc.path]
                    try? installProc.run()
                    installProc.waitUntilExit()
                    if installProc.terminationStatus == 0,
                       FileManager.default.isExecutableFile(atPath: serverBin.path) {
                        return serverBin
                    }
                }
            }

            let pyCandidates = [
                "/opt/homebrew/bin/python3.12",
                "/opt/homebrew/bin/python3.11",
                "/opt/homebrew/bin/python3",
                "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
                "/usr/local/bin/python3",
                "/usr/bin/python3",
            ]
            if let pyPath = pyCandidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) {
                let venvProc = Process()
                venvProc.executableURL = URL(fileURLWithPath: pyPath)
                venvProc.arguments = ["-m", "venv", venvDir.path]
                try? venvProc.run()
                venvProc.waitUntilExit()
                if venvProc.terminationStatus == 0 {
                    let installProc = Process()
                    installProc.executableURL = venvDir.appending(path: "bin/pip")
                    installProc.arguments = ["install", engineSrc.path]
                    try? installProc.run()
                    installProc.waitUntilExit()
                    if installProc.terminationStatus == 0,
                       FileManager.default.isExecutableFile(atPath: serverBin.path) {
                        return serverBin
                    }
                }
            }
            return nil
        }.value
    }
}
