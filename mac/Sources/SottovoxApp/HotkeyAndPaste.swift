import AppKit
import ApplicationServices
import Carbon.HIToolbox

@MainActor
enum PasteHelper {
    static var isAccessibilityTrusted: Bool {
        AXIsProcessTrusted()
    }

    static func promptAccessibility() {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        let options = [key: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)
    }

    static func openAccessibilitySettings() {
        promptAccessibility()
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            NSWorkspace.shared.open(url)
        }
    }

    static func copyToClipboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    /// Reads up to 4 characters immediately preceding the cursor in the frontmost app's focused field.
    private static func precedingCursorTail() -> String? {
        guard isAccessibilityTrusted else { return nil }
        if let frontApp = NSWorkspace.shared.frontmostApplication {
            let axApp = AXUIElementCreateApplication(frontApp.processIdentifier)
            _ = AXUIElementSetAttributeValue(axApp, "AXManualAccessibility" as CFString, kCFBooleanTrue)
        }
        let systemWide = AXUIElementCreateSystemWide()
        var focusedRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(systemWide, kAXFocusedUIElementAttribute as CFString, &focusedRef) == .success,
              let focusedRef else {
            return nil
        }
        let element = focusedRef as! AXUIElement

        var rangeRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, &rangeRef) == .success,
              let rangeRef else {
            return nil
        }
        var range = CFRange()
        guard AXValueGetValue(rangeRef as! AXValue, .cfRange, &range) else {
            return nil
        }
        guard range.location > 0 else { return "" }

        let lookback = min(4, range.location)
        var tailRange = CFRange(location: range.location - lookback, length: lookback)
        if let tailRangeVal = AXValueCreate(.cfRange, &tailRange) {
            var strRef: CFTypeRef?
            if AXUIElementCopyParameterizedAttributeValue(
                element,
                kAXStringForRangeParameterizedAttribute as CFString,
                tailRangeVal,
                &strRef
            ) == .success, let s = strRef as? String {
                return s
            }
        }

        var valueRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &valueRef) == .success,
           let full = valueRef as? String {
            let utf16 = full.utf16
            let endIdx = utf16.index(utf16.startIndex, offsetBy: min(range.location, utf16.count))
            let startOffset = max(0, range.location - lookback)
            let startIdx = utf16.index(utf16.startIndex, offsetBy: min(startOffset, utf16.count))
            return String(utf16[startIdx..<endIdx])
        }
        return nil
    }

    /// Adapts leading spacing and clause-continuation casing to the character(s) before the cursor.
    static func prepareTextForActiveField(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let tail = precedingCursorTail(), !tail.isEmpty else {
            return text
        }

        var adjusted = trimmed
        let nonWS = tail.trimmingCharacters(in: .whitespacesAndNewlines)
        if let lastNonWS = nonWS.last, (lastNonWS == "," || lastNonWS == ";") {
            adjusted = lowercaseLeadingWordIfClause(adjusted)
        }

        if let lastChar = tail.last, !lastChar.isWhitespace, !"( [{\"'\u{201C}\u{2018}".contains(lastChar) {
            return " " + adjusted
        }
        return adjusted
    }

    private static func lowercaseLeadingWordIfClause(_ text: String) -> String {
        guard let first = text.first, first.isUppercase else { return text }
        let firstWord = text.split(whereSeparator: { !$0.isLetter && $0 != "'" }).first.map(String.init) ?? ""
        if firstWord == "I" || firstWord.hasPrefix("I'") { return text }
        let upperLetters = firstWord.filter(\.isLetter)
        if upperLetters.count > 1 && upperLetters.allSatisfy(\.isUppercase) { return text }
        return first.lowercased() + text.dropFirst()
    }

    /// Pastes clipboard contents into the active frontmost application via Cmd+V.
    /// Adapts leading spacing and punctuation continuation to the cursor context,
    /// then uses an isolated `.privateState` event source after a brief clipboard-settle delay
    /// so held trigger modifiers (like Right Option) don't bleed into Cmd+V.
    @discardableResult
    static func pasteIntoActiveApp(_ text: String) -> Bool {
        guard isAccessibilityTrusted else {
            copyToClipboard(text)
            return false
        }
        let prepared = prepareTextForActiveField(text)
        copyToClipboard(prepared)
        let cmdKey = CGKeyCode(kVK_Command)
        let vKey = CGKeyCode(kVK_ANSI_V)
        guard let src = CGEventSource(stateID: .privateState),
              let cmdDown = CGEvent(keyboardEventSource: src, virtualKey: cmdKey, keyDown: true),
              let vDown = CGEvent(keyboardEventSource: src, virtualKey: vKey, keyDown: true),
              let vUp = CGEvent(keyboardEventSource: src, virtualKey: vKey, keyDown: false),
              let cmdUp = CGEvent(keyboardEventSource: src, virtualKey: cmdKey, keyDown: false) else {
            return false
        }
        cmdDown.flags = .maskCommand
        vDown.flags = .maskCommand
        vUp.flags = .maskCommand
        cmdUp.flags = []
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
            cmdDown.post(tap: .cghidEventTap)
            vDown.post(tap: .cghidEventTap)
            vUp.post(tap: .cghidEventTap)
            cmdUp.post(tap: .cghidEventTap)
        }
        return true
    }
}

@MainActor
final class PushToTalkMonitor {
    var onBegin: (() -> Void)?
    var onEnd: (() -> Void)?
    var allowLocalSpace: (() -> Bool)?
    var globalFnEnabled = false
    var globalRightOptionEnabled = true

    private var localMonitor: Any?
    private var globalFlagsMonitor: Any?
    private var resignObserver: NSObjectProtocol?
    private var spaceHeld = false
    private var fnHeld = false
    private var rightOptionHeld = false

    static var isInputMonitoringGranted: Bool {
        CGPreflightListenEventAccess()
    }

    static func requestInputMonitoring() {
        _ = CGRequestListenEventAccess()
    }

    static func openInputMonitoringSettings() {
        requestInputMonitoring()
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent") {
            NSWorkspace.shared.open(url)
        }
    }

    func start() {
        guard localMonitor == nil else { return }

        // In-app Spacebar and Fn / Right Option hold while Hush is focused.
        localMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp, .flagsChanged]) { [weak self] event in
            if event.type == .flagsChanged {
                let code = event.keyCode
                let flags = event.modifierFlags
                Task { @MainActor [weak self] in
                    self?.handleFlagsChanged(keyCode: code, flags: flags)
                }
                return event
            }
            guard event.keyCode == 49 else { return event }
            let isDown = event.type == .keyDown
            let isRepeat = isDown && event.isARepeat
            let handled = MainActor.assumeIsolated { () -> Bool in
                guard let self else { return false }
                if !isDown {
                    guard self.spaceHeld else { return false }
                    self.spaceHeld = false
                    self.onEnd?()
                    return true
                }
                if self.spaceHeld || isRepeat {
                    return self.spaceHeld
                }
                guard self.allowLocalSpace?() == true else { return false }
                if NSApp.keyWindow?.firstResponder is NSText { return false }
                self.spaceHeld = true
                self.onBegin?()
                return true
            }
            return handled ? nil : event
        }

        // Release any active space hold if the app loses focus before keyUp arrives.
        resignObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didResignActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if self.spaceHeld {
                    self.spaceHeld = false
                    self.onEnd?()
                }
            }
        }

        // Global Fn / Globe (keyCode 63) or Right Option (keyCode 61) hold across any app.
        globalFlagsMonitor = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            let code = event.keyCode
            let flags = event.modifierFlags
            Task { @MainActor [weak self] in
                self?.handleFlagsChanged(keyCode: code, flags: flags)
            }
        }
    }

    private func handleFlagsChanged(keyCode: UInt16, flags: NSEvent.ModifierFlags) {
        if keyCode == 63 && globalFnEnabled {
            let fnDown = flags.contains(.function)
            if fnDown && !fnHeld {
                fnHeld = true
                onBegin?()
            } else if !fnDown && fnHeld {
                fnHeld = false
                onEnd?()
            }
        } else if keyCode == 61 && globalRightOptionEnabled {
            let optionDown = flags.contains(.option)
            if optionDown && !rightOptionHeld {
                rightOptionHeld = true
                onBegin?()
            } else if !optionDown && rightOptionHeld {
                rightOptionHeld = false
                onEnd?()
            }
        }
    }

    func stop() {
        if let localMonitor { NSEvent.removeMonitor(localMonitor) }
        if let globalFlagsMonitor { NSEvent.removeMonitor(globalFlagsMonitor) }
        if let resignObserver { NotificationCenter.default.removeObserver(resignObserver) }
        localMonitor = nil
        globalFlagsMonitor = nil
        resignObserver = nil
        spaceHeld = false
        fnHeld = false
        rightOptionHeld = false
    }
}
