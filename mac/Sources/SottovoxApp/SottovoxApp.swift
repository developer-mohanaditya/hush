import AppKit
import SwiftUI

@MainActor
enum WindowRouter {
    static func showStudio(openWindow: OpenWindowAction?) {
        NSApp.setActivationPolicy(.regular)
        if let studio = NSApp.windows.first(where: { win in
            let isPanel = win is NSPanel
            let id = win.identifier?.rawValue ?? ""
            return !isPanel && (id.contains("studio") || win.title == "Hush")
        }) {
            if studio.isMiniaturized {
                studio.deminiaturize(nil)
            }
            studio.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }
        openWindow?(id: "studio")
        NSApp.activate(ignoringOtherApps: true)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var state: AppState?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        MainActor.assumeIsolated {
            WindowRouter.showStudio(openWindow: nil)
        }
        return true
    }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated {
            state?.hotkeys.stop()
            state?.engine.shutdown()
        }
    }
}

@main
struct SottovoxApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @Environment(\.openWindow) private var openWindow
    @State private var state = AppState()
    @State private var pillController = FloatingPillController()

    var body: some Scene {
        Window("Hush", id: "studio") {
            ContentView()
                .environment(state)
                .onAppear {
                    delegate.state = state
                    state.pillController = pillController
                    pillController.attach(state: state) {
                        WindowRouter.showStudio(openWindow: openWindow)
                    }
                }
        }
        .windowResizability(.contentMinSize)

        MenuBarExtra("Hush", systemImage: state.audio.isRecording ? "waveform.circle.fill" : "waveform") {
            MenuBarPopoverView()
                .environment(state)
                .onAppear {
                    delegate.state = state
                    state.pillController = pillController
                    pillController.attach(state: state) {
                        WindowRouter.showStudio(openWindow: openWindow)
                    }
                }
        }
        .menuBarExtraStyle(.window)
    }
}
