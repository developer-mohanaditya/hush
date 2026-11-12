// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Sottovox",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "Sottovox", targets: ["SottovoxApp"])
    ],
    targets: [
        .executableTarget(
            name: "SottovoxApp",
            path: "Sources/SottovoxApp"
        )
    ]
)
