
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.farshekashan.keyfarsi"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.smhkeyboard.ia"
        minSdk = 23
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    // Keep ONNX files uncompressed inside the APK.
    androidResources {
        noCompress += "onnx"
    }
}

// Remove obsolete Kotlin compatibility modules.
// Their classes are already included in kotlin-stdlib.
configurations.configureEach {
    exclude(
        group = "org.jetbrains.kotlin",
        module = "kotlin-stdlib-jdk7"
    )

    exclude(
        group = "org.jetbrains.kotlin",
        module = "kotlin-stdlib-jdk8"
    )
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")

    implementation("androidx.activity:activity-ktx:1.9.3")

    implementation("androidx.emoji2:emoji2:1.5.0")

    implementation("androidx.emoji2:emoji2-views-helper:1.5.0")

    // ONNX Runtime for Android
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
}
