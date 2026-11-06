/*
 * ParkGuard ESP32 gate controller
 * ==============================
 *
 * Hardware role
 * -------------
 * The ESP32 is the real-time edge controller for the parking gate. It:
 *   1. Watches the approach ultrasonic sensor and asks the Python server to
 *      start a camera scan (GET /start_camera).
 *   2. Watches the parking-slot ultrasonic sensor before opening the gate.
 *   3. Drives the gate servo, the buzzer, and the 16x2 I2C LCD.
 *   4. Receives decisions from the Python server over local Wi-Fi:
 *        /authorized   -> check the slot and open the gate when possible
 *        /unauthorized -> show "Access Denied" (buzzer is driven separately)
 *        /buzzer       -> play the repeated-attempt alarm sequence
 *        /lockdown     -> keep the gate closed and show a lockdown screen
 *        /unlock       -> clear lockdown and return to standby
 *        /greet        -> show a personalised welcome message
 *        /servo        -> open or close the gate directly
 *        /status       -> JSON diagnostics for both ultrasonic sensors
 *
 * Safety notes
 * ------------
 * * The servo must use a separate 5 V supply with a common ground. Do not power
 *   a loaded servo from the ESP32 3.3 V pin.
 * * HC-SR04 ECHO pins are 5 V outputs. Use a voltage divider (or a level
 *   shifter) on ECHO before connecting it to the ESP32.
 * * The gate opens only when the parking-slot sensor reports free space.
 *
 * Wi-Fi / server credentials
 * --------------------------
 * Copy ``secrets.example.h`` to ``secrets.h`` in this folder and fill in your
 * own values. ``secrets.h`` is git-ignored. The placeholders below keep the
 * sketch compilable when no local secrets file exists.
 *
 * Required Arduino libraries
 * --------------------------
 * * ESP32Servo            (Kevin Harrington / John K. Bennett)
 * * LiquidCrystal_I2C     (Frank de Brabander)
 * * ESP32 board package   (WiFi, WebServer, HTTPClient, Wire are built in)
 */

#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ESP32Servo.h>

// ---------------------------------------------------------------------------
// Local secrets. Create secrets.h from secrets.example.h; never commit it.
// ---------------------------------------------------------------------------
#if __has_include("secrets.h")
#include "secrets.h"
#endif

#ifndef WIFI_SSID
#define WIFI_SSID "YOUR_WIFI_SSID"
#endif
#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#endif
#ifndef PARKGUARD_SERVER_IP
#define PARKGUARD_SERVER_IP "YOUR_PARKGUARD_SERVER_IP"
#endif
#ifndef PARKGUARD_SERVER_PORT
#define PARKGUARD_SERVER_PORT 5000
#endif

// ---------------------------------------------------------------------------
// Pin map (ESP32 DevKit v1 / 30-pin board)
// ---------------------------------------------------------------------------
// Ultrasonic sensor #1: vehicle approach trigger, facing the driveway.
// TRIG is an ESP32 output; ECHO is an input (use a divider if the sensor is 5 V).
constexpr uint8_t APPROACH_TRIG_PIN = 13;
constexpr uint8_t APPROACH_ECHO_PIN = 32;

// Ultrasonic sensor #2: parking-slot occupancy sensor, mounted over the space.
// TRIG is an ESP32 output; ECHO is an input (use a divider if the sensor is 5 V).
constexpr uint8_t PARKING_TRIG_PIN = 14;
constexpr uint8_t PARKING_ECHO_PIN = 25;

// Passive buzzer signal pin. The other buzzer lead goes to GND.
constexpr uint8_t BUZZER_PIN = 33;

// Gate servo signal pin. Servo V+ goes to an external 5 V supply.
// Servo GND and ESP32 GND must be connected together.
constexpr uint8_t SERVO_PIN = 12;
constexpr uint8_t GATE_CLOSED_ANGLE = 0;   // Gate physically closed
constexpr uint8_t GATE_OPEN_ANGLE = 90;    // Gate physically open

// I2C LCD (16 columns x 2 rows, backpack address 0x27 by default).
// The ESP32's default I2C pins are 21/22, so this project maps them explicitly.
constexpr uint8_t LCD_I2C_ADDRESS = 0x27;
constexpr uint8_t LCD_COLUMNS = 16;
constexpr uint8_t LCD_ROWS = 2;
constexpr uint8_t I2C_SDA_PIN = 26;
constexpr uint8_t I2C_SCL_PIN = 27;

// ---------------------------------------------------------------------------
// Behaviour tuning
// ---------------------------------------------------------------------------
// A car closer than this to sensor #1 is considered "arrived".
constexpr long APPROACH_THRESHOLD_CM = 15;

// A car closer than this to sensor #2 is considered parked, so the garage is full.
constexpr long PARKING_FULL_THRESHOLD_CM = 20;

// HC-SR04 timeout: 30 ms is about 5 m. Timeouts return NO_ECHO_DISTANCE_CM.
constexpr unsigned long ECHO_TIMEOUT_US = 30000;
constexpr long NO_ECHO_DISTANCE_CM = 999;

// Do not ask the Python server for another scan during this cool-down window.
constexpr unsigned long APPROACH_COOLDOWN_MS = 5000;

// Keep the gate open long enough for the vehicle to pass.
constexpr unsigned long GATE_OPEN_MS = 2500;

// Poll the ultrasonic sensors every 100 ms.
constexpr unsigned long SENSOR_POLL_MS = 100;

// Wi-Fi connection and HTTP client timeouts.
constexpr unsigned long WIFI_CONNECT_TIMEOUT_MS = 20000;
constexpr uint16_t HTTP_TIMEOUT_MS = 1500;

// ---------------------------------------------------------------------------
// Global objects and state
// ---------------------------------------------------------------------------
WebServer server(80);
LiquidCrystal_I2C lcd(LCD_I2C_ADDRESS, LCD_COLUMNS, LCD_ROWS);
Servo gateServo;

unsigned long lastTriggerTime = 0;
unsigned long lastSensorCheck = 0;
unsigned long servoCloseAt = 0;
bool servoIsOpen = false;
bool lockdown = false;

// ---------------------------------------------------------------------------
// Buzzer sequences
// ---------------------------------------------------------------------------
// Each step is (frequency in Hz, duration in ms). Frequency 0 means silence.
struct SeqStep {
    uint16_t frequency;
    uint16_t duration;
};

// "Garage full" notification: one long beep followed by a pause.
const SeqStep FULL_SEQ[] = {
    {1000, 300},
    {0, 1000},
};

// Repeated-attempt alarm: four short high beeps separated by short pauses.
// This is sent by the Python server through /buzzer, not by /unauthorized.
const SeqStep DENIED_SEQ[] = {
    {2500, 200}, {0, 150},
    {2500, 200}, {0, 150},
    {2500, 200}, {0, 150},
    {2500, 200}, {0, 650},
};

const SeqStep *seqSteps = nullptr;
uint8_t seqCount = 0;
uint8_t seqIndex = 0;
unsigned long stepStart = 0;
bool seqActive = false;

void applyStep() {
    if (seqSteps[seqIndex].frequency > 0) {
        tone(BUZZER_PIN, seqSteps[seqIndex].frequency);
    } else {
        noTone(BUZZER_PIN);
    }
}

void startSequence(const SeqStep *steps, uint8_t count) {
    if (steps == nullptr || count == 0) {
        return;
    }
    seqSteps = steps;
    seqCount = count;
    seqIndex = 0;
    seqActive = true;
    stepStart = millis();
    applyStep();
}

void updateSequence() {
    if (!seqActive) {
        return;
    }

    if (millis() - stepStart >= seqSteps[seqIndex].duration) {
        stepStart += seqSteps[seqIndex].duration;
        seqIndex++;

        if (seqIndex >= seqCount) {
            seqActive = false;
            noTone(BUZZER_PIN);
            // Do not overwrite a lockdown screen when the alarm finishes.
            if (lockdown) {
                lcd.clear();
                lcd.setCursor(0, 0);
                lcd.print("LOCKDOWN ACTIVE");
            } else {
                lcd.clear();
                lcd.setCursor(0, 0);
                lcd.print("System Ready...");
            }
            return;
        }
        applyStep();
    }
}

// ---------------------------------------------------------------------------
// LCD helpers
// ---------------------------------------------------------------------------
String fitToLcd(const String &value) {
    if (value.length() <= LCD_COLUMNS) {
        return value;
    }
    return value.substring(0, LCD_COLUMNS);
}

void showLines(const String &line1, const String &line2 = "") {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(fitToLcd(line1));
    if (line2.length() > 0) {
        lcd.setCursor(0, 1);
        lcd.print(fitToLcd(line2));
    }
}

void showReadyScreen() {
    if (lockdown) {
        showLines("LOCKDOWN ACTIVE", "Reset required");
    } else {
        showLines("System Ready...", "Waiting for car");
    }
}

// ---------------------------------------------------------------------------
// Ultrasonic sensors
// ---------------------------------------------------------------------------
long getDistanceCm(uint8_t trigPin, uint8_t echoPin) {
    // HC-SR04 sequence: 10 us trigger pulse, then measure the echo pulse width.
    digitalWrite(trigPin, LOW);
    delayMicroseconds(5);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    unsigned long duration = pulseIn(echoPin, HIGH, ECHO_TIMEOUT_US);
    if (duration == 0) {
        return NO_ECHO_DISTANCE_CM;
    }

    // Speed of sound is about 0.034 cm/us; divide by 2 for the round trip.
    return static_cast<long>(duration * 0.034 / 2);
}

bool parkingIsFull() {
    long distance = getDistanceCm(PARKING_TRIG_PIN, PARKING_ECHO_PIN);
    return distance > 0 && distance <= PARKING_FULL_THRESHOLD_CM;
}

// ---------------------------------------------------------------------------
// Gate control
// ---------------------------------------------------------------------------
void closeGate() {
    gateServo.write(GATE_CLOSED_ANGLE);
    servoIsOpen = false;
    servoCloseAt = 0;
}

// Open only when the parking slot is free. Returns true when the gate opened.
bool openGateIfSpaceAvailable() {
    if (parkingIsFull()) {
        showLines("Garage Full!", "Please Wait...");
        startSequence(FULL_SEQ, sizeof(FULL_SEQ) / sizeof(FULL_SEQ[0]));
        return false;
    }

    gateServo.write(GATE_OPEN_ANGLE);
    servoIsOpen = true;
    servoCloseAt = millis() + GATE_OPEN_MS;
    return true;
}

// ---------------------------------------------------------------------------
// Python server communication
// ---------------------------------------------------------------------------
bool triggerPythonScan() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[HTTP] Cannot trigger scan: Wi-Fi is not connected.");
        return false;
    }

    String url = String("http://") + PARKGUARD_SERVER_IP + ":" +
                 String(PARKGUARD_SERVER_PORT) + "/start_camera";

    WiFiClient client;
    HTTPClient http;
    if (!http.begin(client, url)) {
        Serial.println("[HTTP] Could not create the request.");
        return false;
    }

    http.setTimeout(HTTP_TIMEOUT_MS);
    int statusCode = http.GET();
    http.end();

    Serial.printf("[HTTP] GET %s -> %d\n", url.c_str(), statusCode);
    return statusCode > 0 && statusCode < 400;
}

// ---------------------------------------------------------------------------
// HTTP handlers called by the Python server
// ---------------------------------------------------------------------------
void handleAuthorized() {
    if (!openGateIfSpaceAvailable()) {
        server.send(200, "text/plain", "FULL");
        return;
    }

    showLines("WELCOME!", "Gate Open");
    server.send(200, "text/plain", "OPEN");
}

void handleUnauthorized() {
    // The first denied attempt only shows a message. The Python server calls
    // /buzzer on repeated attempts, matching the project security policy.
    showLines("Access Denied!", "Unauthorized!");
    server.send(200, "text/plain", "DENIED");
}

void handleBuzzer() {
    showLines("Security Alert", "Repeated attempt");
    startSequence(DENIED_SEQ, sizeof(DENIED_SEQ) / sizeof(DENIED_SEQ[0]));
    server.send(200, "text/plain", "BUZZER");
}

void handleLockdown() {
    lockdown = true;
    closeGate();
    showLines("LOCKDOWN ACTIVE", "Reset required");
    server.send(200, "text/plain", "LOCKED");
}

void handleUnlock() {
    lockdown = false;
    closeGate();
    showReadyScreen();
    server.send(200, "text/plain", "UNLOCKED");
}

void handleGreeting() {
    String name = server.arg("name");
    if (name.length() == 0) {
        name = server.arg("message");
    }
    if (name.length() == 0) {
        name = "Guest";
    }

    showLines("Welcome " + name, "Have a nice day");
    server.send(200, "text/plain", "GREETED");
}

void handleServo() {
    String action = server.arg("action");
    action.toLowerCase();

    if (action == "close") {
        closeGate();
        showReadyScreen();
        server.send(200, "text/plain", "CLOSED");
        return;
    }

    // Anything other than an explicit close is treated as an open request.
    if (!openGateIfSpaceAvailable()) {
        server.send(200, "text/plain", "FULL");
        return;
    }

    showLines("Gate Open", "Welcome");
    server.send(200, "text/plain", "OPEN");
}

void handleStatus() {
    long approachDistance = getDistanceCm(APPROACH_TRIG_PIN, APPROACH_ECHO_PIN);
    long parkingDistance = getDistanceCm(PARKING_TRIG_PIN, PARKING_ECHO_PIN);

    String json = String("{\"approach_cm\":") + approachDistance +
                  ",\"parking_cm\":" + parkingDistance +
                  ",\"parking_full\":" + (parkingIsFull() ? "true" : "false") +
                  ",\"gate_open\":" + (servoIsOpen ? "true" : "false") +
                  ",\"lockdown\":" + (lockdown ? "true" : "false") + "}";

    server.send(200, "application/json", json);
}

void handleRoot() {
    server.send(200, "text/plain", "ParkGuard ESP32 gate controller");
}

void handleNotFound() {
    server.send(404, "text/plain", "Not found");
}

// ---------------------------------------------------------------------------
// Arduino setup
// ---------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);

    // Configure the ultrasonic pins, buzzer, and servo.
    pinMode(APPROACH_TRIG_PIN, OUTPUT);
    pinMode(APPROACH_ECHO_PIN, INPUT);
    pinMode(PARKING_TRIG_PIN, OUTPUT);
    pinMode(PARKING_ECHO_PIN, INPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    noTone(BUZZER_PIN);

    gateServo.setPeriodHertz(50);            // Standard hobby-servo PWM frequency
    gateServo.attach(SERVO_PIN, 500, 2400);  // Pulse width range for SG90/MG996R-class servos
    gateServo.write(GATE_CLOSED_ANGLE);

    // Start the I2C bus on the explicitly documented LCD pins.
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    lcd.init();
    lcd.backlight();
    showLines("ParkGuard", "Booting...");

    // Connect to Wi-Fi without blocking forever if the router is unavailable.
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    showLines("Connecting WiFi", "");
    unsigned long wifiStartedAt = millis();
    while (WiFi.status() != WL_CONNECTED &&
           millis() - wifiStartedAt < WIFI_CONNECT_TIMEOUT_MS) {
        delay(250);
        Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.print("[WiFi] Connected. ESP32 IP: ");
        Serial.println(WiFi.localIP());
        showLines("ESP IP Address:", WiFi.localIP().toString());
        delay(2000);
    } else {
        Serial.println("[WiFi] Connection failed; running without the network.");
        showLines("WiFi FAILED", "Check secrets.h");
        delay(2000);
    }

    // Register every endpoint used by main.py.
    server.on("/", handleRoot);
    server.on("/authorized", handleAuthorized);
    server.on("/unauthorized", handleUnauthorized);
    server.on("/buzzer", handleBuzzer);
    server.on("/lockdown", handleLockdown);
    server.on("/unlock", handleUnlock);
    server.on("/greet", handleGreeting);
    server.on("/servo", handleServo);
    server.on("/status", handleStatus);
    server.onNotFound(handleNotFound);
    server.begin();

    showReadyScreen();
    Serial.println("[ParkGuard] ESP32 gate controller ready.");
}

// ---------------------------------------------------------------------------
// Arduino main loop
// ---------------------------------------------------------------------------
void loop() {
    server.handleClient();
    updateSequence();

    // Close the gate automatically after the crossing window.
    if (servoIsOpen && millis() >= servoCloseAt) {
        closeGate();
        showReadyScreen();
    }

    // Poll the approach sensor only when the system is idle and not locked.
    if (!lockdown && !seqActive && !servoIsOpen &&
        millis() - lastSensorCheck >= SENSOR_POLL_MS) {
        lastSensorCheck = millis();

        long carDistance = getDistanceCm(APPROACH_TRIG_PIN, APPROACH_ECHO_PIN);
        bool carArrived = carDistance > 0 && carDistance <= APPROACH_THRESHOLD_CM;

        if (carArrived && millis() - lastTriggerTime > APPROACH_COOLDOWN_MS) {
            lastTriggerTime = millis();
            showLines("Car Detected!", "Scanning Face...");

            if (!triggerPythonScan()) {
                showLines("Server Offline", "Check Wi-Fi");
            }
        }
    }
}
