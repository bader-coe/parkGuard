# ParkGuard wiring guide

This document describes the reference wiring used by
`firmware/parkguard_esp32/parkguard_esp32.ino`. The pin constants in the sketch
are the single source of truth; update both the sketch and this table if you
change the hardware.

> **Before powering up:** disconnect the servo and sensors, verify the ESP32
> boots, then connect one subsystem at a time. Never connect a 5 V signal
> directly to an ESP32 input.

---

## Bill of materials

| Qty | Part | Notes |
|---:|---|---|
| 1 | ESP32 DevKit v1 (30-pin) | 3.3 V logic |
| 1 | USB webcam | Connected to the Python host, not the ESP32 |
| 2 | HC-SR04 ultrasonic sensor | One approach, one parking-slot |
| 1 | SG90 or MG996R servo | Gate arm |
| 1 | Passive buzzer | 3.3 V compatible |
| 1 | 16×2 I2C LCD | PCF8574 backpack, address `0x27` or `0x3F` |
| 1 | 5 V regulated supply | Servo + sensors; at least 2 A for MG996R |
| 2 | 1 kΩ + 2 kΩ resistors | ECHO voltage dividers |
| — | Jumper wires, breadboard | — |

---

## Pin map

| ESP32 GPIO | Connected to | Mode | Comment |
|---:|---|---|---|
| 13 | Approach HC-SR04 `TRIG` | Output | 10 µs trigger pulse |
| 32 | Approach HC-SR04 `ECHO` | Input | Through a 5 V → 3.3 V divider |
| 14 | Parking HC-SR04 `TRIG` | Output | 10 µs trigger pulse |
| 25 | Parking HC-SR04 `ECHO` | Input | Through a 5 V → 3.3 V divider |
| 33 | Buzzer `+` | Output | Passive buzzer; `-` to GND |
| 12 | Servo signal (orange/yellow) | Output | 50 Hz PWM; external 5 V power |
| 26 | LCD `SDA` | I2C data | `Wire.begin(26, 27)` |
| 27 | LCD `SCL` | I2C clock | `Wire.begin(26, 27)` |
| GND | All grounds | — | ESP32, servo supply, sensors, LCD, buzzer |

GPIO 12 is an ESP32 strapping pin. If the board fails to boot with the servo
attached, move the servo signal to another free output (for example GPIO 15) and
change `SERVO_PIN` in the firmware.

---

## Ultrasonic sensors (HC-SR04)

Each sensor has four wires:

| HC-SR04 | ESP32 |
|---|---|
| `VCC` | 5 V |
| `GND` | GND |
| `TRIG` | GPIO 13 (approach) or GPIO 14 (parking) |
| `ECHO` | GPIO 32 (approach) or GPIO 25 (parking), via divider |

### ECHO voltage divider

The HC-SR04 `ECHO` output is 5 V, but the ESP32 input is 3.3 V. Use a divider:

```text
HC-SR04 ECHO ──[ 1 kΩ ]──┬── ESP32 GPIO
                          │
                       [ 2 kΩ ]
                          │
                         GND
```

This produces approximately `5 V × 2 / (1 + 2) = 3.33 V`. A ready-made
bidirectional level shifter is also acceptable.

### Mounting

- **Approach sensor:** faces the driveway, roughly 30–60 cm above the ground.
  A car within `APPROACH_THRESHOLD_CM` (15 cm) triggers a scan.
- **Parking sensor:** points down at the parking space. A reading within
  `PARKING_FULL_THRESHOLD_CM` (20 cm) means the garage is full.

Tune both thresholds in the firmware after mounting. Keep the sensors away from
direct sunlight, rain, and moving gate parts.

---

## Servo gate

| Servo wire | Connection |
|---|---|
| Brown / black | Common ground |
| Red | External 5 V supply |
| Orange / yellow / white | ESP32 GPIO 12 |

- Use a separate 5 V supply; do not power the servo from the ESP32.
- Connect the supply ground to the ESP32 ground.
- Add a 470–1000 µF capacitor across the servo supply if the board resets when
  the servo moves.
- `GATE_CLOSED_ANGLE = 0` and `GATE_OPEN_ANGLE = 90` are defined in the sketch.
  Adjust them for your gate mechanism and re-test with the arm disconnected.

---

## I2C LCD

| LCD backpack | ESP32 |
|---|---|
| `VCC` | 5 V (most backpacks are 5 V) |
| `GND` | GND |
| `SDA` | GPIO 26 |
| `SCL` | GPIO 27 |

The sketch uses `LiquidCrystal_I2C lcd(0x27, 16, 2)` and
`Wire.begin(26, 27)`. If your LCD is blank:

1. Run an I2C scanner and confirm the address (`0x27` or `0x3F`).
2. Change `LCD_I2C_ADDRESS` in the sketch.
3. Adjust the contrast potentiometer on the backpack.
4. Confirm SDA/SCL are not swapped.

---

## Buzzer

| Buzzer | ESP32 |
|---|---|
| `+` / signal | GPIO 33 |
| `-` | GND |

The firmware assumes a **passive** buzzer because it drives `tone()` at
different frequencies. If you have an active buzzer, it will still beep, but the
frequency sequences will not sound different.

### Sequences

| Sequence | When | Pattern |
|---|---|---|
| `FULL_SEQ` | Authorized but parking slot is occupied | 1 kHz for 300 ms, then 1 s silence |
| `DENIED_SEQ` | Repeated unauthorized attempt (`/buzzer`) | Four 2.5 kHz beeps, 200 ms on / 150 ms off, then a longer pause |

The first unauthorized attempt does not buzz. The Python service calls
`/buzzer` only after `PARKGUARD_BUZZER_AFTER_ATTEMPTS` (default `2`).

---

## Power distribution

```text
External 5 V supply
├── Servo red
├── HC-SR04 #1 VCC
├── HC-SR04 #2 VCC
├── LCD VCC
└── GND ────────────────┐
                        │
ESP32 GND ──────────────┘  (common ground)
ESP32 3.3 V ── logic only (never servo power)
```

- Use a supply rated for the servo stall current. An MG996R can draw over 1 A
  briefly.
- Keep the ESP32 powered from USB or a separate 5 V regulator.
- Route servo power wires away from the ultrasonic signal wires to reduce noise.

---

## Firmware configuration

1. Copy `firmware/parkguard_esp32/secrets.example.h` to
   `firmware/parkguard_esp32/secrets.h`.
2. Set `WIFI_SSID`, `WIFI_PASSWORD`, and `PARKGUARD_SERVER_IP`.
   `PARKGUARD_SERVER_IP` must be the **LAN IP of the computer running
   `main.py`**, not `127.0.0.1`.
3. Install **ESP32Servo** and **LiquidCrystal_I2C** from the Arduino Library
   Manager.
4. Select **ESP32 Dev Module**, upload, and open Serial Monitor at `115200`.
5. Confirm the LCD shows the ESP32 IP, then start `python main.py` on the host.

`secrets.h` is git-ignored. Never commit real Wi-Fi credentials or a LAN IP.

---

## Bring-up checklist

- [ ] ESP32 boots and prints the Wi-Fi IP on the LCD.
- [ ] `GET http://<esp32-ip>/status` returns both sensor distances.
- [ ] Parking sensor reports `parking_full: true` when an object is within 20 cm.
- [ ] `GET http://<esp32-ip>/servo?action=open` opens the gate when the slot is free.
- [ ] The gate stays closed and `FULL_SEQ` plays when the slot is occupied.
- [ ] `GET /buzzer` plays the repeated-attempt alarm.
- [ ] `GET /greet?name=Alice` shows a welcome message.
- [ ] The Python host can reach the ESP32 and the ESP32 can reach port `5000`
      on the Python host.
- [ ] The first unauthorized attempt logs without a buzzer; the second triggers
      `/buzzer` and `/lockdown`.
- [ ] An authorized scan clears lockdown and opens the gate.
