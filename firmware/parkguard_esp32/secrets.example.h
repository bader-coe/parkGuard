/*
 * Copy this file to secrets.h in the same folder, then fill in your own values.
 *
 * secrets.h is listed in .gitignore and must never be committed.
 * The ESP32 and the computer running main.py must be on the same LAN.
 */
#pragma once

#define WIFI_SSID "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"

// LAN address of the computer running: python main.py
// Do not use "localhost" or 127.0.0.1: the ESP32 needs the computer's LAN IP.
#define PARKGUARD_SERVER_IP "192.0.2.10"
#define PARKGUARD_SERVER_PORT 5000
