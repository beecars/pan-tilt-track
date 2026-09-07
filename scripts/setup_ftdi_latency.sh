#!/usr/bin/env bash
# Fixes the FTDI (U2D2) latency timer, which defaults to 16ms and caps
# round-trip rate near 60Hz regardless of configured baud. Installs a udev
# rule so it survives reboots/replugs, and adds the invoking user to
# dialout so the control process needn't run as root.
set -euo pipefail

RULE_PATH="/etc/udev/rules.d/99-ftdi-latency.rules"

sudo tee "$RULE_PATH" > /dev/null <<'EOF'
ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

if [ -e /dev/ttyUSB0 ]; then
    echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer > /dev/null
    echo "Applied immediately to ttyUSB0: $(cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer)ms"
fi

sudo usermod -aG dialout "$USER"
echo "Added $USER to dialout group (log out/in for it to take effect)."
echo "udev rule installed at $RULE_PATH -- future U2D2 plug-ins will get 1ms automatically."
