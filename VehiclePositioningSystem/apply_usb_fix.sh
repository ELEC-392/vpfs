#!/usr/bin/env bash
# apply_usb_fix.sh — run once with sudo to permanently disable USB autosuspend
# for Logitech Brio cameras.  Two complementary approaches:
#   1. Kernel boot parameter (survives reboots, affects all USB)
#   2. Systemd service (finer-grained, Brio-only, takes effect immediately)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Step 1: Kernel boot parameter (usbcore.autosuspend=-1) ==="
GRUB_FILE=/etc/default/grub
if grep -q "usbcore.autosuspend" "$GRUB_FILE"; then
    echo "  Parameter already present in $GRUB_FILE — skipping."
else
    sed -i 's/\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"/\1 usbcore.autosuspend=-1"/' "$GRUB_FILE"
    update-grub
    echo "  Added usbcore.autosuspend=-1 to GRUB_CMDLINE_LINUX_DEFAULT."
    echo "  This takes effect after the next reboot."
fi

echo ""
echo "=== Step 2: Brio autosuspend systemd service (immediate) ==="
cp "$SCRIPT_DIR/brio-autosuspend.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable brio-autosuspend.service
systemctl start  brio-autosuspend.service
echo "  Service installed and started."

echo ""
echo "=== Step 3: Reload udev rules ==="
cp "$SCRIPT_DIR/99-brio-camera.rules" /etc/udev/rules.d/
udevadm control --reload-rules
udevadm trigger --action=add --subsystem-match=usb
echo "  Udev rules reloaded."

echo ""
echo "=== Done. Verify with: ==="
echo "  systemctl status brio-autosuspend.service"
echo "  cat /sys/bus/usb/devices/*/power/control 2>/dev/null | sort -u"
