#!/usr/bin/env bash
# apply_usb_fix.sh — run once with sudo to permanently disable USB autosuspend
# for Logitech Brio cameras.  Three complementary layers:
#   1. Kernel boot parameter (survives reboots, affects all USB)
#   2. PCI runtime PM disable (the real root cause — xHCI controller sleep)
#   3. Systemd service + udev rules (persistent across reboots)
#   4. xhci-reset helper + sudoers rule (software recovery of HC died crashes)

set -u  # error on undefined variables, but don't abort on command failures
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
echo "=== Step 2: Disable xHCI PCI runtime PM right now (immediate) ==="
for pcidev in /sys/bus/pci/devices/*/; do
    driver=$(readlink "$pcidev/driver" 2>/dev/null | xargs basename 2>/dev/null || true)
    if [ "$driver" = "xhci_hcd" ]; then
        echo on > "$pcidev/power/control"              2>/dev/null && echo "  on: $pcidev" || echo "  FAILED (control): $pcidev"
        echo -1 > "$pcidev/power/autosuspend_delay_ms" 2>/dev/null || true
    fi
done

echo ""
echo "=== Step 3: Disable USB root hub autosuspend right now ==="
for hub in /sys/bus/usb/devices/usb*/; do
    echo on > "$hub/power/control"              2>/dev/null && echo "  on: $hub" || echo "  FAILED: $hub"
    echo -1 > "$hub/power/autosuspend_delay_ms" 2>/dev/null || true
done

echo ""
echo "=== Step 4: Brio autosuspend systemd service (persistent across reboots) ==="
cp "$SCRIPT_DIR/brio-autosuspend.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable brio-autosuspend.service
systemctl restart brio-autosuspend.service
echo "  Service installed, enabled and restarted."

echo ""
echo "=== Step 5: Reload udev rules ==="
cp "$SCRIPT_DIR/99-brio-camera.rules" /etc/udev/rules.d/
udevadm control --reload-rules
udevadm trigger --action=add --subsystem-match=pci
udevadm trigger --action=add --subsystem-match=usb
echo "  Udev rules reloaded."

echo ""
echo "=== Step 6: xhci-reset helper + sudoers rule (for software recovery) ==="
# This lets the VPS Python process reset the xHCI controller after an
# 'HC died' crash, without requiring the script to run as root.
cat > /usr/local/bin/xhci-reset << 'EOF'
#!/bin/sh
# Reset all PCI devices bound to xhci_hcd.  Called by the VPS recovery logic
# after an xHCI host controller crash ("HC died; cleaning up" in dmesg).
for pcidev in /sys/bus/pci/drivers/xhci_hcd/*/; do
    reset="$pcidev/reset"
    if [ -f "$reset" ]; then
        echo 1 > "$reset" && echo "xhci-reset: reset $pcidev" || echo "xhci-reset: failed $pcidev"
    fi
done
EOF
chmod +x /usr/local/bin/xhci-reset

SUDOERS_FILE=/etc/sudoers.d/xhci-reset
SUDOERS_LINE="%sudo ALL=(root) NOPASSWD: /usr/local/bin/xhci-reset"
if [ -f "$SUDOERS_FILE" ] && grep -qF "$SUDOERS_LINE" "$SUDOERS_FILE" 2>/dev/null; then
    echo "  Sudoers rule already present — skipping."
else
    echo "$SUDOERS_LINE" > "$SUDOERS_FILE"
    chmod 440 "$SUDOERS_FILE"
    echo "  Installed sudoers rule: any member of 'sudo' group can run xhci-reset without password."
fi

echo ""
echo "=== Verify ==="
echo "  xHCI PCI power/control:"
for pcidev in /sys/bus/pci/devices/*/; do
    driver=$(readlink "$pcidev/driver" 2>/dev/null | xargs basename 2>/dev/null || true)
    [ "$driver" = "xhci_hcd" ] && echo "    $(cat $pcidev/power/control 2>/dev/null)  $pcidev" || true
done
echo "  USB root hub power/control:"
for hub in /sys/bus/usb/devices/usb*/; do
    echo "    $(cat $hub/power/control 2>/dev/null)  $hub"
done
