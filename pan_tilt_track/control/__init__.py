"""Outer (vision) control loop: pixel error -> servo goal positions.

Import from the submodules directly (`.gain`, `.loop`) rather than this
package's namespace -- `.loop` pulls in dynamixel_sdk and ultralytics, which
`tests/test_gain.py` deliberately avoids needing.
"""
