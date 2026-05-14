from __future__ import annotations

import unittest

from config import resolve_camera_source


class RuntimeCameraSourceTests(unittest.TestCase):
    def test_numeric_string_camera_source_becomes_int(self) -> None:
        self.assertEqual(resolve_camera_source("0"), 0)
        self.assertEqual(resolve_camera_source(" 1 "), 1)

    def test_url_camera_source_stays_string(self) -> None:
        url = "http://192.168.1.50:4747/video"
        self.assertEqual(resolve_camera_source(url), url)


if __name__ == "__main__":
    unittest.main()
