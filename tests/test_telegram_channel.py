from __future__ import annotations

import unittest

from calosum.adapters.channel_telegram import TelegramChannelAdapter


class TelegramPolicyTests(unittest.TestCase):
    def _adapter_with_policy(self, dm_policy: str, allowlist: list[str]) -> TelegramChannelAdapter:
        adapter = object.__new__(TelegramChannelAdapter)
        adapter.dm_policy = dm_policy
        adapter.allowlist_ids = set(allowlist)
        return adapter

    def test_open_policy_allows_any_sender(self) -> None:
        adapter = self._adapter_with_policy("open", [])
        self.assertTrue(adapter._is_sender_allowed("123"))
        self.assertTrue(adapter._is_sender_allowed("456"))

    def test_allowlist_policy_blocks_unknown_sender(self) -> None:
        adapter = self._adapter_with_policy("allowlist", ["1001", "1002"])
        self.assertTrue(adapter._is_sender_allowed("1001"))
        self.assertFalse(adapter._is_sender_allowed("9999"))


if __name__ == "__main__":
    unittest.main()
