import sys
from calosum.domain.telemetry import OTLPJsonlTelemetrySink

sink = OTLPJsonlTelemetrySink(".calosum-runtime/telemetry/events.jsonl")
events = sink._read_persisted_events()
print(f"Read {len(events)} events.")
for event in events:
    print(event.session_id, event.channel)
