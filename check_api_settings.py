from calosum.bootstrap.api import get_settings, get_builder
settings = get_settings()
print(f"Profile: {settings.profile}")
print(f"OTLP: {settings.otlp_jsonl}")
builder = get_builder()
bus = builder.build_telemetry_bus()
print(f"Bus: {type(bus.sink).__name__}")
