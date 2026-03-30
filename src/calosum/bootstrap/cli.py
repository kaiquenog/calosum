from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calosum.bootstrap.factory import CalosumAgentBuilder
from calosum.domain.metacognition import CognitiveVariantSpec
from calosum.shared.serialization import to_json, to_primitive
from calosum.bootstrap.settings import (
    InfrastructureProfile,
    InfrastructureSettings,
    should_enable_local_persistence_defaults,
    with_local_persistence_defaults,
)
from calosum.shared.types import Modality, MultimodalSignal, UserTurn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calosum cognitive agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_turn = subparsers.add_parser("run-turn", help="Process a single user turn")
    run_turn.add_argument("--session-id", required=True)
    run_turn.add_argument("--text", required=True)
    run_turn.add_argument(
        "--signal-json",
        action="append",
        default=[],
        help="JSON object describing a multimodal signal; may be repeated",
    )
    run_turn.add_argument(
        "--variants-json",
        action="append",
        default=[],
        help="Optional cognitive variant JSON objects; if provided, group mode is used",
    )
    run_turn.add_argument("--sleep-mode", action="store_true")
    run_turn.add_argument(
        "--infra-profile",
        choices=[profile.value for profile in InfrastructureProfile],
    )
    run_turn.add_argument("--memory-dir")
    run_turn.add_argument("--otlp-jsonl")

    run_scenario = subparsers.add_parser("run-scenario", help="Process a scenario JSON file")
    run_scenario.add_argument("scenario_path")
    run_scenario.add_argument(
        "--infra-profile",
        choices=[profile.value for profile in InfrastructureProfile],
    )
    run_scenario.add_argument("--memory-dir")
    run_scenario.add_argument("--otlp-jsonl")

    sleep_cmd = subparsers.add_parser("sleep", help="Run sleep mode consolidation and DSPy Night Trainer")
    sleep_cmd.add_argument(
        "--infra-profile",
        choices=[profile.value for profile in InfrastructureProfile],
    )
    sleep_cmd.add_argument("--memory-dir")
    sleep_cmd.add_argument("--otlp-jsonl")

    idle_cmd = subparsers.add_parser("idle", help="Run endogenous goal generation (Background Foraging)")
    idle_cmd.add_argument(
        "--infra-profile",
        choices=[profile.value for profile in InfrastructureProfile],
    )
    idle_cmd.add_argument("--memory-dir")
    idle_cmd.add_argument("--otlp-jsonl")

    chat_cmd = subparsers.add_parser("chat", help="Start an interactive chat REPL")
    chat_cmd.add_argument("--session-id", default="terminal-session")
    chat_cmd.add_argument(
        "--infra-profile",
        choices=[profile.value for profile in InfrastructureProfile],
    )
    chat_cmd.add_argument("--memory-dir")
    chat_cmd.add_argument("--otlp-jsonl")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = _resolve_settings(args)
    builder = CalosumAgentBuilder(settings)
    agent = builder.build()

    if args.command == "run-turn":
        result = _handle_run_turn(agent, args)
        result["infrastructure"] = builder.describe()
        print(to_json(result))
        return 0

    if args.command == "chat":
        return _handle_chat(agent, args)

    if args.command == "sleep":
        return _handle_sleep(agent, args, builder)

    if args.command == "idle":
        return _handle_idle(agent, args)

    if args.command == "run-scenario":
        result = _handle_run_scenario(agent, Path(args.scenario_path))
        result["infrastructure"] = builder.describe()
        print(to_json(result))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _handle_idle(agent, args: argparse.Namespace) -> int:
    import sys
    print("Initiating endogenous goal generation (Background Foraging Mode)...")
    result = agent.idle_foraging()
    
    if hasattr(result, "selected_result"):
        turn_result = result.selected_result
    else:
        turn_result = result
        
    print(f"Foraging Reasoning: {turn_result.left_result.reasoning_summary}")
    for action in turn_result.left_result.actions:
        print(f"Action Executed: {action.action_type} - {action.payload}")
        
    print("Foraging cycle completed and memory updated.")
    return 0

def _handle_sleep(agent, args: argparse.Namespace, builder: CalosumAgentBuilder) -> int:
    import sys
    print("Running memory consolidation (Sleep Mode)...")
    report = agent.sleep_mode()
    print(f"Consolidated {report.episodes_considered} episodes.")
    print(f"Promoted {len(report.promoted_rules)} semantic rules.")
    
    print("\nStarting Night Trainer (DSPy optimization)...")
    trainer = builder.build_night_trainer()
    result = trainer.run_training_cycle()
    
    print(f"Night Trainer Result: {json.dumps(result, indent=2)}")
    
    return 0 if result.get("status") in ("success", "skipped") else 1

def _handle_run_turn(agent, args: argparse.Namespace) -> dict[str, Any]:
    user_turn = UserTurn(
        session_id=args.session_id,
        user_text=args.text,
        signals=[_signal_from_dict(json.loads(raw)) for raw in args.signal_json],
    )

    variants = [_variant_from_dict(json.loads(raw)) for raw in args.variants_json]
    if variants:
        group_result = agent.process_group_turn(user_turn, variants)
        payload: dict[str, Any] = {"group_result": to_primitive(group_result)}
    else:
        turn_result = agent.process_turn(user_turn)
        payload = {"turn_result": to_primitive(turn_result)}

    if args.sleep_mode:
        payload["sleep_mode"] = to_primitive(agent.sleep_mode())
        payload["dashboard"] = agent.cognitive_dashboard(args.session_id)

    return payload


def _handle_run_scenario(agent, scenario_path: Path) -> dict[str, Any]:
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    session_id = scenario["session_id"]
    outputs: list[dict[str, Any]] = []

    for turn_data in scenario.get("turns", []):
        user_turn = UserTurn(
            session_id=session_id,
            user_text=turn_data["text"],
            signals=[_signal_from_dict(item) for item in turn_data.get("signals", [])],
        )
        variants_data = turn_data.get("group_variants", [])
        if variants_data:
            group_result = agent.process_group_turn(
                user_turn,
                [_variant_from_dict(item) for item in variants_data],
            )
            outputs.append({"group_result": to_primitive(group_result)})
        else:
            outputs.append({"turn_result": to_primitive(agent.process_turn(user_turn))})

    payload: dict[str, Any] = {"session_id": session_id, "results": outputs}
    if scenario.get("sleep_mode"):
        payload["sleep_mode"] = to_primitive(agent.sleep_mode())
    payload["dashboard"] = agent.cognitive_dashboard(session_id)
    return payload


def _signal_from_dict(data: dict[str, Any]) -> MultimodalSignal:
    return MultimodalSignal(
        modality=Modality(data["modality"]),
        source=data["source"],
        payload=data.get("payload"),
        quality=data.get("quality", 1.0),
        metadata=data.get("metadata", {}),
    )


def _variant_from_dict(data: dict[str, Any]) -> CognitiveVariantSpec:
    return CognitiveVariantSpec(
        variant_id=data["variant_id"],
        tokenizer_overrides=data.get("tokenizer_overrides", {}),
        left_overrides=data.get("left_overrides", {}),
        notes=data.get("notes", []),
    )


def _handle_chat(agent, args: argparse.Namespace) -> int:
    import sys
    print(f"Calosum Interactive Chat. Session [{args.session_id}]. Type 'exit' to quit.")
    while True:
        try:
            text = input("> ")
            if text.strip().lower() in ("exit", "quit", "\\q"):
                break
            if not text.strip():
                continue
                
            user_turn = UserTurn(
                session_id=args.session_id,
                user_text=text,
                signals=[],
            )
            result = agent.process_turn(user_turn)
            
            if hasattr(result, "selected_result"):
                turn_result = result.selected_result
            else:
                turn_result = result
            
            # Feedback from logical processing and actions
            if turn_result.left_result.response_text:
                print(f"Calosum (Reasoning): {turn_result.left_result.response_text}")
                if "falhou" in turn_result.left_result.response_text.lower():
                    print(f"-> Diagnostic Trace: {turn_result.left_result.reasoning_summary}")
                
            for action in turn_result.left_result.actions:
                if action.action_type == "respond_text":
                    print(f"Calosum (Action): {action.payload.get('text', '')}")
                else:
                    print(f"[{action.action_type.upper()}] {action.payload}")
            print()
        except EOFError:
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error handling turn: {e}", file=sys.stderr)
            
    return 0


def _resolve_settings(args: argparse.Namespace) -> InfrastructureSettings:
    settings = InfrastructureSettings.from_sources(args=args)
    if args.command == "chat" and should_enable_local_persistence_defaults(settings, args=args):
        return with_local_persistence_defaults(settings)
    return settings


if __name__ == "__main__":
    raise SystemExit(main())
