import json
import os
from typing import Dict, List

import requests

from agent.tool.tool_base import Tool


class HarnessGTool(Tool):
    """Stateful Harness-G action tool using the legacy <query> JSON container."""

    is_harness_g = True

    def __init__(self):
        name = "search"
        description = (
            "Harness-G evidence navigation tool. The query must be INIT or an available "
            "action id such as A0. Use SELECT to keep a useful sentence as evidence; use "
            "LOOKUP to find missing info about a listed entity (just choose the LOOKUP "
            "action id; the retrieval query is built for you); "
            "use ANSWER_WITH to keep the listed sentence(s) and finish; "
            "use ANSWER when the selected evidence is enough. Do not invent entity names or "
            "look up dates / nationalities / known attribute values. Do not append || queries."
        )
        query_param_desc = "Harness-G action: INIT, A0, A1 (action id only, no || query)"
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": query_param_desc,
                }
            },
            "required": ["query"],
        }
        super().__init__(name, description, parameters)
        self.search_api_url = os.environ.get("HARNESS_G_API_URL", "http://localhost:8001/harness_g_step")
        self.timeout = float(os.environ.get("HARNESS_G_API_TIMEOUT", "300"))
        self.session = requests.Session()
        self.session.trust_env = False

    def execute(self, args: Dict) -> str:
        return self.batch_execute([args])[0]

    @staticmethod
    def _unwrap_api_result(item) -> str:
        if isinstance(item, dict):
            if "observation" in item:
                return str(item.get("observation", ""))
            return str(item.get("results", ""))
        if isinstance(item, str):
            text = item
            try:
                parsed = json.loads(text)
            except Exception:
                return text
            if isinstance(parsed, dict) and "results" in parsed:
                return str(parsed.get("results", ""))
            return text
        return json.dumps(item, ensure_ascii=False)

    @staticmethod
    def _extract_snc_step(item):
        """Return the optional SNC side-channel payload from an API result item."""
        if isinstance(item, dict):
            return item.get("snc_step")
        return None

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        requests_payload = [
            {
                "session_id": args.get("session_id", f"legacy_{idx}"),
                "question": args.get("question", ""),
                "action": args.get("query", ""),
            }
            for idx, args in enumerate(args_list)
        ]
        response = self.session.post(self.search_api_url, json={"requests": requests_payload}, timeout=self.timeout)
        response.raise_for_status()
        results = response.json()
        return [self._unwrap_api_result(item) for item in results]

    def batch_execute_with_envs(self, args_list: List[Dict], envs_list: List[object]) -> List[str]:
        requests_payload = []
        for idx, (args, env) in enumerate(zip(args_list, envs_list)):
            requests_payload.append(
                {
                    "session_id": getattr(env, "session_id", None) or args.get("session_id") or f"env_{idx}",
                    "question": getattr(env, "question", None) or args.get("question", ""),
                    "action": args.get("query", ""),
                }
            )
        response = self.session.post(self.search_api_url, json={"requests": requests_payload}, timeout=self.timeout)
        response.raise_for_status()
        results = response.json()
        if not isinstance(results, list):
            raise RuntimeError(f"Harness-G API returned non-list response: {results}")
        observations = []
        for item, env in zip(results, envs_list):
            observations.append(self._unwrap_api_result(item))
            try:
                # Stash the optional SNC side-channel on the env so step_batch can
                # forward it via the info dict. None when SNC is disabled.
                env._snc_last_step = self._extract_snc_step(item)
            except Exception:
                pass
        return observations
